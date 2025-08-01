"""
Advanced Language Processing Chain
Author: Drmusab
Last Modified: 2025-05-26 21:15:00 UTC

This module provides comprehensive language processing capabilities integrated with the
AI assistant's core architecture, including LLM integration, prompt engineering,
context management, and response generation with advanced reasoning capabilities.
"""

import hashlib
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Protocol, Type, Union

import asyncio
import numpy as np
import tiktoken
import torch
from transformers import AutoModel, AutoTokenizer

from src.assistant.interaction_handler import InteractionContext, UserProfile

# Assistant core
from src.assistant.session_manager import SessionManager

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentFailed,
    ComponentHealthChanged,
    ComponentInitialized,
    ComponentRegistered,
    ComponentStarted,
    ComponentStopped,
)
from src.core.health_check import HealthCheck
from src.integrations.cache.cache_strategy import CacheStrategy

# Integration imports
from src.integrations.llm.base_provider import BaseLLMProvider
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager
from src.learning.feedback_processor import FeedbackProcessor

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.processing.natural_language.entity_extractor import EntityExtractor

# Processing imports
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer


class ProcessingMode(Enum):
    """Language processing modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    STREAMING = "streaming"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


class ResponseType(Enum):
    """Types of language responses."""

    ANSWER = "answer"
    QUESTION = "question"
    CLARIFICATION = "clarification"
    INSTRUCTION = "instruction"
    EXPLANATION = "explanation"
    SUMMARY = "summary"
    ANALYSIS = "analysis"
    CREATIVE_CONTENT = "creative_content"
    CODE_GENERATION = "code_generation"
    TASK_COMPLETION = "task_completion"


class PromptTemplate(Enum):
    """Built-in prompt templates."""

    GENERAL_CONVERSATION = "general_conversation"
    QUESTION_ANSWERING = "question_answering"
    TASK_ASSISTANCE = "task_assistance"
    CREATIVE_WRITING = "creative_writing"
    CODE_ASSISTANCE = "code_assistance"
    ANALYSIS_REQUEST = "analysis_request"
    LEARNING_SUPPORT = "learning_support"
    PROBLEM_SOLVING = "problem_solving"


class ReasoningStrategy(Enum):
    """Reasoning strategies for complex tasks."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    STEP_BY_STEP = "step_by_step"
    ANALOGICAL = "analogical"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"


@dataclass
class LanguageRequest:
    """Comprehensive language processing request."""

    text: str
    mode: ProcessingMode = ProcessingMode.CONVERSATIONAL
    response_type: ResponseType = ResponseType.ANSWER

    # Context and session
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    interaction_context: Optional[InteractionContext] = None

    # Processing options
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    enable_reasoning: bool = False
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT

    # Personalization
    user_profile: Optional[UserProfile] = None
    style_preferences: Dict[str, Any] = field(default_factory=dict)

    # Quality and safety
    enable_content_filtering: bool = True
    enable_fact_checking: bool = False
    quality_threshold: float = 0.8

    # Advanced options
    custom_prompt_template: Optional[str] = None
    system_instructions: Optional[str] = None
    constraints: List[str] = field(default_factory=list)

    # Performance
    cache_response: bool = True
    priority: int = 1  # 1-5 scale
    timeout_seconds: float = 30.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class LanguageResponse:
    """Comprehensive language processing response."""

    text: str
    response_type: ResponseType
    confidence: float = 0.0

    # Processing information
    processing_time: float = 0.0
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    factuality_score: float = 0.0
    safety_score: float = 0.0

    # Reasoning and sources
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)

    # Enhanced analysis
    intent_analysis: Optional[Dict[str, Any]] = None
    sentiment_analysis: Optional[Dict[str, Any]] = None
    entity_extraction: Optional[List[Dict[str, Any]]] = None

    # Session and context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Follow-up suggestions
    suggested_follow_ups: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    warnings: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for ongoing conversations."""

    conversation_id: str
    participant_ids: List[str]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context_summary: str = ""
    current_topic: Optional[str] = None
    conversation_stage: str = "ongoing"  # starting, ongoing, concluding

    # Memory and knowledge
    relevant_memories: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_context: Dict[str, Any] = field(default_factory=dict)

    # Preferences and style
    style_adaptations: Dict[str, Any] = field(default_factory=dict)
    conversation_goals: List[str] = field(default_factory=list)

    # Temporal information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_minutes: float = 0.0


@dataclass
class PromptEngineering:
    """Advanced prompt engineering configuration."""

    base_template: str = ""
    system_instructions: str = ""
    context_injection_strategy: str = "prepend"  # prepend, append, interleave

    # Dynamic prompt components
    persona_instructions: Optional[str] = None
    task_specific_guidance: Optional[str] = None
    output_format_specification: Optional[str] = None

    # Advanced techniques
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    chain_of_thought_examples: List[str] = field(default_factory=list)
    constraint_specifications: List[str] = field(default_factory=list)

    # Dynamic adaptation
    difficulty_adaptation: bool = True
    user_level_adjustment: bool = True
    context_aware_formatting: bool = True


class LanguageChainError(Exception):
    """Custom exception for language chain operations."""

    def __init__(
        self, message: str, error_code: Optional[str] = None, component: Optional[str] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.component = component
        self.timestamp = datetime.now(timezone.utc)


class PromptManager:
    """Advanced prompt management and engineering."""

    def __init__(self, logger, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        self.templates: Dict[str, str] = {}
        self.dynamic_components: Dict[str, Callable] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        self.templates = {
            PromptTemplate.GENERAL_CONVERSATION.value: (
                "You are a helpful, knowledgeable, and friendly AI assistant. "
                "Provide accurate, relevant, and engaging responses to user queries. "
                "Consider the context and adapt your communication style appropriately.\n\n"
                "Context: {context}\n"
                "User: {user_input}\n"
                "Assistant:"
            ),
            PromptTemplate.QUESTION_ANSWERING.value: (
                "You are an expert knowledge assistant. Provide accurate, comprehensive, "
                "and well-structured answers to questions. Include relevant context and "
                "cite sources when possible.\n\n"
                "Question: {user_input}\n"
                "Context: {context}\n\n"
                "Please provide a detailed answer:"
            ),
            PromptTemplate.TASK_ASSISTANCE.value: (
                "You are a task-oriented assistant specialized in helping users complete "
                "specific tasks efficiently. Break down complex tasks into manageable steps "
                "and provide clear, actionable guidance.\n\n"
                "Task: {user_input}\n"
                "Context: {context}\n"
                "User Profile: {user_profile}\n\n"
                "Step-by-step assistance:"
            ),
            PromptTemplate.CREATIVE_WRITING.value: (
                "You are a creative writing assistant with expertise in various literary "
                "forms and styles. Help users with creative projects while maintaining "
                "originality and quality.\n\n"
                "Creative Request: {user_input}\n"
                "Style Preferences: {style_preferences}\n"
                "Context: {context}\n\n"
                "Creative response:"
            ),
            PromptTemplate.CODE_ASSISTANCE.value: (
                "You are a programming assistant with expertise in multiple languages "
                "and best practices. Provide clean, efficient, and well-documented code "
                "solutions with explanations.\n\n"
                "Programming Request: {user_input}\n"
                "Context: {context}\n"
                "Requirements: {constraints}\n\n"
                "Code solution with explanation:"
            ),
            PromptTemplate.ANALYSIS_REQUEST.value: (
                "You are an analytical expert capable of deep analysis and insight "
                "generation. Provide thorough, objective, and well-reasoned analysis "
                "with supporting evidence.\n\n"
                "Analysis Subject: {user_input}\n"
                "Context: {context}\n"
                "Analysis Focus: {analysis_focus}\n\n"
                "Detailed analysis:"
            ),
            PromptTemplate.LEARNING_SUPPORT.value: (
                "You are an educational assistant focused on facilitating learning and "
                "understanding. Adapt explanations to the user's level and provide "
                "engaging, clear instruction.\n\n"
                "Learning Topic: {user_input}\n"
                "User Level: {user_level}\n"
                "Learning Goals: {learning_goals}\n"
                "Context: {context}\n\n"
                "Educational response:"
            ),
            PromptTemplate.PROBLEM_SOLVING.value: (
                "You are a problem-solving expert skilled in analytical thinking and "
                "solution development. Break down complex problems and provide "
                "systematic approaches to solutions.\n\n"
                "Problem: {user_input}\n"
                "Context: {context}\n"
                "Constraints: {constraints}\n"
                "Reasoning Strategy: {reasoning_strategy}\n\n"
                "Problem analysis and solution:"
            ),
        }

    def build_prompt(
        self,
        request: LanguageRequest,
        conversation_context: Optional[ConversationContext] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build a comprehensive prompt from request and context."""
        try:
            # Select appropriate template
            template = self._select_template(request)

            # Prepare context variables
            context_vars = self._prepare_context_variables(
                request, conversation_context, additional_context
            )

            # Apply dynamic enhancements
            enhanced_template = self._apply_dynamic_enhancements(template, request)

            # Format the final prompt
            prompt = enhanced_template.format(**context_vars)

            # Apply post-processing
            prompt = self._post_process_prompt(prompt, request)

            return prompt

        except Exception as e:
            self.logger.error(f"Prompt building failed: {str(e)}")
            return self._build_fallback_prompt(request)

    def _select_template(self, request: LanguageRequest) -> str:
        """Select appropriate prompt template based on request."""
        if request.custom_prompt_template:
            return request.custom_prompt_template

        # Map processing mode to template
        mode_template_map = {
            ProcessingMode.CONVERSATIONAL: PromptTemplate.GENERAL_CONVERSATION,
            ProcessingMode.REASONING: PromptTemplate.PROBLEM_SOLVING,
            ProcessingMode.CREATIVE: PromptTemplate.CREATIVE_WRITING,
            ProcessingMode.ANALYTICAL: PromptTemplate.ANALYSIS_REQUEST,
        }

        template_key = mode_template_map.get(
            request.mode, PromptTemplate.GENERAL_CONVERSATION
        ).value

        return self.templates.get(
            template_key, self.templates[PromptTemplate.GENERAL_CONVERSATION.value]
        )

    def _prepare_context_variables(
        self,
        request: LanguageRequest,
        conversation_context: Optional[ConversationContext],
        additional_context: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Prepare all context variables for prompt formatting."""
        variables = {
            "user_input": request.text,
            "context": "",
            "user_profile": "",
            "style_preferences": "",
            "constraints": ", ".join(request.constraints),
            "reasoning_strategy": request.reasoning_strategy.value,
            "analysis_focus": request.metadata.get("analysis_focus", "general"),
            "user_level": request.metadata.get("user_level", "intermediate"),
            "learning_goals": request.metadata.get("learning_goals", "understanding"),
        }

        # Add conversation context
        if conversation_context:
            variables["context"] = self._format_conversation_context(conversation_context)

        # Add user profile information
        if request.user_profile:
            variables["user_profile"] = self._format_user_profile(request.user_profile)

        # Add style preferences
        if request.style_preferences:
            variables["style_preferences"] = self._format_style_preferences(
                request.style_preferences
            )

        # Add additional context
        if additional_context:
            for key, value in additional_context.items():
                if key not in variables:
                    variables[key] = str(value)

        return variables

    def _apply_dynamic_enhancements(self, template: str, request: LanguageRequest) -> str:
        """Apply dynamic enhancements to the template."""
        enhanced = template

        # Add system instructions if provided
        if request.system_instructions:
            enhanced = f"{request.system_instructions}\n\n{enhanced}"

        # Add reasoning instructions if enabled
        if request.enable_reasoning:
            reasoning_instruction = self._get_reasoning_instruction(request.reasoning_strategy)
            enhanced = f"{enhanced}\n\n{reasoning_instruction}"

        return enhanced

    def _get_reasoning_instruction(self, strategy: ReasoningStrategy) -> str:
        """Get reasoning instructions based on strategy."""
        instructions = {
            ReasoningStrategy.CHAIN_OF_THOUGHT: (
                "Think through this step by step, showing your reasoning process clearly."
            ),
            ReasoningStrategy.TREE_OF_THOUGHTS: (
                "Consider multiple approaches and evaluate the best path forward."
            ),
            ReasoningStrategy.STEP_BY_STEP: ("Break this down into clear, sequential steps."),
            ReasoningStrategy.ANALOGICAL: (
                "Use analogies and comparisons to explain your reasoning."
            ),
        }
        return instructions.get(strategy, instructions[ReasoningStrategy.CHAIN_OF_THOUGHT])

    def _format_conversation_context(self, context: ConversationContext) -> str:
        """Format conversation context for prompt inclusion."""
        if not context.messages:
            return "No previous conversation history."

        # Include recent messages (last 5)
        recent_messages = context.messages[-5:]
        formatted_messages = []

        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate long messages
            formatted_messages.append(f"{role}: {content}")

        context_str = "\n".join(formatted_messages)

        if context.current_topic:
            context_str = f"Current topic: {context.current_topic}\n{context_str}"

        return context_str

    def _format_user_profile(self, profile: UserProfile) -> str:
        """Format user profile for prompt inclusion."""
        profile_items = []

        if profile.preferred_language:
            profile_items.append(f"Language: {profile.preferred_language}")

        if profile.learning_style:
            profile_items.append(f"Learning style: {profile.learning_style}")

        if profile.interaction_patterns:
            communication_style = profile.interaction_patterns.get("communication_style", "")
            if communication_style:
                profile_items.append(f"Communication style: {communication_style}")

        return ", ".join(profile_items) if profile_items else "No specific profile information"

    def _format_style_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format style preferences for prompt inclusion."""
        if not preferences:
            return "No specific style preferences"

        pref_items = []
        for key, value in preferences.items():
            pref_items.append(f"{key}: {value}")

        return ", ".join(pref_items)

    def _post_process_prompt(self, prompt: str, request: LanguageRequest) -> str:
        """Apply post-processing to the prompt."""
        # Ensure prompt doesn't exceed reasonable length
        max_prompt_length = 4000  # Adjust based on model context window
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."

        return prompt

    def _build_fallback_prompt(self, request: LanguageRequest) -> str:
        """Build a simple fallback prompt."""
        return f"Please respond to the following: {request.text}"


class ResponseProcessor:
    """Advanced response processing and enhancement."""

    def __init__(
        self,
        logger,
        intent_manager: IntentManager,
        sentiment_analyzer: SentimentAnalyzer,
        entity_extractor: EntityExtractor,
    ):
        self.logger = logger
        self.intent_manager = intent_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.entity_extractor = entity_extractor

    async def process_response(
        self, raw_response: str, request: LanguageRequest, processing_metadata: Dict[str, Any]
    ) -> LanguageResponse:
        """Process and enhance the raw LLM response."""
        try:
            # Create base response object
            response = LanguageResponse(
                text=raw_response,
                response_type=request.response_type,
                session_id=request.session_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                processing_time=processing_metadata.get("processing_time", 0.0),
                model_used=processing_metadata.get("model_used", ""),
                prompt_tokens=processing_metadata.get("prompt_tokens", 0),
                completion_tokens=processing_metadata.get("completion_tokens", 0),
                total_tokens=processing_metadata.get("total_tokens", 0),
            )

            # Enhanced analysis
            await self._enhance_with_analysis(response, request)

            # Quality assessment
            await self._assess_response_quality(response, request)

            # Generate follow-up suggestions
            await self._generate_follow_ups(response, request)

            # Extract reasoning steps if present
            await self._extract_reasoning_steps(response)

            return response

        except Exception as e:
            self.logger.error(f"Response processing failed: {str(e)}")
            # Return basic response on error
            return LanguageResponse(
                text=raw_response,
                response_type=request.response_type,
                warnings=[f"Processing enhancement failed: {str(e)}"],
            )

    async def _enhance_with_analysis(
        self, response: LanguageResponse, request: LanguageRequest
    ) -> None:
        """Enhance response with intent, sentiment, and entity analysis."""
        try:
            # Intent analysis on the original request
            if request.text:
                intent_result = await self.intent_manager.detect_intent(request.text)
                response.intent_analysis = {
                    "detected_intent": intent_result.intent_name,
                    "confidence": intent_result.confidence,
                    "entities": [entity.to_dict() for entity in intent_result.entities],
                }

            # Sentiment analysis on the response
            sentiment_result = await self.sentiment_analyzer.analyze_sentiment(response.text)
            response.sentiment_analysis = {
                "sentiment": sentiment_result.sentiment,
                "confidence": sentiment_result.confidence,
                "scores": sentiment_result.scores,
            }

            # Entity extraction from response
            entities = await self.entity_extractor.extract_entities(response.text)
            response.entity_extraction = [entity.to_dict() for entity in entities]

        except Exception as e:
            self.logger.warning(f"Analysis enhancement failed: {str(e)}")
            response.warnings.append(f"Analysis enhancement failed: {str(e)}")

    async def _assess_response_quality(
        self, response: LanguageResponse, request: LanguageRequest
    ) -> None:
        """Assess various quality metrics of the response."""
        try:
            # Basic quality metrics
            response.coherence_score = self._assess_coherence(response.text)
            response.relevance_score = self._assess_relevance(response.text, request.text)
            response.safety_score = self._assess_safety(response.text)

            # Overall confidence based on multiple factors
            response.confidence = (
                response.coherence_score * 0.3
                + response.relevance_score * 0.4
                + response.safety_score * 0.3
            )

        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {str(e)}")
            response.warnings.append(f"Quality assessment failed: {str(e)}")

    def _assess_coherence(self, text: str) -> float:
        """Assess the coherence of the response text."""
        # Simple coherence assessment
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.8  # Short responses are generally coherent

        # Check for basic coherence indicators
        coherence_indicators = [
            len(text) > 10,  # Reasonable length
            not text.startswith("I'm sorry"),  # Not just an apology
            "?" in text or "." in text,  # Proper punctuation
            len(sentences) > 0,  # Has sentences
        ]

        return sum(coherence_indicators) / len(coherence_indicators)

    def _assess_relevance(self, response_text: str, request_text: str) -> float:
        """Assess relevance of response to the request."""
        # Simple keyword overlap assessment
        request_words = set(request_text.lower().split())
        response_words = set(response_text.lower().split())

        if not request_words:
            return 0.5

        overlap = len(request_words.intersection(response_words))
        return min(overlap / len(request_words), 1.0)

    def _assess_safety(self, text: str) -> float:
        """Assess the safety of the response content."""
        # Basic safety assessment
        unsafe_indicators = [
            "harmful",
            "dangerous",
            "illegal",
            "violence",
            "hate",
            "discriminatory",
            "offensive",
            "inappropriate",
        ]

        text_lower = text.lower()
        unsafe_count = sum(1 for indicator in unsafe_indicators if indicator in text_lower)

        # Return safety score (1.0 is completely safe)
        return max(1.0 - (unsafe_count * 0.2), 0.0)

    async def _generate_follow_ups(
        self, response: LanguageResponse, request: LanguageRequest
    ) -> None:
        """Generate follow-up suggestions based on the response."""
        try:
            follow_ups = []

            # Based on response type
            if request.response_type == ResponseType.ANSWER:
                follow_ups.extend(
                    [
                        "Would you like me to elaborate on any specific part?",
                        "Do you have any follow-up questions?",
                        "Is there anything else you'd like to know about this topic?",
                    ]
                )
            elif request.response_type == ResponseType.INSTRUCTION:
                follow_ups.extend(
                    [
                        "Would you like me to break down any of these steps further?",
                        "Do you need help with implementing any specific step?",
                        "Are there any prerequisites I should explain?",
                    ]
                )
            elif request.response_type == ResponseType.ANALYSIS:
                follow_ups.extend(
                    [
                        "Would you like me to analyze any specific aspect in more detail?",
                        "Should I provide examples to illustrate these points?",
                        "Do you want me to compare this with alternatives?",
                    ]
                )

            # Based on detected entities or topics
            if response.entity_extraction:
                entities = [entity["text"] for entity in response.entity_extraction[:3]]
                for entity in entities:
                    follow_ups.append(f"Would you like to know more about {entity}?")

            response.suggested_follow_ups = follow_ups[:5]  # Limit to 5 suggestions

        except Exception as e:
            self.logger.warning(f"Follow-up generation failed: {str(e)}")

    async def _extract_reasoning_steps(self, response: LanguageResponse) -> None:
        """Extract reasoning steps from the response if present."""
        try:
            text = response.text
            reasoning_steps = []

            # Look for numbered steps
            lines = text.split("\n")
            current_step = None
            step_content = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if line starts with a number or step indicator
                if (
                    line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))
                    or line.startswith(("Step 1", "Step 2", "Step 3", "Step 4", "Step 5"))
                    or line.startswith(("First,", "Second,", "Third,", "Finally,"))
                ):

                    # Save previous step if exists
                    if current_step and step_content:
                        reasoning_steps.append(
                            {
                                "step": current_step,
                                "content": " ".join(step_content),
                                "type": "reasoning",
                            }
                        )

                    # Start new step
                    current_step = line
                    step_content = []
                else:
                    step_content.append(line)

            # Add final step
            if current_step and step_content:
                reasoning_steps.append(
                    {"step": current_step, "content": " ".join(step_content), "type": "reasoning"}
                )

            response.reasoning_steps = reasoning_steps

        except Exception as e:
            self.logger.warning(f"Reasoning extraction failed: {str(e)}")


class ContextManager:
    """Advanced context management for conversations and sessions."""

    def __init__(self, logger, memory_manager: MemoryManager, session_manager: SessionManager):
        self.logger = logger
        self.memory_manager = memory_manager
        self.session_manager = session_manager
        self.conversations: Dict[str, ConversationContext] = {}
        self.context_lock = threading.Lock()

    async def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get or create conversation context."""
        with self.context_lock:
            if conversation_id not in self.conversations:
                # Try to load from memory
                context = await self._load_conversation_from_memory(conversation_id)
                if context:
                    self.conversations[conversation_id] = context
                else:
                    # Create new context
                    self.conversations[conversation_id] = ConversationContext(
                        conversation_id=conversation_id, participant_ids=[]
                    )

            return self.conversations[conversation_id]

    async def update_conversation_context(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Update conversation context with new exchange."""
        context = await self.get_conversation_context(conversation_id)
        if not context:
            return

        with self.context_lock:
            # Add messages
            context.messages.append(
                {
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.now(timezone.utc),
                    "user_id": user_id,
                }
            )

            context.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": datetime.now(timezone.utc),
                }
            )

            # Update metadata
            context.last_updated = datetime.now(timezone.utc)
            if user_id and user_id not in context.participant_ids:
                context.participant_ids.append(user_id)

            # Maintain reasonable context size
            if len(context.messages) > 50:  # Keep last 50 messages
                context.messages = context.messages[-50:]

            # Update context summary periodically
            if len(context.messages) % 10 == 0:
                await self._update_context_summary(context)

            # Store in memory
            await self._store_conversation_in_memory(context)

    async def get_relevant_context(
        self, request: LanguageRequest, max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """Get relevant context for processing a request."""
        context = {}

        try:
            # Get conversation context
            if request.conversation_id:
                conv_context = await self.get_conversation_context(request.conversation_id)
                if conv_context:
                    context["conversation"] = conv_context

            # Get relevant memories
            if request.user_id:
                memories = await self.memory_manager.retrieve_relevant_memories(
                    query=request.text, user_id=request.user_id, limit=5
                )
                context["memories"] = memories

            # Get session context
            if request.session_id:
                session_info = await self.session_manager.get_session(request.session_id)
                if session_info:
                    context["session"] = {
                        "user_preferences": session_info.context.user_preferences,
                        "interaction_history": session_info.context.interaction_history[-5:],
                        "current_topic": session_info.context.current_topic,
                    }

            return context

        except Exception as e:
            self.logger.error(f"Context retrieval failed: {str(e)}")
            return {}

    async def _load_conversation_from_memory(
        self, conversation_id: str
    ) -> Optional[ConversationContext]:
        """Load conversation context from memory system."""
        try:
            # Implementation depends on memory system structure
            # This is a placeholder for the actual implementation
            return None
        except Exception as e:
            self.logger.error(f"Failed to load conversation from memory: {str(e)}")
            return None

    async def _store_conversation_in_memory(self, context: ConversationContext) -> None:
        """Store conversation context in memory system."""
        try:
            # Implementation depends on memory system structure
            # This is a placeholder for the actual implementation
            pass
        except Exception as e:
            self.logger.error(f"Failed to store conversation in memory: {str(e)}")

    async def _update_context_summary(self, context: ConversationContext) -> None:
        """Update the conversation context summary."""
        try:
            if len(context.messages) < 5:
                return

            # Simple summarization - in practice, use a dedicated summarization model
            recent_messages = context.messages[-10:]
            topics = []

            for msg in recent_messages:
                content = msg.get("content", "")
                # Extract key topics (simplified)
                words = content.split()
                for word in words:
                    if len(word) > 5 and word.isalpha():
                        topics.append(word.lower())

            # Find most common topics
            topic_counts = defaultdict(int)
            for topic in topics:
                topic_counts[topic] += 1

            main_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            context.current_topic = main_topics[0][0] if main_topics else None

            # Create simple summary
            summary_parts = []
            if main_topics:
                topic_names = [topic for topic, _ in main_topics]
                summary_parts.append(f"Main topics: {', '.join(topic_names)}")

            summary_parts.append(f"Messages exchanged: {len(context.messages)}")
            context.context_summary = ". ".join(summary_parts)

        except Exception as e:
            self.logger.error(f"Context summary update failed: {str(e)}")


class EnhancedLanguageChain:
    """
    Advanced Language Processing Chain for the AI Assistant System.

    This class provides comprehensive natural language processing capabilities
    including LLM integration, prompt engineering, context management, response
    generation, and advanced reasoning. It integrates seamlessly with the
    assistant's core architecture.

    Features:
    - Multi-provider LLM integration with intelligent routing
    - Advanced prompt engineering with dynamic templates
    - Context-aware conversation management
    - Response quality assessment and enhancement
    - Reasoning and step-by-step problem solving
    - Caching and performance optimization
    - Learning and adaptation from user interactions
    - Comprehensive monitoring and observability
    """

    def __init__(self, container: Container):
        """
        Initialize the Enhanced Language Chain.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core components
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # LLM and routing
        self.model_router = container.get(ModelRouter)

        # Processing components
        self.intent_manager = container.get(IntentManager)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)
        self.entity_extractor = container.get(EntityExtractor)

        # Memory and learning
        self.memory_manager = container.get(MemoryManager)
        self.session_manager = container.get(SessionManager)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Setup core managers
        self.prompt_manager = PromptManager(self.logger, self.config.get("language_chain", {}))
        self.response_processor = ResponseProcessor(
            self.logger, self.intent_manager, self.sentiment_analyzer, self.entity_extractor
        )
        self.context_manager = ContextManager(
            self.logger, self.memory_manager, self.session_manager
        )

        # Setup infrastructure
        self._setup_monitoring()
        self._setup_caching()
        self._setup_performance_tracking()

        # Register health check
        self.health_check.register_component("language_chain", self._health_check_callback)

        self.logger.info("EnhancedLanguageChain initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)

            # Register language processing metrics
            self.metrics.register_counter("language_requests_total")
            self.metrics.register_counter("language_responses_total")
            self.metrics.register_histogram("language_processing_duration_seconds")
            self.metrics.register_histogram("language_response_length_tokens")
            self.metrics.register_gauge("language_response_quality_score")
            self.metrics.register_counter("language_errors_total")
            self.metrics.register_gauge("active_conversations")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    def _setup_caching(self) -> None:
        """Setup caching for responses and contexts."""
        try:
            self.cache_strategy = self.container.get(CacheStrategy)
            self.cache_enabled = True
            self.cache_ttl = self.config.get("language_chain.cache_ttl", 3600)
        except Exception as e:
            self.logger.warning(f"Failed to setup caching: {str(e)}")
            self.cache_strategy = None
            self.cache_enabled = False

    def _setup_performance_tracking(self) -> None:
        """Setup performance tracking and optimization."""
        self.processing_history = deque(maxlen=1000)
        self.response_times = deque(maxlen=100)
        self.quality_scores = deque(maxlen=100)

        # Performance thresholds
        self.max_response_time = self.config.get("language_chain.max_response_time", 30.0)
        self.min_quality_score = self.config.get("language_chain.min_quality_score", 0.7)

    @handle_exceptions
    async def process_language_request(self, request: LanguageRequest) -> LanguageResponse:
        """
        Process a comprehensive language request.

        Args:
            request: Language processing request

        Returns:
            Enhanced language response

        Raises:
            LanguageChainError: If processing fails
        """
        start_time = time.time()

        try:
            with self.tracer.trace("language_request_processing") if self.tracer else None:
                # Update metrics
                if self.metrics:
                    self.metrics.increment("language_requests_total")

                # Check cache first
                cache_key = None
                if self.cache_enabled and request.cache_response:
                    cache_key = self._generate_cache_key(request)
                    cached_response = await self._get_cached_response(cache_key)
                    if cached_response:
                        self.logger.debug(f"Cache hit for request: {cache_key[:20]}...")
                        return cached_response

                # Get relevant context
                context = await self.context_manager.get_relevant_context(request)

                # Build prompt
                conversation_context = context.get("conversation")
                prompt = self.prompt_manager.build_prompt(request, conversation_context, context)

                # Route to appropriate model and process
                response_data = await self._route_and_process(request, prompt)

                # Process and enhance response
                processing_metadata = {
                    "processing_time": time.time() - start_time,
                    "model_used": response_data.get("model_used", ""),
                    "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": response_data.get("usage", {}).get("total_tokens", 0),
                }

                response = await self.response_processor.process_response(
                    response_data.get("text", ""), request, processing_metadata
                )

                # Update conversation context
                if request.conversation_id:
                    await self.context_manager.update_conversation_context(
                        request.conversation_id, request.text, response.text, request.user_id
                    )

                # Cache response if enabled
                if self.cache_enabled and cache_key and request.cache_response:
                    await self._cache_response(cache_key, response)

                # Store for learning
                await self._store_interaction_for_learning(request, response)

                # Update performance tracking
                self._update_performance_tracking(response, time.time() - start_time)

                # Update metrics
                if self.metrics:
                    self.metrics.increment("language_responses_total")
                    self.metrics.record(
                        "language_processing_duration_seconds", time.time() - start_time
                    )
                    self.metrics.record("language_response_length_tokens", response.total_tokens)
                    self.metrics.set("language_response_quality_score", response.confidence)

                self.logger.debug(
                    f"Language request processed successfully "
                    f"(Time: {time.time() - start_time:.2f}s, "
                    f"Quality: {response.confidence:.2f})"
                )

                return response

        except Exception as e:
            if self.metrics:
                self.metrics.increment("language_errors_total")

            error_msg = f"Language processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise LanguageChainError(error_msg, component="language_chain") from e

    async def _route_and_process(self, request: LanguageRequest, prompt: str) -> Dict[str, Any]:
        """Route request to appropriate model and process."""
        try:
            # Determine model requirements based on request
            model_requirements = self._determine_model_requirements(request)

            # Route to appropriate model
            provider = await self.model_router.route_request(
                task_type="text_generation",
                requirements=model_requirements,
                context={
                    "mode": request.mode.value,
                    "response_type": request.response_type.value,
                    "enable_reasoning": request.enable_reasoning,
                },
            )

            # Prepare generation parameters
            generation_params = {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": None,
            }

            # Generate response
            response = await provider.generate_text(prompt=prompt, **generation_params)

            return {
                "text": response.text,
                "model_used": provider.model_name,
                "usage": response.usage_stats,
            }

        except Exception as e:
            raise LanguageChainError(f"Model routing and processing failed: {str(e)}") from e

    def _determine_model_requirements(self, request: LanguageRequest) -> Dict[str, Any]:
        """Determine model requirements based on request characteristics."""
        requirements = {
            "min_context_length": 4096,
            "capabilities": ["text_generation"],
            "performance_tier": "standard",
        }

        # Adjust based on processing mode
        if request.mode == ProcessingMode.REASONING:
            requirements["capabilities"].append("reasoning")
            requirements["performance_tier"] = "high"
        elif request.mode == ProcessingMode.CREATIVE:
            requirements["capabilities"].append("creative_writing")
            requirements["temperature_range"] = (0.7, 1.0)
        elif request.mode == ProcessingMode.ANALYTICAL:
            requirements["capabilities"].append("analysis")
            requirements["performance_tier"] = "high"

        # Adjust based on response type
        if request.response_type == ResponseType.CODE_GENERATION:
            requirements["capabilities"].append("code_generation")
        elif request.response_type == ResponseType.ANALYSIS:
            requirements["capabilities"].append("detailed_analysis")

        # Adjust context requirements
        if request.conversation_id:
            requirements["min_context_length"] = 8192  # Need more context for conversations

        return requirements

    async def stream_language_response(self, request: LanguageRequest) -> AsyncGenerator[str, None]:
        """
        Stream a language response for real-time interaction.

        Args:
            request: Language processing request

        Yields:
            Response text chunks
        """
        try:
            # Get context and build prompt
            context = await self.context_manager.get_relevant_context(request)
            conversation_context = context.get("conversation")
            prompt = self.prompt_manager.build_prompt(request, conversation_context, context)

            # Route to streaming-capable model
            model_requirements = self._determine_model_requirements(request)
            model_requirements["streaming"] = True

            provider = await self.model_router.route_request(
                task_type="text_generation",
                requirements=model_requirements,
                context={"streaming": True},
            )

            # Stream response
            full_response = ""
            async for chunk in provider.stream_text(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                full_response += chunk
                yield chunk

            # Update conversation context with full response
            if request.conversation_id:
                await self.context_manager.update_conversation_context(
                    request.conversation_id, request.text, full_response, request.user_id
                )

        except Exception as e:
            error_msg = f"Streaming failed: {str(e)}"
            self.logger.error(error_msg)
            yield f"Error: {error_msg}"

    async def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Get a summary of the conversation."""
        try:
            context = await self.context_manager.get_conversation_context(conversation_id)
            if not context or not context.messages:
                return None

            # Create summarization request
            messages_text = "\n".join(
                [
                    f"{msg['role']}: {msg['content']}"
                    for msg in context.messages[-20:]  # Last 20 messages
                ]
            )

            summary_request = LanguageRequest(
                text=f"Please provide a concise summary of this conversation:\n\n{messages_text}",
                mode=ProcessingMode.ANALYTICAL,
                response_type=ResponseType.SUMMARY,
                max_tokens=200,
                temperature=0.3,
            )

            summary_response = await self.process_language_request(summary_request)
            return summary_response.text

        except Exception as e:
            self.logger.error(f"Conversation summary failed: {str(e)}")
            return None

    def _generate_cache_key(self, request: LanguageRequest) -> str:
        """Generate a cache key for the request."""
        # Create a hash of relevant request parameters
        cache_data = {
            "text": request.text,
            "mode": request.mode.value,
            "response_type": request.response_type.value,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "enable_reasoning": request.enable_reasoning,
            "user_profile_hash": self._hash_user_profile(request.user_profile),
        }

        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _hash_user_profile(self, profile: Optional[UserProfile]) -> str:
        """Create a hash of user profile for caching."""
        if not profile:
            return "no_profile"

        profile_data = {
            "language": profile.preferred_language,
            "interaction_mode": (
                profile.preferred_interaction_mode.value
                if profile.preferred_interaction_mode
                else None
            ),
            "learning_style": profile.learning_style,
        }

        return hashlib.md5(json.dumps(profile_data, sort_keys=True).encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str) -> Optional[LanguageResponse]:
        """Get cached response if available."""
        if not self.cache_strategy:
            return None

        try:
            cached_data = await self.cache_strategy.get(f"lang_response:{cache_key}")
            if cached_data:
                return LanguageResponse(**json.loads(cached_data))
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")

        return None

    async def _cache_response(self, cache_key: str, response: LanguageResponse) -> None:
        """Cache the response."""
        if not self.cache_strategy:
            return

        try:
            # Convert response to dict and cache
            response_dict = {
                "text": response.text,
                "response_type": response.response_type.value,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "model_used": response.model_used,
                "timestamp": response.timestamp.isoformat(),
            }

            await self.cache_strategy.set(
                f"lang_response:{cache_key}", json.dumps(response_dict), ttl=self.cache_ttl
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {str(e)}")

    async def _store_interaction_for_learning(
        self, request: LanguageRequest, response: LanguageResponse
    ) -> None:
        """Store interaction data for learning and improvement."""
        try:
            interaction_data = {
                "request": {
                    "text": request.text,
                    "mode": request.mode.value,
                    "response_type": request.response_type.value,
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                },
                "response": {
                    "text": response.text,
                    "confidence": response.confidence,
                    "processing_time": response.processing_time,
                    "quality_scores": {
                        "coherence": response.coherence_score,
                        "relevance": response.relevance_score,
                        "safety": response.safety_score,
                    },
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self.feedback_processor.store_interaction(interaction_data)

        except Exception as e:
            self.logger.warning(f"Learning storage failed: {str(e)}")

    def _update_performance_tracking(
        self, response: LanguageResponse, processing_time: float
    ) -> None:
        """Update performance tracking metrics."""
        self.processing_history.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "processing_time": processing_time,
                "confidence": response.confidence,
                "model_used": response.model_used,
            }
        )

        self.response_times.append(processing_time)
        self.quality_scores.append(response.confidence)

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the language chain."""
        try:
            # Check model router health
            router_healthy = await self.model_router.health_check()

            # Check average response time
            avg_response_time = (
                sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
            )

            # Check average quality score
            avg_quality = (
                sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0
            )

            health_status = {
                "status": "healthy",
                "model_router_healthy": router_healthy.get("status") == "healthy",
                "average_response_time": avg_response_time,
                "average_quality_score": avg_quality,
                "active_conversations": len(self.context_manager.conversations),
                "cache_enabled": self.cache_enabled,
                "total_processed": len(self.processing_history),
            }

            # Determine overall health
            if (
                not router_healthy.get("status") == "healthy"
                or avg_response_time > self.max_response_time
                or avg_quality < self.min_quality_score
            ):
                health_status["status"] = "degraded"

            return health_status

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources and connections."""
        try:
            self.logger.info("Cleaning up EnhancedLanguageChain...")

            # Clear caches
            if hasattr(self, "conversations"):
                self.context_manager.conversations.clear()

            # Clear performance tracking
            self.processing_history.clear()
            self.response_times.clear()
            self.quality_scores.clear()

            self.logger.info("EnhancedLanguageChain cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def __del__(self):
        """Destructor."""
        try:
            asyncio.create_task(self.cleanup())
        except Exception:
            pass  # Ignore cleanup errors during destruction
