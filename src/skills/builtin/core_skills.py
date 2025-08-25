"""
Core Skills for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:28:12 UTC

This module provides essential built-in skills for the AI assistant,
including information retrieval, summarization, question answering,
content generation, and system utility skills that integrate with
all core components of the assistant architecture.
"""

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, Union

import asyncio
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ContextAdapted,
    FeedbackReceived,
    SkillExecutionCompleted,
    SkillExecutionFailed,
    SkillExecutionStarted,
    SkillRegistered,
)
from src.core.health_check import HealthCheck
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.external_apis.web_search import WebSearchAPI

# Integration components
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.memory.operations.retrieval import MemoryRetrieval
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy
from src.processing.natural_language.entity_extractor import EntityExtractor

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.reasoning.inference_engine import InferenceEngine
from src.reasoning.knowledge_graph import KnowledgeGraph

# Reasoning systems
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.planning.task_planner import TaskPlanner


class SkillCategory(Enum):
    """Categories of core skills."""

    INFORMATION = "information"  # Information retrieval and presentation
    GENERATION = "generation"  # Content generation and creation
    COMPREHENSION = "comprehension"  # Understanding and analysis
    UTILITY = "utility"  # System and utility functions
    PERSONALIZATION = "personalization"  # User-specific capabilities
    INTERACTION = "interaction"  # Conversation and interaction flow
    COACHING = "coaching"  # Coaching and personal development


class SkillPriority(Enum):
    """Priority levels for skills execution."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class SkillResult:
    """Standard result container for skill execution."""

    success: bool
    result: Any
    execution_time: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)


@dataclass
class SkillMetadata:
    """Metadata for skill registration."""

    skill_id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: SkillCategory = SkillCategory.UTILITY
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    required_components: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    author: str = "Drmusab"
    tags: Set[str] = field(default_factory=set)
    is_stateful: bool = False
    timeout_seconds: float = 30.0
    cache_ttl_seconds: int = 300
    priority: SkillPriority = SkillPriority.NORMAL


class SkillError(Exception):
    """Custom exception for skill execution errors."""

    def __init__(
        self, message: str, skill_id: Optional[str] = None, error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.skill_id = skill_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class BaseSkill:
    """Base class for all skills."""

    def __init__(self, container: Container):
        """
        Initialize the skill.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(f"skill.{self.get_metadata().skill_id}")

        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)

        # Performance tracking
        self.execution_stats: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Optional components that might be requested
        self._optional_components = {}

    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata (must be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement get_metadata()")

    async def setup(self) -> None:
        """Setup skill (optional)."""
        pass

    async def cleanup(self) -> None:
        """Cleanup resources (optional)."""
        pass

    @handle_exceptions
    async def execute(self, **params) -> SkillResult:
        """
        Execute the skill.

        Args:
            **params: Parameters for skill execution

        Returns:
            Execution result
        """
        metadata = self.get_metadata()
        start_time = time.time()

        # Emit skill execution started event
        await self.event_bus.emit(
            SkillExecutionStarted(
                skill_id=metadata.skill_id, skill_name=metadata.name, parameters=params
            )
        )

        try:
            # Validate parameters
            self._validate_parameters(params)

            # Execute skill implementation
            result = await self._execute(**params)

            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time

            # Update execution stats
            self.execution_stats[metadata.skill_id].append(execution_time)

            # Emit skill execution completed event
            await self.event_bus.emit(
                SkillExecutionCompleted(
                    skill_id=metadata.skill_id,
                    skill_name=metadata.name,
                    execution_time=execution_time,
                    success=result.success,
                )
            )

            self.logger.info(
                f"Skill {metadata.skill_id} executed in {execution_time:.2f}s with success={result.success}"
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.error_counts[metadata.skill_id] += 1

            # Emit skill execution failed event
            await self.event_bus.emit(
                SkillExecutionFailed(
                    skill_id=metadata.skill_id,
                    skill_name=metadata.name,
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            )

            self.logger.error(f"Skill {metadata.skill_id} failed: {str(e)}")

            return SkillResult(
                success=False, result=None, execution_time=execution_time, errors=[str(e)]
            )

    async def _execute(self, **params) -> SkillResult:
        """
        Implementation of skill execution (must be implemented by subclasses).

        Args:
            **params: Parameters for skill execution

        Returns:
            Execution result
        """
        raise NotImplementedError("Subclasses must implement _execute()")

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """
        Validate parameters against skill metadata.

        Args:
            params: Parameters to validate

        Raises:
            SkillError: If parameters are invalid
        """
        metadata = self.get_metadata()

        # Check required parameters
        for param_name, param_info in metadata.parameters.items():
            if param_info.get("required", False) and param_name not in params:
                raise SkillError(f"Required parameter '{param_name}' is missing", metadata.skill_id)

            # Type validation if provided
            if param_name in params and "type" in param_info:
                param_type = param_info["type"]
                value = params[param_name]

                if param_type == "string" and not isinstance(value, str):
                    raise SkillError(
                        f"Parameter '{param_name}' must be a string", metadata.skill_id
                    )
                elif param_type == "integer" and not isinstance(value, int):
                    raise SkillError(
                        f"Parameter '{param_name}' must be an integer", metadata.skill_id
                    )
                elif param_type == "float" and not isinstance(value, (int, float)):
                    raise SkillError(
                        f"Parameter '{param_name}' must be a number", metadata.skill_id
                    )
                elif param_type == "boolean" and not isinstance(value, bool):
                    raise SkillError(
                        f"Parameter '{param_name}' must be a boolean", metadata.skill_id
                    )
                elif param_type == "array" and not isinstance(value, list):
                    raise SkillError(
                        f"Parameter '{param_name}' must be an array", metadata.skill_id
                    )
                elif param_type == "object" and not isinstance(value, dict):
                    raise SkillError(
                        f"Parameter '{param_name}' must be an object", metadata.skill_id
                    )

    def _get_component(self, component_type: Type) -> Any:
        """
        Get a component from the container.

        Args:
            component_type: Type of component to get

        Returns:
            Component instance
        """
        if component_type not in self._optional_components:
            try:
                component = self.container.get(component_type)
                self._optional_components[component_type] = component
            except Exception as e:
                self.logger.warning(f"Failed to get component {component_type.__name__}: {str(e)}")
                self._optional_components[component_type] = None

        return self._optional_components[component_type]


class InformationRetrievalSkill(BaseSkill):
    """Skill for retrieving information from various sources."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.memory_retrieval = container.get(MemoryRetrieval)
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.model_router = container.get(ModelRouter)

        # Optional components
        self.web_search = self._get_component(WebSearchAPI)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="information_retrieval",
            name="Information Retrieval",
            description="Retrieves information from memory, knowledge base, or external sources",
            category=SkillCategory.INFORMATION,
            parameters={
                "query": {
                    "type": "string",
                    "description": "The information query",
                    "required": True,
                },
                "sources": {
                    "type": "array",
                    "description": "Sources to search (memory, knowledge_base, web)",
                    "required": False,
                    "default": ["memory", "knowledge_base"],
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "required": False,
                    "default": 5,
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for the query",
                    "required": False,
                },
            },
            examples=[
                {
                    "query": "What is machine learning?",
                    "sources": ["knowledge_base"],
                    "max_results": 3,
                },
                {
                    "query": "Find information about renewable energy",
                    "sources": ["memory", "knowledge_base", "web"],
                    "max_results": 5,
                },
            ],
            required_components=["MemoryRetrieval", "KnowledgeGraph", "ModelRouter"],
            tags={"retrieval", "search", "knowledge", "information"},
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute information retrieval."""
        query = params["query"]
        sources = params.get("sources", ["memory", "knowledge_base"])
        max_results = params.get("max_results", 5)
        context = params.get("context", {})

        results = []
        metadata = {}

        # Query memory if requested
        if "memory" in sources:
            memory_results = await self.memory_retrieval.retrieve_relevant(
                query=query, limit=max_results, context=context
            )
            results.extend(memory_results)
            metadata["memory_results_count"] = len(memory_results)

        # Query knowledge graph if requested
        if "knowledge_base" in sources:
            knowledge_results = await self.knowledge_graph.query(query=query, limit=max_results)
            results.extend(knowledge_results)
            metadata["knowledge_results_count"] = len(knowledge_results)

        # Query web if requested and available
        if "web" in sources and self.web_search:
            try:
                web_results = await self.web_search.search(query=query, max_results=max_results)
                results.extend(web_results)
                metadata["web_results_count"] = len(web_results)
            except Exception as e:
                self.logger.warning(f"Web search failed: {str(e)}")
                metadata["web_search_error"] = str(e)

        # Sort and deduplicate results
        unique_results = []
        seen_urls = set()
        seen_texts = set()

        for result in sorted(results, key=lambda x: x.get("relevance", 0), reverse=True):
            # Skip duplicates
            url = result.get("url")
            content = result.get("content", "")
            content_hash = hash(content[:100])  # Use first 100 chars for deduplication

            if (url and url in seen_urls) or content_hash in seen_texts:
                continue

            if url:
                seen_urls.add(url)
            seen_texts.add(content_hash)

            unique_results.append(result)

            if len(unique_results) >= max_results:
                break

        metadata["total_results"] = len(results)
        metadata["unique_results"] = len(unique_results)

        # Determine confidence based on result quality
        confidence = min(0.9, 0.3 + (0.6 * min(1.0, len(unique_results) / max(1, max_results))))

        # Build sources list for result
        sources_list = [
            {
                "type": result.get("source_type", "unknown"),
                "name": result.get("source_name", "Unknown"),
                "url": result.get("url"),
                "relevance": result.get("relevance", 0),
            }
            for result in unique_results
            if result.get("url") or result.get("source_name")
        ]

        return SkillResult(
            success=True,
            result=unique_results,
            confidence=confidence,
            metadata=metadata,
            sources=sources_list,
        )


class TextGenerationSkill(BaseSkill):
    """Skill for generating text based on prompts and context."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.model_router = container.get(ModelRouter)
        self.language_chain = container.get(LanguageChain)
        self.context_manager = container.get(ContextManager)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="text_generation",
            name="Text Generation",
            description="Generates text based on prompts and context",
            category=SkillCategory.GENERATION,
            parameters={
                "prompt": {
                    "type": "string",
                    "description": "The generation prompt",
                    "required": True,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to generate",
                    "required": False,
                    "default": 500,
                },
                "temperature": {
                    "type": "float",
                    "description": "Generation temperature (creativity)",
                    "required": False,
                    "default": 0.7,
                },
                "context_id": {
                    "type": "string",
                    "description": "Context ID for stateful generation",
                    "required": False,
                },
                "style": {
                    "type": "string",
                    "description": "Style of the generated text",
                    "required": False,
                    "default": "informative",
                },
            },
            examples=[
                {
                    "prompt": "Write a short story about a robot learning to paint",
                    "max_tokens": 1000,
                    "temperature": 0.8,
                    "style": "creative",
                },
                {
                    "prompt": "Explain quantum computing in simple terms",
                    "max_tokens": 300,
                    "temperature": 0.3,
                    "style": "educational",
                },
            ],
            required_components=["ModelRouter", "LanguageChain", "ContextManager"],
            tags={"generation", "text", "writing", "content"},
            is_stateful=True,
            cache_ttl_seconds=0,  # Don't cache creative generations
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute text generation."""
        prompt = params["prompt"]
        max_tokens = params.get("max_tokens", 500)
        temperature = params.get("temperature", 0.7)
        context_id = params.get("context_id")
        style = params.get("style", "informative")

        # Get context if context_id is provided
        context = {}
        if context_id:
            context = await self.context_manager.get_context(context_id) or {}

        # Adjust system prompt based on style
        system_prompt = self._get_system_prompt_for_style(style)

        # Generate text
        generation_result = await self.language_chain.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            model_params={"max_tokens": max_tokens, "temperature": temperature},
        )

        # Update context if needed
        context_updates = {}
        if context_id:
            # Add to conversation history
            context_updates = {
                "last_prompt": prompt,
                "last_generation": generation_result["text"],
                "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.context_manager.update_context(context_id, context_updates)

        return SkillResult(
            success=True,
            result=generation_result["text"],
            confidence=generation_result.get("confidence", 0.8),
            metadata={
                "model": generation_result.get("model", "unknown"),
                "tokens_used": generation_result.get("tokens_used", 0),
                "style": style,
            },
            context_updates=context_updates,
        )

    def _get_system_prompt_for_style(self, style: str) -> str:
        """Get appropriate system prompt for the requested style."""
        style_prompts = {
            "informative": "You are a helpful assistant providing clear, accurate, and informative content. Focus on facts and clarity.",
            "creative": "You are a creative writer with a vivid imagination. Create engaging, original content with descriptive language and narrative flair.",
            "professional": "You are a professional business writer. Maintain a formal tone with precise language and well-structured content suitable for business contexts.",
            "casual": "You are a friendly, conversational writer. Use an approachable, relaxed tone with simple language and occasional humor.",
            "technical": "You are a technical expert. Provide detailed, precise explanations with appropriate terminology and structured information.",
            "educational": "You are an educational content creator. Explain concepts clearly with examples, analogies, and a supportive, encouraging tone.",
            "persuasive": "You are a persuasive writer. Present compelling arguments with evidence, addressing potential objections and using rhetorical techniques effectively.",
        }

        return style_prompts.get(style.lower(), style_prompts["informative"])


class SummarizationSkill(BaseSkill):
    """Skill for summarizing content to different lengths and styles."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.model_router = container.get(ModelRouter)
        self.language_chain = container.get(LanguageChain)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="summarization",
            name="Content Summarization",
            description="Summarizes content to different lengths and styles",
            category=SkillCategory.COMPREHENSION,
            parameters={
                "content": {
                    "type": "string",
                    "description": "The content to summarize",
                    "required": True,
                },
                "length": {
                    "type": "string",
                    "description": "Desired summary length (brief, medium, detailed)",
                    "required": False,
                    "default": "medium",
                },
                "format": {
                    "type": "string",
                    "description": "Summary format (paragraph, bullet_points, key_points)",
                    "required": False,
                    "default": "paragraph",
                },
                "focus": {
                    "type": "string",
                    "description": "Aspect to focus on in the summary",
                    "required": False,
                },
            },
            examples=[
                {
                    "content": "Long article about climate change...",
                    "length": "brief",
                    "format": "bullet_points",
                },
                {
                    "content": "Research paper on quantum computing...",
                    "length": "detailed",
                    "format": "paragraph",
                    "focus": "practical applications",
                },
            ],
            required_components=["ModelRouter", "LanguageChain"],
            tags={"summarization", "compression", "content", "analysis"},
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute summarization."""
        content = params["content"]
        length = params.get("length", "medium")
        format_type = params.get("format", "paragraph")
        focus = params.get("focus")

        # Validate content length
        if len(content) < 100:
            return SkillResult(
                success=False,
                result=content,
                confidence=1.0,
                errors=["Content too short to summarize meaningfully"],
            )

        # Build summarization prompt
        length_guide = {
            "brief": "Create a very concise summary in 1-2 sentences.",
            "medium": "Create a comprehensive summary in 3-5 sentences.",
            "detailed": "Create a detailed summary that captures all key information.",
        }

        format_guide = {
            "paragraph": "Format the summary as a cohesive paragraph.",
            "bullet_points": "Format the summary as a list of bullet points.",
            "key_points": "Format the summary as 'Key Points:' followed by numbered items.",
        }

        focus_instruction = f"Focus specifically on aspects related to {focus}." if focus else ""

        summarization_prompt = f"""
        Summarize the following content:
        
        {content}
        
        {length_guide.get(length, length_guide["medium"])}
        {format_guide.get(format_type, format_guide["paragraph"])}
        {focus_instruction}
        
        Make sure the summary is accurate, comprehensive, and captures the most important information.
        """

        # Generate summary
        summary_result = await self.language_chain.generate_text(
            prompt=summarization_prompt,
            system_prompt="You are an expert summarizer that creates accurate, clear, and concise summaries.",
            model_params={"temperature": 0.3},  # Lower temperature for more deterministic summary
        )

        # Clean up summary (remove any prefixes like "Summary:" that the model might add)
        summary_text = summary_result["text"].strip()
        summary_text = re.sub(r"^(Summary:?\s*|Here\'s a summary:?\s*)", "", summary_text)

        # Calculate a confidence score based on content length and model confidence
        content_length_factor = min(
            1.0, len(content) / 10000
        )  # Longer content may be harder to summarize well
        confidence = 0.7 * summary_result.get("confidence", 0.8) + 0.3 * (1 - content_length_factor)

        return SkillResult(
            success=True,
            result=summary_text,
            confidence=confidence,
            metadata={
                "original_length": len(content),
                "summary_length": len(summary_text),
                "compression_ratio": len(summary_text) / max(1, len(content)),
                "summary_type": f"{length}_{format_type}",
            },
        )


class QuestionAnsweringSkill(BaseSkill):
    """Skill for answering questions based on available knowledge."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.model_router = container.get(ModelRouter)
        self.memory_manager = container.get(MemoryManager)
        self.semantic_memory = container.get(SemanticMemory)
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.inference_engine = container.get(InferenceEngine)

        # Optional components
        self.web_search = self._get_component(WebSearchAPI)

        # Cache for similar questions
        self.question_cache = {}

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="question_answering",
            name="Question Answering",
            description="Answers questions based on available knowledge sources",
            category=SkillCategory.COMPREHENSION,
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to answer",
                    "required": True,
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for answering",
                    "required": False,
                },
                "include_sources": {
                    "type": "boolean",
                    "description": "Whether to include sources in the answer",
                    "required": False,
                    "default": True,
                },
                "search_external": {
                    "type": "boolean",
                    "description": "Whether to search external sources if needed",
                    "required": False,
                    "default": False,
                },
            },
            examples=[
                {"question": "What is the capital of France?", "include_sources": True},
                {
                    "question": "How does photosynthesis work?",
                    "search_external": True,
                    "context": {"user_knowledge_level": "high_school"},
                },
            ],
            required_components=[
                "ModelRouter",
                "MemoryManager",
                "SemanticMemory",
                "KnowledgeGraph",
                "InferenceEngine",
            ],
            tags={"question", "answering", "knowledge", "information"},
            cache_ttl_seconds=3600,  # Cache for 1 hour
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute question answering."""
        question = params["question"]
        context = params.get("context", {})
        include_sources = params.get("include_sources", True)
        search_external = params.get("search_external", False)

        # Check cache for similar questions
        cache_key = hashlib.md5(question.lower().encode()).hexdigest()
        if cache_key in self.question_cache:
            cached_result = self.question_cache[cache_key]
            self.logger.info(f"Using cached answer for similar question: {question}")
            return cached_result

        # Step 1: Search semantic memory and knowledge graph
        memory_results = await self.semantic_memory.search(query=question, limit=5)

        knowledge_results = await self.knowledge_graph.query(query=question, limit=5)

        external_results = []
        # Step 2: Search external sources if requested and needed
        if (
            search_external
            and self.web_search
            and (len(memory_results) + len(knowledge_results) < 2)
        ):
            try:
                external_results = await self.web_search.search(query=question, max_results=3)
            except Exception as e:
                self.logger.warning(f"External search failed: {str(e)}")

        # Step 3: Combine and rank evidence
        all_evidence = []
        all_evidence.extend(
            [{"text": r["content"], "source": "memory", "metadata": r} for r in memory_results]
        )
        all_evidence.extend(
            [
                {"text": r["content"], "source": "knowledge_graph", "metadata": r}
                for r in knowledge_results
            ]
        )
        all_evidence.extend(
            [{"text": r["content"], "source": "web", "metadata": r} for r in external_results]
        )

        # Step 4: Use inference engine to generate answer from evidence
        inference_result = await self.inference_engine.answer_question(
            question=question, evidence=all_evidence, context=context
        )

        answer = inference_result["answer"]
        confidence = inference_result.get("confidence", 0.0)
        reasoning = inference_result.get("reasoning", [])

        # Step 5: Format sources if requested
        sources = []
        if include_sources and inference_result.get("used_evidence", []):
            for evidence in inference_result["used_evidence"]:
                if evidence["source"] == "memory":
                    sources.append(
                        {
                            "type": "memory",
                            "description": evidence["metadata"].get("description", "Memory entry"),
                            "timestamp": evidence["metadata"].get("timestamp"),
                        }
                    )
                elif evidence["source"] == "knowledge_graph":
                    sources.append(
                        {
                            "type": "knowledge",
                            "description": evidence["metadata"].get(
                                "description", "Knowledge entry"
                            ),
                            "confidence": evidence["metadata"].get("confidence", 0.8),
                        }
                    )
                elif evidence["source"] == "web":
                    sources.append(
                        {
                            "type": "web",
                            "url": evidence["metadata"].get("url", ""),
                            "title": evidence["metadata"].get("title", "Web result"),
                        }
                    )

        # Create result
        result = SkillResult(
            success=True,
            result=answer,
            confidence=confidence,
            metadata={
                "reasoning_trace": reasoning,
                "evidence_count": len(all_evidence),
                "sources_used": len(sources),
            },
            sources=sources,
        )

        # Cache result if confidence is high enough
        if confidence > 0.7:
            self.question_cache[cache_key] = result
            # Limit cache size
            if len(self.question_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = sorted(self.question_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.question_cache[key]

        return result


class ContextAwarenessSkill(BaseSkill):
    """Skill for tracking, updating, and retrieving interaction context."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.entity_extractor = container.get(EntityExtractor)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="context_awareness",
            name="Context Awareness",
            description="Manages interaction context and awareness",
            category=SkillCategory.UTILITY,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (get, update, analyze, track)",
                    "required": True,
                },
                "context_id": {
                    "type": "string",
                    "description": "Context identifier",
                    "required": True,
                },
                "data": {
                    "type": "object",
                    "description": "Data for update operations",
                    "required": False,
                },
                "text": {
                    "type": "string",
                    "description": "Text to analyze for context tracking",
                    "required": False,
                },
            },
            examples=[
                {"operation": "get", "context_id": "session_123"},
                {
                    "operation": "update",
                    "context_id": "session_123",
                    "data": {"user_location": "New York", "weather": "sunny"},
                },
                {
                    "operation": "track",
                    "context_id": "session_123",
                    "text": "I'm looking for a restaurant in Seattle for dinner tomorrow.",
                },
            ],
            required_components=[
                "ContextManager",
                "WorkingMemory",
                "EntityExtractor",
                "SentimentAnalyzer",
            ],
            tags={"context", "awareness", "tracking", "memory"},
            is_stateful=True,
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute context awareness operations."""
        operation = params["operation"]
        context_id = params["context_id"]

        if operation == "get":
            return await self._get_context(context_id)
        elif operation == "update":
            data = params.get("data", {})
            return await self._update_context(context_id, data)
        elif operation == "analyze":
            return await self._analyze_context(context_id)
        elif operation == "track":
            text = params.get("text", "")
            return await self._track_context(context_id, text)
        else:
            return SkillResult(
                success=False, result=None, errors=[f"Unknown operation: {operation}"]
            )

    async def _get_context(self, context_id: str) -> SkillResult:
        """Get current context."""
        context = await self.context_manager.get_context(context_id) or {}
        working_memory = await self.working_memory.get_session_data(context_id) or {}

        combined_context = {**context, "working_memory": working_memory}

        return SkillResult(
            success=True,
            result=combined_context,
            confidence=1.0,
            metadata={
                "context_keys": list(context.keys()),
                "memory_keys": list(working_memory.keys()),
            },
        )

    async def _update_context(self, context_id: str, data: Dict[str, Any]) -> SkillResult:
        """Update context with new data."""
        await self.context_manager.update_context(context_id, data)

        # Emit context adaptation event
        await self.event_bus.emit(ContextAdapted(context_id=context_id, updates=list(data.keys())))

        return SkillResult(
            success=True,
            result={"updated": list(data.keys())},
            confidence=1.0,
            context_updates=data,
        )

    async def _analyze_context(self, context_id: str) -> SkillResult:
        """Analyze current context for insights."""
        context = await self.context_manager.get_context(context_id) or {}
        working_memory = await self.working_memory.get_session_data(context_id) or {}

        # Analyze for insights
        insights = {
            "context_age": None,
            "dominant_topics": [],
            "user_preferences": {},
            "interaction_patterns": {},
            "context_stability": 0.0,
        }

        # Extract created_at if available
        if "created_at" in context:
            try:
                created_at = datetime.fromisoformat(context["created_at"])
                now = datetime.now(timezone.utc)
                age_seconds = (now - created_at).total_seconds()
                insights["context_age"] = age_seconds
            except Exception:
                pass

        # Extract topics from conversation history
        if "conversation_history" in context:
            topic_counts = defaultdict(int)
            for entry in context["conversation_history"]:
                if isinstance(entry, dict) and "topic" in entry:
                    topic_counts[entry["topic"]] += 1

            insights["dominant_topics"] = [
                {"topic": topic, "count": count}
                for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[
                    :3
                ]
            ]

        # Extract user preferences
        if "user_preferences" in context:
            insights["user_preferences"] = context["user_preferences"]

        # Analyze interaction patterns
        if "interaction_history" in context:
            time_patterns = []
            for i in range(1, len(context["interaction_history"])):
                if isinstance(context["interaction_history"][i], dict) and isinstance(
                    context["interaction_history"][i - 1], dict
                ):
                    try:
                        time1 = datetime.fromisoformat(
                            context["interaction_history"][i - 1].get("timestamp", "")
                        )
                        time2 = datetime.fromisoformat(
                            context["interaction_history"][i].get("timestamp", "")
                        )
                        time_diff = (time2 - time1).total_seconds()
                        time_patterns.append(time_diff)
                    except Exception:
                        pass

            if time_patterns:
                insights["interaction_patterns"]["avg_response_time"] = sum(time_patterns) / len(
                    time_patterns
                )
                insights["interaction_patterns"]["interaction_count"] = len(
                    context["interaction_history"]
                )

        # Calculate context stability (how consistent the context has been)
        if "context_updates" in working_memory:
            updates = working_memory["context_updates"]
            if isinstance(updates, list) and len(updates) > 1:
                # More updates means less stability
                stability = 1.0 - min(0.9, len(updates) / 20)
                insights["context_stability"] = stability

        return SkillResult(
            success=True,
            result=insights,
            confidence=0.8,
            metadata={"context_size": len(context), "memory_size": len(working_memory)},
        )

    async def _track_context(self, context_id: str, text: str) -> SkillResult:
        """Track and update context based on text input."""
        if not text:
            return SkillResult(
                success=False, result=None, errors=["No text provided for context tracking"]
            )

        # Extract entities
        entities = await self.entity_extractor.extract(text)

        # Analyze sentiment
        sentiment = await self.sentiment_analyzer.analyze(text)

        # Prepare context updates
        updates = {
            "last_text": text,
            "last_processed": datetime.now(timezone.utc).isoformat(),
            "extracted_entities": entities,
            "sentiment": sentiment,
        }

        # Special handling for specific entity types
        entity_updates = {}
        for entity in entities:
            entity_type = entity.get("type", "").lower()
            entity_value = entity.get("value")

            if entity_type in ["location", "place", "city", "country"]:
                entity_updates["user_location"] = entity_value
            elif entity_type in ["person", "people"]:
                entity_updates["mentioned_people"] = entity_updates.get("mentioned_people", []) + [
                    entity_value
                ]
            elif entity_type in ["organization", "company"]:
                entity_updates["mentioned_organizations"] = entity_updates.get(
                    "mentioned_organizations", []
                ) + [entity_value]
            elif entity_type in ["date", "time", "datetime"]:
                entity_updates["mentioned_datetime"] = entity_value
            elif entity_type in ["product", "service"]:
                entity_updates["mentioned_products"] = entity_updates.get(
                    "mentioned_products", []
                ) + [entity_value]

        # Update context with entity-specific updates
        if entity_updates:
            updates.update(entity_updates)

        # Track context updates in working memory
        await self.working_memory.update(
            context_id,
            {
                "context_updates": [
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "updates": list(updates.keys()),
                    }
                ]
            },
        )

        # Update context
        await self.context_manager.update_context(context_id, updates)

        # Emit context adaptation event
        await self.event_bus.emit(
            ContextAdapted(context_id=context_id, updates=list(updates.keys()))
        )

        return SkillResult(
            success=True,
            result={"tracked_updates": list(updates.keys())},
            confidence=0.9,
            metadata={
                "entity_count": len(entities),
                "sentiment": sentiment.get("sentiment", "neutral"),
                "sentiment_score": sentiment.get("score", 0.0),
            },
            context_updates=updates,
        )


class PersonalizationSkill(BaseSkill):
    """Skill for personalizing interactions based on user preferences and patterns."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.memory_manager = container.get(MemoryManager)
        self.episodic_memory = container.get(EpisodicMemory)
        self.preference_learner = self._get_component(PreferenceLearner)
        self.model_router = container.get(ModelRouter)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="personalization",
            name="Personalization",
            description="Personalizes interactions based on user preferences and patterns",
            category=SkillCategory.PERSONALIZATION,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (get_profile, adapt_content, update_preferences, analyze_patterns)",
                    "required": True,
                },
                "user_id": {"type": "string", "description": "User identifier", "required": True},
                "content": {
                    "type": "string",
                    "description": "Content to personalize",
                    "required": False,
                },
                "preferences": {
                    "type": "object",
                    "description": "Preference updates",
                    "required": False,
                },
            },
            examples=[
                {"operation": "get_profile", "user_id": "user_123"},
                {
                    "operation": "adapt_content",
                    "user_id": "user_123",
                    "content": "Here is some general information about machine learning...",
                },
                {
                    "operation": "update_preferences",
                    "user_id": "user_123",
                    "preferences": {"language": "Spanish", "topic_interests": ["AI", "Space"]},
                },
            ],
            required_components=[
                "MemoryManager",
                "EpisodicMemory",
                "PreferenceLearner",
                "ModelRouter",
            ],
            tags={"personalization", "preferences", "adaptation", "user_experience"},
            is_stateful=True,
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute personalization operations."""
        operation = params["operation"]
        user_id = params["user_id"]

        if operation == "get_profile":
            return await self._get_user_profile(user_id)
        elif operation == "adapt_content":
            content = params.get("content", "")
            return await self._adapt_content(user_id, content)
        elif operation == "update_preferences":
            preferences = params.get("preferences", {})
            return await self._update_preferences(user_id, preferences)
        elif operation == "analyze_patterns":
            return await self._analyze_user_patterns(user_id)
        else:
            return SkillResult(
                success=False, result=None, errors=[f"Unknown operation: {operation}"]
            )

    async def _get_user_profile(self, user_id: str) -> SkillResult:
        """Get comprehensive user profile."""
        profile = {"user_id": user_id}

        # Get user preferences
        if self.preference_learner:
            preferences = await self.preference_learner.get_user_preferences(user_id)
            if preferences:
                profile["preferences"] = preferences

        # Get user interaction patterns from episodic memory
        user_memories = await self.episodic_memory.get_user_memories(user_id, limit=50)

        # Extract patterns from memories
        if user_memories:
            # Analyze interaction frequency
            timestamps = []
            session_durations = []
            topics = defaultdict(int)

            for memory in user_memories:
                if isinstance(memory, dict):
                    # Collect timestamps
                    if "timestamp" in memory:
                        try:
                            timestamps.append(datetime.fromisoformat(memory["timestamp"]))
                        except (ValueError, TypeError):
                            pass

                    # Collect session durations
                    if "duration" in memory:
                        session_durations.append(memory["duration"])

                    # Track topics
                    if "topics_discussed" in memory and isinstance(
                        memory["topics_discussed"], list
                    ):
                        for topic in memory["topics_discussed"]:
                            topics[topic] += 1

            # Calculate patterns
            interaction_patterns = {}

            if timestamps:
                timestamps.sort()
                # Determine time of day patterns
                hours = [ts.hour for ts in timestamps]
                morning = sum(1 for h in hours if 5 <= h < 12)
                afternoon = sum(1 for h in hours if 12 <= h < 18)
                evening = sum(1 for h in hours if 18 <= h < 22)
                night = sum(1 for h in hours if h >= 22 or h < 5)

                interaction_patterns["time_of_day"] = {
                    "morning": morning / len(hours) if hours else 0,
                    "afternoon": afternoon / len(hours) if hours else 0,
                    "evening": evening / len(hours) if hours else 0,
                    "night": night / len(hours) if hours else 0,
                }

                # Determine day of week patterns
                days = [ts.weekday() for ts in timestamps]
                day_counts = [days.count(i) for i in range(7)]

                interaction_patterns["day_of_week"] = {
                    "monday": day_counts[0] / len(days) if days else 0,
                    "tuesday": day_counts[1] / len(days) if days else 0,
                    "wednesday": day_counts[2] / len(days) if days else 0,
                    "thursday": day_counts[3] / len(days) if days else 0,
                    "friday": day_counts[4] / len(days) if days else 0,
                    "saturday": day_counts[5] / len(days) if days else 0,
                    "sunday": day_counts[6] / len(days) if days else 0,
                }

            if session_durations:
                interaction_patterns["avg_session_duration"] = sum(session_durations) / len(
                    session_durations
                )

            if topics:
                profile["topic_interests"] = [
                    {"topic": topic, "score": count / sum(topics.values())}
                    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
                ]

            profile["interaction_patterns"] = interaction_patterns

        return SkillResult(
            success=True,
            result=profile,
            confidence=0.9 if "preferences" in profile else 0.7,
            metadata={"profile_completeness": len(profile) / 5},  # Estimate of profile completeness
        )

    async def _adapt_content(self, user_id: str, content: str) -> SkillResult:
        """Adapt content based on user preferences."""
        if not content:
            return SkillResult(
                success=False, result=None, errors=["No content provided for adaptation"]
            )

        # Get user profile
        profile_result = await self._get_user_profile(user_id)
        profile = profile_result.result if profile_result.success else {"user_id": user_id}

        # Extract relevant preferences
        preferences = profile.get("preferences", {})
        topic_interests = profile.get("topic_interests", [])

        # Apply personalization rules
        adaptations = []

        # Language preference
        preferred_language = preferences.get("language")
        if preferred_language and preferred_language.lower() != "english":
            # We'd need a translation component here
            # For now, just note it as an adaptation that would happen
            adaptations.append(f"Would translate content to {preferred_language}")

        # Detail level preference
        detail_level = preferences.get("detail_level", "medium")
        if detail_level == "high" and len(content) < 500:
            adaptations.append("Would expand content with more details")
        elif detail_level == "low" and len(content) > 200:
            adaptations.append("Would condense content for brevity")

        # Adapt to topic interests
        if topic_interests:
            top_topics = [item["topic"] for item in topic_interests[:3]]
            adaptations.append(
                f"Would emphasize aspects related to user interests: {', '.join(top_topics)}"
            )

        # Communication style preference
        style = preferences.get("communication_style", "neutral")
        if style == "formal":
            adaptations.append("Would adjust to formal communication style")
        elif style == "casual":
            adaptations.append("Would adjust to casual communication style")
        elif style == "technical":
            adaptations.append("Would use more technical terminology")

        # Create personalization prompt
        if adaptations:
            personalization_prompt = f"""
            Personalize the following content for a user with these preferences:
            
            Language preference: {preferences.get('language', 'English')}
            Detail level: {preferences.get('detail_level', 'medium')}
            Communication style: {preferences.get('communication_style', 'neutral')}
            Topic interests: {', '.join(top_topics) if topic_interests else 'Not specified'}
            
            Specifically:
            {' '.join(adaptations)}
            
            Original content:
            {content}
            """

            # Generate personalized content
            personalized_content = await self.model_router.generate_text(
                prompt=personalization_prompt, model_params={"temperature": 0.7}
            )

            adapted_content = personalized_content["text"]
        else:
            # No adaptations needed
            adapted_content = content

        return SkillResult(
            success=True,
            result=adapted_content,
            confidence=0.8 if adaptations else 1.0,
            metadata={
                "adaptations_applied": adaptations,
                "original_length": len(content),
                "adapted_length": len(adapted_content),
            },
        )

    async def _update_preferences(self, user_id: str, preferences: Dict[str, Any]) -> SkillResult:
        """Update user preferences."""
        if not preferences:
            return SkillResult(
                success=False, result=None, errors=["No preferences provided for update"]
            )

        if not self.preference_learner:
            return SkillResult(
                success=False, result=None, errors=["Preference learning component not available"]
            )

        # Update preferences
        updated = await self.preference_learner.update_preferences(user_id, preferences)

        # Record preference update in episodic memory
        await self.episodic_memory.store(
            {
                "event_type": "preference_update",
                "user_id": user_id,
                "preferences_updated": list(preferences.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        return SkillResult(
            success=updated,
            result={"updated_preferences": list(preferences.keys())},
            confidence=1.0 if updated else 0.0,
            metadata={"preference_count": len(preferences)},
        )

    async def _analyze_user_patterns(self, user_id: str) -> SkillResult:
        """Analyze user interaction patterns."""
        # Get user memories
        user_memories = await self.episodic_memory.get_user_memories(user_id, limit=100)

        if not user_memories:
            return SkillResult(
                success=False, result=None, errors=["Insufficient data to analyze user patterns"]
            )

        # Analyze patterns
        patterns = {
            "interaction_frequency": {},
            "content_preferences": {},
            "response_patterns": {},
            "session_patterns": {},
        }

        # Analyze interaction frequency
        timestamps = []
        for memory in user_memories:
            if isinstance(memory, dict) and "timestamp" in memory:
                try:
                    timestamps.append(datetime.fromisoformat(memory["timestamp"]))
                except (ValueError, TypeError):
                    pass

        if timestamps:
            timestamps.sort()

            # Calculate time between interactions
            if len(timestamps) > 1:
                intervals = [
                    (timestamps[i] - timestamps[i - 1]).total_seconds()
                    for i in range(1, len(timestamps))
                ]

                patterns["interaction_frequency"]["avg_interval_seconds"] = sum(intervals) / len(
                    intervals
                )
                patterns["interaction_frequency"]["min_interval_seconds"] = min(intervals)
                patterns["interaction_frequency"]["max_interval_seconds"] = max(intervals)

            # Determine regular usage times
            hours = [ts.hour for ts in timestamps]
            hour_counts = [hours.count(i) for i in range(24)]
            peak_hour = hour_counts.index(max(hour_counts))

            patterns["interaction_frequency"]["peak_hour"] = peak_hour
            patterns["interaction_frequency"]["active_hours"] = [
                i for i in range(24) if hour_counts[i] > sum(hour_counts) / (24 * 2)
            ]

        # Analyze content preferences
        topics_seen = defaultdict(int)
        for memory in user_memories:
            if isinstance(memory, dict):
                # Extract topics
                if "topics_discussed" in memory and isinstance(memory["topics_discussed"], list):
                    for topic in memory["topics_discussed"]:
                        topics_seen[topic] += 1

                # Extract content types
                if "content_type" in memory:
                    content_type = memory["content_type"]
                    patterns["content_preferences"][content_type] = (
                        patterns["content_preferences"].get(content_type, 0) + 1
                    )

        if topics_seen:
            patterns["content_preferences"]["top_topics"] = [
                {"topic": topic, "count": count}
                for topic, count in sorted(topics_seen.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ]

        # Analyze session patterns
        session_durations = []
        for memory in user_memories:
            if isinstance(memory, dict) and "duration" in memory:
                session_durations.append(memory["duration"])

        if session_durations:
            patterns["session_patterns"]["avg_duration"] = sum(session_durations) / len(
                session_durations
            )
            patterns["session_patterns"]["max_duration"] = max(session_durations)
            patterns["session_patterns"]["min_duration"] = min(session_durations)

        return SkillResult(
            success=True,
            result=patterns,
            confidence=min(
                0.9, 0.5 + (len(user_memories) / 200)
            ),  # More memories = higher confidence
            metadata={
                "memories_analyzed": len(user_memories),
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


class SystemUtilitySkill(BaseSkill):
    """Skill for system management and utility functions."""

    def __init__(self, container: Container):
        super().__init__(container)

        # Required components
        self.event_bus = container.get(EventBus)
        self.metrics = container.get(MetricsCollector)
        self.config = container.get(ConfigLoader)
        self.health_check = container.get(HealthCheck)
        self.database = self._get_component(DatabaseManager)
        self.cache = self._get_component(CacheStrategy)

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="system_utility",
            name="System Utility",
            description="Provides system management and utility functions",
            category=SkillCategory.UTILITY,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (health_check, get_metrics, clear_cache, get_config, purge_data)",
                    "required": True,
                },
                "target": {
                    "type": "string",
                    "description": "Target component or system",
                    "required": False,
                },
                "parameters": {
                    "type": "object",
                    "description": "Operation-specific parameters",
                    "required": False,
                },
            },
            examples=[
                {"operation": "health_check", "target": "all"},
                {"operation": "get_metrics", "target": "memory_usage"},
                {"operation": "clear_cache", "parameters": {"cache_type": "response_cache"}},
            ],
            required_components=["EventBus", "MetricsCollector", "ConfigLoader", "HealthCheck"],
            tags={"system", "utility", "admin", "management"},
            required_permissions=["system:admin"],
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute system utility operations."""
        operation = params["operation"]
        target = params.get("target", "all")
        parameters = params.get("parameters", {})

        if operation == "health_check":
            return await self._perform_health_check(target)
        elif operation == "get_metrics":
            return await self._get_metrics(target)
        elif operation == "clear_cache":
            return await self._clear_cache(parameters.get("cache_type"))
        elif operation == "get_config":
            return await self._get_config(target)
        elif operation == "purge_data":
            return await self._purge_data(target, parameters)
        else:
            return SkillResult(
                success=False, result=None, errors=[f"Unknown operation: {operation}"]
            )

    async def _perform_health_check(self, target: str) -> SkillResult:
        """Perform health check on system components."""
        if target == "all":
            # Get health status of all components
            health_results = await self.health_check.check_all()

            overall_health = True
            unhealthy_components = []

            for component, status in health_results.items():
                if status.get("status") != "healthy":
                    overall_health = False
                    unhealthy_components.append(
                        {
                            "component": component,
                            "status": status.get("status"),
                            "error": status.get("error"),
                        }
                    )

            return SkillResult(
                success=True,
                result={
                    "overall_health": "healthy" if overall_health else "degraded",
                    "component_status": health_results,
                    "unhealthy_components": unhealthy_components,
                },
                confidence=1.0,
                metadata={
                    "component_count": len(health_results),
                    "check_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            return SkillResult(
                success=False,
                data={"error": "Health check service not available"},
                message="Health check system not available",
            )
