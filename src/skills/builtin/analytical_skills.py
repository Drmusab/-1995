"""
Advanced Analytical Skills for AI Assistant
Author: Drmusab
Last Modified: 2025-07-05 21:33:17 UTC

This module provides comprehensive analytical capabilities for the AI assistant,
including data analysis, pattern recognition, text analysis, summarization,
statistical processing, trend identification, and other analytical skills
that support decision making and insight generation.
"""

import json
import logging
import math
import re
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import asyncio
import numpy as np
import pandas as pd

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    AnalysisCompleted,
    AnalysisError,
    AnalysisStarted,
    LearningEventOccurred,
    SkillExecutionCompleted,
    SkillExecutionFailed,
    SkillExecutionStarted,
)

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager
from src.integrations.storage.file_storage import FileStorage

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor

# Memory and context
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.processing.natural_language.entity_extractor import EntityExtractor

# Processing components
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.reasoning.decision_making.decision_tree import DecisionTree
from src.reasoning.inference_engine import InferenceEngine
from src.reasoning.knowledge_graph import KnowledgeGraph

# Reasoning
from src.reasoning.logic_engine import LogicEngine


class AnalysisType(Enum):
    """Types of analysis that can be performed."""

    TEXT = "text"
    NUMERIC = "numeric"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    COMPARATIVE = "comparative"
    SENTIMENT = "sentiment"
    CATEGORICAL = "categorical"
    CORRELATION = "correlation"
    CLUSTER = "cluster"
    ANOMALY = "anomaly"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    DIAGNOSTIC = "diagnostic"
    DECISION = "decision"


class AnalysisLevel(Enum):
    """Levels of analysis depth."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class OutputFormat(Enum):
    """Output formats for analysis results."""

    TEXT = "text"
    JSON = "json"
    TABLE = "table"
    CHART = "chart"
    GRAPH = "graph"
    SUMMARY = "summary"
    REPORT = "report"
    DASHBOARD = "dashboard"


class AnalyticalSkill(ABC):
    """Abstract base class for all analytical skills."""

    def __init__(self, container: Container):
        """
        Initialize the analytical skill.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Core components
        self.model_router = container.get(ModelRouter)
        self.language_chain = container.get(LanguageChain)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)
        self.entity_extractor = container.get(EntityExtractor)

        # Memory and context
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)

        # Reasoning components
        self.logic_engine = container.get(LogicEngine)
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.inference_engine = container.get(InferenceEngine)
        self.decision_tree = container.get(DecisionTree)

        # Learning components
        self.continual_learner = container.get(ContinualLearner)

        # Storage components
        try:
            self.database = container.get(DatabaseManager)
            self.file_storage = container.get(FileStorage)
        except Exception as e:
            self.logger.warning(f"Storage components not available: {str(e)}")
            self.database = None
            self.file_storage = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Register metrics
        if self.metrics:
            self.metrics.register_counter(f"skill_{self.get_skill_id()}_executions_total")
            self.metrics.register_histogram(f"skill_{self.get_skill_id()}_execution_time_seconds")
            self.metrics.register_counter(f"skill_{self.get_skill_id()}_errors_total")

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the analytical skill."""
        pass

    @abstractmethod
    def get_skill_id(self) -> str:
        """Get the unique skill identifier."""
        pass

    @abstractmethod
    def get_skill_description(self) -> str:
        """Get the skill description."""
        pass

    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the skill parameters schema."""
        return {}

    def get_skill_examples(self) -> List[Dict[str, Any]]:
        """Get examples of skill usage."""
        return []

    def get_skill_category(self) -> str:
        """Get the skill category."""
        return "analytical"

    async def _track_execution(self, execution_id: str, **kwargs) -> None:
        """Track skill execution in memory for learning."""
        try:
            # Store execution data
            execution_data = {
                "skill_id": self.get_skill_id(),
                "execution_id": execution_id,
                "timestamp": datetime.now(timezone.utc),
                "parameters": kwargs,
                "category": self.get_skill_category(),
            }

            await self.episodic_memory.store(execution_data)

            # Update learning system
            await self.continual_learner.learn_from_skill_execution(execution_data)

        except Exception as e:
            self.logger.warning(f"Failed to track skill execution: {str(e)}")


class TextAnalyzer(AnalyticalSkill):
    """Advanced text analysis skill for extracting insights from text."""

    def get_skill_id(self) -> str:
        return "text_analyzer"

    def get_skill_description(self) -> str:
        return "Analyzes text to extract insights, themes, patterns, sentiment, entities, and key information."

    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "text": {"type": "string", "description": "Text content to analyze", "required": True},
            "analysis_types": {
                "type": "array",
                "description": "Types of analysis to perform",
                "items": {
                    "type": "string",
                    "enum": [
                        "keywords",
                        "themes",
                        "sentiment",
                        "entities",
                        "readability",
                        "structure",
                        "summary",
                        "discourse",
                        "complexity",
                        "all",
                    ],
                },
                "default": ["keywords", "themes", "sentiment", "summary"],
            },
            "depth": {
                "type": "string",
                "description": "Analysis depth level",
                "enum": [level.value for level in AnalysisLevel],
                "default": AnalysisLevel.INTERMEDIATE.value,
            },
            "output_format": {
                "type": "string",
                "description": "Format of the output",
                "enum": ["text", "json", "table"],
                "default": "json",
            },
            "max_keywords": {
                "type": "integer",
                "description": "Maximum number of keywords to extract",
                "default": 10,
            },
            "summary_length": {
                "type": "string",
                "description": "Length of summary to generate",
                "enum": ["short", "medium", "long"],
                "default": "medium",
            },
        }

    def get_skill_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "parameters": {
                    "text": "Climate change is one of the most pressing challenges of our time...",
                    "analysis_types": ["keywords", "themes", "sentiment"],
                    "depth": "intermediate",
                },
                "description": "Analyze a text about climate change to extract keywords, themes, and sentiment",
            },
            {
                "parameters": {
                    "text": "The company's quarterly financial report indicates a 15% growth in revenue...",
                    "analysis_types": ["summary", "entities", "keywords"],
                    "summary_length": "short",
                },
                "description": "Create a short summary of a financial report and extract key entities and keywords",
            },
            {
                "parameters": {
                    "text": "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer...",
                    "analysis_types": ["all"],
                    "depth": "advanced",
                },
                "description": "Perform comprehensive advanced analysis on a literary text",
            },
        ]

    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze text to extract insights and information.

        Args:
            text: Text content to analyze
            analysis_types: Types of analysis to perform
            depth: Analysis depth level
            output_format: Format of the output
            max_keywords: Maximum number of keywords to extract
            summary_length: Length of summary to generate

        Returns:
            Analysis results in the specified format
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment(f"skill_{self.get_skill_id()}_executions_total")

        # Emit execution started event
        await self.event_bus.emit(
            SkillExecutionStarted(
                skill_id=self.get_skill_id(), execution_id=execution_id, parameters=kwargs
            )
        )

        # Emit analysis started event
        await self.event_bus.emit(
            AnalysisStarted(
                analysis_id=execution_id,
                analysis_type="text_analysis",
                content_size=len(kwargs.get("text", "")),
            )
        )

        try:
            with self.tracer.trace("text_analysis") if self.tracer else None:
                # Extract parameters with defaults
                text = kwargs.get("text", "")
                if not text:
                    raise ValueError("Text content is required for analysis")

                analysis_types = kwargs.get(
                    "analysis_types", ["keywords", "themes", "sentiment", "summary"]
                )
                depth = kwargs.get("depth", AnalysisLevel.INTERMEDIATE.value)
                output_format = kwargs.get("output_format", "json")
                max_keywords = kwargs.get("max_keywords", 10)
                summary_length = kwargs.get("summary_length", "medium")

                # Handle "all" analysis type
                if "all" in analysis_types:
                    analysis_types = [
                        "keywords",
                        "themes",
                        "sentiment",
                        "entities",
                        "readability",
                        "structure",
                        "summary",
                        "discourse",
                        "complexity",
                    ]

                # Initialize results container
                results = {
                    "text_stats": {
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "sentence_count": len(re.split(r"[.!?]+", text)),
                        "paragraph_count": len(text.split("\n\n")),
                    },
                    "analysis": {},
                }

                # Process each requested analysis type
                analysis_tasks = []

                if "keywords" in analysis_types:
                    analysis_tasks.append(self._extract_keywords(text, max_keywords, depth))

                if "themes" in analysis_types:
                    analysis_tasks.append(self._extract_themes(text, depth))

                if "sentiment" in analysis_types:
                    analysis_tasks.append(self._analyze_sentiment(text, depth))

                if "entities" in analysis_types:
                    analysis_tasks.append(self._extract_entities(text, depth))

                if "readability" in analysis_types:
                    analysis_tasks.append(self._analyze_readability(text))

                if "structure" in analysis_types:
                    analysis_tasks.append(self._analyze_structure(text, depth))

                if "summary" in analysis_types:
                    analysis_tasks.append(self._generate_summary(text, summary_length, depth))

                if "discourse" in analysis_types:
                    analysis_tasks.append(self._analyze_discourse(text, depth))

                if "complexity" in analysis_types:
                    analysis_tasks.append(self._analyze_complexity(text))

                # Execute analysis tasks concurrently
                analysis_results = await asyncio.gather(*analysis_tasks)

                # Merge results
                for result in analysis_results:
                    results["analysis"].update(result)

                # Format output
                if output_format == "text":
                    results = self._format_as_text(results)
                elif output_format == "table":
                    results = self._format_as_table(results)

                # Calculate execution time
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                # Add metadata to results
                results["metadata"] = {
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "analysis_types": analysis_types,
                    "depth": depth,
                }

                # Track execution for learning
                await self._track_execution(
                    execution_id,
                    analysis_types=analysis_types,
                    depth=depth,
                    text_length=len(text),
                    execution_time=execution_time,
                )

                # Emit completion events
                await self.event_bus.emit(
                    AnalysisCompleted(
                        analysis_id=execution_id,
                        execution_time=execution_time,
                        result_size=len(str(results)),
                    )
                )

                await self.event_bus.emit(
                    SkillExecutionCompleted(
                        skill_id=self.get_skill_id(),
                        execution_id=execution_id,
                        execution_time=execution_time,
                    )
                )

                if self.metrics:
                    self.metrics.record(
                        f"skill_{self.get_skill_id()}_execution_time_seconds", execution_time
                    )

                return results

        except Exception as e:
            # Handle errors
            if self.metrics:
                self.metrics.increment(f"skill_{self.get_skill_id()}_errors_total")

            await self.event_bus.emit(AnalysisError(analysis_id=execution_id, error_message=str(e)))

            await self.event_bus.emit(
                SkillExecutionFailed(
                    skill_id=self.get_skill_id(), execution_id=execution_id, error_message=str(e)
                )
            )

            self.logger.error(f"Text analysis failed: {str(e)}")
            raise

    async def _extract_keywords(self, text: str, max_keywords: int, depth: str) -> Dict[str, Any]:
        """Extract keywords and key phrases from text."""
        # Base prompt for keyword extraction
        prompt = f"Extract the top {max_keywords} most important keywords and key phrases from this text. "

        # Add depth-specific instructions
        if depth == AnalysisLevel.BASIC.value:
            prompt += "For each keyword, provide a relevance score from 0-1."
        elif depth == AnalysisLevel.INTERMEDIATE.value:
            prompt += "For each keyword, provide a relevance score from 0-1 and a brief context."
        elif depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
            prompt += (
                "For each keyword, provide a relevance score from 0-1, context, "
                "semantic category, and relation to other keywords. "
                "Also identify uncommon or domain-specific terminology."
            )

        prompt += f"\n\nText: {text[:4000]}..." if len(text) > 4000 else f"\n\nText: {text}"

        # Use analysis model for keyword extraction
        extraction_result = await self.model_router.generate_text(
            prompt, model="analysis", parameters={"temperature": 0.2}
        )

        # Process extraction result
        keywords = []
        current_keyword = None
        current_info = {}

        for line in extraction_result.split("\n"):
            line = line.strip()
            if not line:
                if current_keyword and current_info:
                    keywords.append({"keyword": current_keyword, **current_info})
                    current_keyword = None
                    current_info = {}
                continue

            # Look for a keyword pattern
            keyword_match = re.match(r'^(\d+\.\s+)?[""]?([^":]+)[""]?\s*(\(.*\))?\s*[:-]', line)
            if keyword_match:
                # Save previous keyword if exists
                if current_keyword and current_info:
                    keywords.append({"keyword": current_keyword, **current_info})
                    current_info = {}

                # Extract new keyword
                current_keyword = keyword_match.group(2).strip()

                # Extract the rest of the line
                rest = line[line.find(":") + 1 :] if ":" in line else line[line.find("-") + 1 :]

                # Check for relevance score
                score_match = re.search(
                    r"(?:relevance|score|rating)?\s*[:=]?\s*(\d+\.\d+|\d+\/\d+|\d+\s*%)",
                    rest,
                    re.IGNORECASE,
                )
                if score_match:
                    score_str = score_match.group(1)
                    if "/" in score_str:
                        num, denom = score_str.split("/")
                        current_info["relevance"] = round(float(num) / float(denom), 2)
                    elif "%" in score_str:
                        current_info["relevance"] = round(
                            float(score_str.replace("%", "")) / 100, 2
                        )
                    else:
                        current_info["relevance"] = float(score_str)
                else:
                    current_info["relevance"] = 1.0  # Default if no score found

                # Get description or context from the rest of the line
                desc = rest.strip()
                if desc and not desc.isdigit():
                    current_info["context"] = desc

            elif current_keyword:
                # Additional information for current keyword
                if "context" in line.lower() or "description" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["context"] = value
                elif "category" in line.lower() or "type" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["category"] = value
                elif "relation" in line.lower() or "related" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["relations"] = value
                elif (
                    "relevance" in line.lower()
                    or "score" in line.lower()
                    or "importance" in line.lower()
                ):
                    if ":" in line:
                        value = line.split(":", 1)[1].strip()
                        if re.match(r"\d+\.\d+|\d+\/\d+|\d+\s*%", value):
                            if "/" in value:
                                num, denom = value.split("/")
                                current_info["relevance"] = round(float(num) / float(denom), 2)
                            elif "%" in value:
                                current_info["relevance"] = round(
                                    float(value.replace("%", "")) / 100, 2
                                )
                            else:
                                current_info["relevance"] = float(value)

        # Add the last keyword if exists
        if current_keyword and current_info:
            keywords.append({"keyword": current_keyword, **current_info})

        # Limit to max_keywords
        keywords = keywords[:max_keywords]

        # Sort by relevance
        keywords = sorted(keywords, key=lambda k: k.get("relevance", 0), reverse=True)

        return {"keywords": keywords}

    async def _extract_themes(self, text: str, depth: str) -> Dict[str, Any]:
        """Extract main themes and topics from text."""
        # Base prompt for theme extraction
        prompt = "Identify the main themes and topics in this text. "

        # Add depth-specific instructions
        if depth == AnalysisLevel.BASIC.value:
            prompt += "List 3-5 main themes with brief descriptions."
        elif depth == AnalysisLevel.INTERMEDIATE.value:
            prompt += (
                "Identify 4-7 main themes. For each theme, provide a description, "
                "supporting evidence from the text, and significance."
            )
        elif depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
            prompt += (
                "Conduct a detailed thematic analysis identifying 5-10 themes. "
                "For each theme, provide a comprehensive description, textual evidence, "
                "significance, interrelationships with other themes, and potential subthemes. "
                "Also identify any underlying or implicit themes."
            )

        prompt += f"\n\nText: {text[:4000]}..." if len(text) > 4000 else f"\n\nText: {text}"

        # Use analysis model for theme extraction
        themes_result = await self.model_router.generate_text(
            prompt, model="analysis", parameters={"temperature": 0.3}
        )

        # Process themes result
        themes = []
        current_theme = None
        current_info = {}

        for line in themes_result.split("\n"):
            line = line.strip()
            if not line:
                if current_theme and current_info:
                    themes.append({"theme": current_theme, **current_info})
                    current_theme = None
                    current_info = {}
                continue

            # Look for a theme pattern
            theme_match = re.match(r"^(\d+\.\s+)?([^:]+)[:-]", line)
            if theme_match:
                # Save previous theme if exists
                if current_theme and current_info:
                    themes.append({"theme": current_theme, **current_info})
                    current_info = {}

                # Extract new theme
                current_theme = theme_match.group(2).strip()

                # Extract description from the rest of the line
                desc = line[line.find(":") + 1 :] if ":" in line else line[line.find("-") + 1 :]
                if desc.strip():
                    current_info["description"] = desc.strip()

            elif current_theme:
                # Additional information for current theme
                if "description" in line.lower() or "overview" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["description"] = value
                elif (
                    "evidence" in line.lower()
                    or "example" in line.lower()
                    or "support" in line.lower()
                ):
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["evidence"] = value
                elif "significance" in line.lower() or "importance" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["significance"] = value
                elif "relationship" in line.lower() or "connection" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    current_info["relationships"] = value
                elif "subtheme" in line.lower():
                    value = line.split(":", 1)[1].strip() if ":" in line else line
                    if "subthemes" not in current_info:
                        current_info["subthemes"] = []
                    current_info["subthemes"].append(value)
                elif not any(
                    key in current_info for key in ["description", "evidence", "significance"]
                ):
                    # If no specific category but content exists, use as description
                    if "description" not in current_info:
                        current_info["description"] = line
                    else:
                        current_info["description"] += " " + line

        # Add the last theme if exists
        if current_theme and current_info:
            themes.append({"theme": current_theme, **current_info})

        return {"themes": themes}

    async def _analyze_sentiment(self, text: str, depth: str) -> Dict[str, Any]:
        """Analyze sentiment in the text."""
        # Use the built-in sentiment analyzer for basic analysis
        basic_sentiment = await self.sentiment_analyzer.analyze(text)

        result = {
            "overall_sentiment": basic_sentiment.get("sentiment", "neutral"),
            "sentiment_score": basic_sentiment.get("confidence", 0.5),
        }

        # For more advanced analysis, use additional processing
        if depth in [
            AnalysisLevel.INTERMEDIATE.value,
            AnalysisLevel.ADVANCED.value,
            AnalysisLevel.EXPERT.value,
        ]:
            # Prompt for deeper sentiment analysis
            prompt = (
                "Perform a detailed sentiment analysis of this text. Include overall sentiment, "
                "emotional tone, intensity, key emotional phrases, and any shifts in sentiment. "
            )

            if depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
                prompt += (
                    "Also analyze underlying emotions, ambivalence, contextual factors affecting sentiment, "
                    "implicit attitudes, and provide a paragraph-by-paragraph sentiment breakdown."
                )

            prompt += f"\n\nText: {text[:4000]}..." if len(text) > 4000 else f"\n\nText: {text}"

            # Use analysis model for detailed sentiment
            detailed_sentiment = await self.model_router.generate_text(
                prompt, model="analysis", parameters={"temperature": 0.3}
            )

            # Extract structured data from the detailed analysis
            emotional_tones = []
            emotional_phrases = []
            sentiment_shifts = []
            paragraph_sentiments = []

            # Simple extraction of key elements
            sections = detailed_sentiment.split("\n\n")
            for section in sections:
                section = section.lower()
                if "tone" in section or "emotion" in section:
                    tones = re.findall(
                        r"(joy|sadness|anger|fear|surprise|disgust|trust|anticipation|optimism|pessimism|anxiety|confidence|confusion|nostalgia)",
                        section,
                    )
                    emotional_tones.extend(tones)

                if "phrase" in section or "expression" in section:
                    # Extract quoted phrases
                    phrases = re.findall(r'"([^"]+)"', section)
                    emotional_phrases.extend(phrases)

                if "shift" in section or "change" in section:
                    shifts = [
                        line.strip() for line in section.split("\n") if line.strip() and ":" in line
                    ]
                    sentiment_shifts.extend(shifts)

                if "paragraph" in section and "sentiment" in section:
                    para_sents = [
                        line.strip() for line in section.split("\n") if line.strip() and ":" in line
                    ]
                    paragraph_sentiments.extend(para_sents)

            # Add detailed results
            result.update(
                {
                    "emotional_tones": list(set(emotional_tones)),
                    "emotional_phrases": emotional_phrases[:5],  # Limit to avoid overwhelming
                    "sentiment_shifts": sentiment_shifts,
                    "paragraph_analysis": paragraph_sentiments,
                }
            )

        return {"sentiment": result}

    async def _extract_entities(self, text: str, depth: str) -> Dict[str, Any]:
        """Extract entities from the text."""
        # Use entity extractor for basic entities
        basic_entities = await self.entity_extractor.extract(text)

        # For more comprehensive entity analysis
        if depth in [
            AnalysisLevel.INTERMEDIATE.value,
            AnalysisLevel.ADVANCED.value,
            AnalysisLevel.EXPERT.value,
        ]:
            # Prompt for detailed entity extraction
            prompt = (
                "Extract and categorize all entities from this text. Include people, organizations, "
                "locations, dates, quantities, events, and concepts. "
            )

            if depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
                prompt += (
                    "For each entity, provide context, significance, relationships to other entities, "
                    "and any attributes or qualifiers. Also identify ambiguous or implicit entities."
                )

            prompt += f"\n\nText: {text[:4000]}..." if len(text) > 4000 else f"\n\nText: {text}"

            # Use analysis model for detailed entity extraction
            detailed_entities = await self.model_router.generate_text(
                prompt, model="analysis", parameters={"temperature": 0.2}
            )

            # Process the detailed entity results
            entity_categories = {
                "people": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "quantities": [],
                "events": [],
                "concepts": [],
                "other": [],
            }

            current_category = None

            for line in detailed_entities.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # Check for category headers
                category_match = re.match(
                    r"^(people|persons?|individuals?|organizations?|companies|locations?|places|dates?|times?|quantities|numbers|events?|concepts?|other)[:\-]",
                    line.lower(),
                )
                if category_match:
                    category = category_match.group(1).lower()
                    if category in ["people", "persons", "individuals"]:
                        current_category = "people"
                    elif category in ["organizations", "companies"]:
                        current_category = "organizations"
                    elif category in ["locations", "places"]:
                        current_category = "locations"
                    elif category in ["dates", "times"]:
                        current_category = "dates"
                    elif category in ["quantities", "numbers"]:
                        current_category = "quantities"
                    elif category in ["events"]:
                        current_category = "events"
                    elif category in ["concepts"]:
                        current_category = "concepts"
                    else:
                        current_category = "other"
                    continue

                # Process entity lines if in a category
                if current_category and line:
                    # Look for entity with details
                    entity_match = re.match(r"^[•\-\*]?\s*([^:]+)(?::(.+))?$", line)
                    if entity_match:
                        entity_name = entity_match.group(1).strip()
                        entity_details = (
                            entity_match.group(2).strip() if entity_match.group(2) else None
                        )

                        entity_info = {"name": entity_name}
                        if entity_details:
                            entity_info["details"] = entity_details

                        entity_categories[current_category].append(entity_info)

            # Combine basic and detailed results
            combined_entities = {
                "basic": basic_entities,
                "detailed": {k: v for k, v in entity_categories.items() if v},
            }

            return {"entities": combined_entities}
        else:
            # Just return basic entities for basic analysis
            return {"entities": basic_entities}

    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability."""
        # Basic text statistics
        word_count = len(text.split())
        sentence_count = len(re.split(r"[.!?]+", text))
        syllable_pattern = re.compile(r"[aeiouy]+", re.IGNORECASE)
        syllable_count = len(syllable_pattern.findall(text))

        # Calculate readability scores
        try:
            # Flesch Reading Ease
            if sentence_count > 0:
                flesch_reading_ease = (
                    206.835
                    - 1.015 * (word_count / sentence_count)
                    - 84.6 * (syllable_count / word_count)
                )
            else:
                flesch_reading_ease = 0

            # Flesch-Kincaid Grade Level
            if sentence_count > 0:
                flesch_kincaid_grade = (
                    0.39 * (word_count / sentence_count)
                    + 11.8 * (syllable_count / word_count)
                    - 15.59
                )
            else:
                flesch_kincaid_grade = 0

            # Approximate reading time (words per minute)
            avg_reading_speed = 200  # words per minute
            reading_time_minutes = word_count / avg_reading_speed

            return {
                "readability": {
                    "flesch_reading_ease": round(flesch_reading_ease, 2),
                    "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
                    "estimated_reading_time_minutes": round(reading_time_minutes, 2),
                    "readability_interpretation": self._interpret_readability(flesch_reading_ease),
                }
            }
        except Exception as e:
            self.logger.warning(f"Error calculating readability scores: {str(e)}")
            return {"readability": {"error": "Could not calculate readability scores"}}

    def _interpret_readability(self, score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very easy to read. Easily understood by an average 11-year-old student."
        elif score >= 80:
            return "Easy to read. Conversational English for consumers."
        elif score >= 70:
            return "Fairly easy to read."
        elif score >= 60:
            return "Plain English. Easily understood by 13- to 15-year-old students."
        elif score >= 50:
            return "Fairly difficult to read."
        elif score >= 30:
            return "Difficult to read. Best understood by college graduates."
        else:
            return "Very difficult to read. Best understood by university graduates."

    async def _analyze_structure(self, text: str, depth: str) -> Dict[str, Any]:
        """Analyze text structure and organization."""
        # Prompt for structure analysis
        prompt = "Analyze the structure and organization of this text. "

        if depth == AnalysisLevel.BASIC.value:
            prompt += "Identify main sections and organizational patterns."
        elif depth == AnalysisLevel.INTERMEDIATE.value:
            prompt += (
                "Identify main sections, organizational patterns, transition techniques, "
                "and evaluate the overall coherence and flow."
            )
        elif depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
            prompt += (
                "Provide a detailed analysis of text structure including: main sections, subsections, "
                "organizational patterns, transition techniques, logical flow, argument structure, "
                "narrative progression, and structural weaknesses or strengths."
            )

        prompt += f"\n\nText: {text[:4000]}..." if len(text) > 4000 else f"\n\nText: {text}"

        # Use analysis model for structure analysis
        structure_analysis = await self.model_router.generate_text(
            prompt, model="analysis", parameters={"temperature": 0.3}
        )

        # Extract key components of the structure analysis
        sections = []
        organization_pattern = ""
        transitions = []
        structural_elements = {}

        # Process analysis results
        for line in structure_analysis.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Look for sections
            if "section" in line.lower() and ":" in line:
                sections.append(line)

            # Look for organizational pattern
            if "pattern" in line.lower() and ":" in line:
                organization_pattern = line.split(":", 1)[1].strip()

            # Look for transitions
            if "transition" in line.lower() and ":" in line:
                transitions.append(line)

            # Look for other structural elements
            structure_types = [
                "introduction",
                "body",
                "conclusion",
                "argument",
                "flow",
                "coherence",
                "narrative",
                "progression",
                "sequence",
                "hierarchy",
            ]

            for struct_type in structure_types:
                if struct_type in line.lower() and ":" in line:
                    key = struct_type.lower()
                    value = line.split(":", 1)[1].strip()
                    structural_elements[key] = value

        return {
            "structure": {
                "sections": sections,
                "organization_pattern": organization_pattern,
                "transitions": transitions,
                "structural_elements": structural_elements,
            }
        }

    async def _generate_summary(self, text: str, length: str, depth: str) -> Dict[str, Any]:
        """Generate a summary of the text."""
        # Determine target summary length
        if length == "short":
            target_length = "1-2 paragraphs"
            max_tokens = 150
        elif length == "medium":
            target_length = "3-4 paragraphs"
            max_tokens = 300
        else:  # long
            target_length = "5-6 paragraphs"
            max_tokens = 500

        # Prompt for summary generation
        prompt = f"Summarize the following text in {target_length}. "

        if depth == AnalysisLevel.BASIC.value:
            prompt += "Focus on the main points and key information."
        elif depth == AnalysisLevel.INTERMEDIATE.value:
            prompt += (
                "Include main points, key arguments, and significant details. "
                "Maintain the original tone and perspective."
            )
        elif depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
            prompt += (
                "Provide a comprehensive summary that captures main points, key arguments, "
                "supporting evidence, nuances, and implications. "
                "Preserve the original tone, maintain logical flow, and reflect the text's complexity. "
                "Also include a one-sentence 'key takeaway' at the end."
            )

        prompt += f"\n\nText: {text[:6000]}..." if len(text) > 6000 else f"\n\nText: {text}"

        # Generate summary
        summary = await self.model_router.generate_text(
            prompt, model="analysis", parameters={"temperature": 0.4, "max_tokens": max_tokens}
        )

        # Extract key takeaway if present
        key_takeaway = ""
        summary_text = summary

        # Look for "Key Takeaway" section
        if "key takeaway" in summary.lower() or "key take away" in summary.lower():
            parts = re.split(r"key\s+take(?:[\-\s])?away\s*[:;]", summary, flags=re.IGNORECASE)
            if len(parts) > 1:
                key_takeaway = parts[-1].strip()
                summary_text = parts[0].strip()

        return {"summary": {"text": summary_text, "key_takeaway": key_takeaway, "length": length}}

    async def _analyze_discourse(self, text: str, depth: str) -> Dict[str, Any]:
        """Analyze discourse patterns and rhetorical strategies."""
        # Skip for basic analysis
        if depth == AnalysisLevel.BASIC.value:
            return {}

        # Prompt for discourse analysis
        prompt = "Analyze the discourse patterns and rhetorical strategies in this text. "

        if depth == AnalysisLevel.INTERMEDIATE.value:
            prompt += (
                "Identify rhetorical devices, persuasion techniques, tone, voice, "
                "and audience considerations."
            )
        elif depth in [AnalysisLevel.ADVANCED.value, AnalysisLevel.EXPERT.value]:
            prompt += (
                "Provide a comprehensive discourse analysis including: rhetorical devices, "
                "persuasion techniques, argumentation strategies, logical fallacies, "
                "ethos/pathos/logos appeals, tone, voice, register, stance, audience considerations, "
                "intertextuality, and ideological positioning."
            )

        prompt += f"\n\nText: {text[:4000]}..." if len(text) > 4000 else f"\n\nText: {text}"

        # Generate discourse analysis
        discourse_analysis = await self.model_router.generate_text(
            prompt, model="analysis", parameters={"temperature": 0.3}
        )

        # Process the analysis into structured data
        rhetorical_devices = []
        persuasion_techniques = []
        tone_voice = {}
        argumentation = {}
        audience = {}

        # Extract structured data
        current_section = None

        for line in discourse_analysis.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            section_match = re.match(
                r"^(rhetorical devices|persuasion techniques|tone|voice|audience|argumentation|appeals|stance|register|fallacies)[:\-]",
                line.lower(),
            )
            if section_match:
                section = section_match.group(1).lower()
                if section in ["rhetorical devices"]:
                    current_section = "rhetorical_devices"
                elif section in ["persuasion techniques"]:
                    current_section = "persuasion_techniques"
                elif section in ["tone", "voice", "register"]:
                    current_section = "tone_voice"
                elif section in ["audience"]:
                    current_section = "audience"
                elif section in ["argumentation", "appeals", "stance", "fallacies"]:
                    current_section = "argumentation"
                continue

            # Process items based on section
            if current_section == "rhetorical_devices":
                if line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line):
                    device = line[line.find(" ") + 1 :] if re.match(r"^[\-\*\d]", line) else line
                    rhetorical_devices.append(device)
            elif current_section == "persuasion_techniques":
                if line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line):
                    technique = line[line.find(" ") + 1 :] if re.match(r"^[\-\*\d]", line) else line
                    persuasion_techniques.append(technique)
            elif current_section == "tone_voice":
                if ":" in line:
                    key, value = line.split(":", 1)
                    tone_voice[key.strip().lower()] = value.strip()
                elif line:
                    # If no specific key, add to general tone description
                    if "description" not in tone_voice:
                        tone_voice["description"] = line
                    else:
                        tone_voice["description"] += " " + line
            elif current_section == "audience":
                if ":" in line:
                    key, value = line.split(":", 1)
                    audience[key.strip().lower()] = value.strip()
                elif line:
                    if "description" not in audience:
                        audience["description"] = line
                    else:
                        audience["description"] += " " + line
            elif current_section == "argumentation":
                if ":" in line:
                    key, value = line.split(":", 1)
                    argumentation[key.strip().lower()] = value.strip()
                elif line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line):
                    item = line[line.find(" ") + 1 :] if re.match(r"^[\-\*\d]", line) else line
                    if "strategies" not in argumentation:
                        argumentation["strategies"] = [item]
                    else:
                        if isinstance(argumentation["strategies"], list):
                            argumentation["strategies"].append(item)
                        else:
                            argumentation["strategies"] = [argumentation["strategies"], item]

        return {
            "discourse": {
                "rhetorical_devices": rhetorical_devices,
                "persuasion_techniques": persuasion_techniques,
                "tone_and_voice": tone_voice,
                "argumentation": argumentation,
                "audience": audience,
            }
        }

    async def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic complexity."""
        # Calculate basic complexity metrics
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        # Sentence length
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        )

        # Word length
        word_lengths = [len(w) for w in words]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0

        # Unique words ratio
        unique_words = set(w.lower() for w in words)
        lexical_diversity = len(unique_words) / len(words) if words else 0

        # Long words percentage (>6 chars)
        long_words = [w for w in words if len(w) > 6]
        long_words_percentage = len(long_words) / len(words) if words else 0

        # Very long sentences (>25 words)
        long_sentences = [s for s in sentence_lengths if s > 25]
        long_sentences_percentage = (
            len(long_sentences) / len(sentence_lengths) if sentence_lengths else 0
        )

        return {
            "complexity": {
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2),
                "lexical_diversity": round(lexical_diversity, 2),
                "long_words_percentage": round(long_words_percentage * 100, 2),
                "long_sentences_percentage": round(long_sentences_percentage * 100, 2),
                "complexity_interpretation": self._interpret_complexity(
                    avg_sentence_length, avg_word_length, lexical_diversity
                ),
            }
        }

    def _interpret_complexity(
        self, avg_sentence_length: float, avg_word_length: float, lexical_diversity: float
    ) -> str:
        """Interpret text complexity based on metrics."""
        # Weighted complexity score
        complexity_score = (
            0.4 * (avg_sentence_length / 25)  # Normalize to ~1 at 25 words
            + 0.3 * (avg_word_length / 6)  # Normalize to ~1 at 6 chars
            + 0.3 * lexical_diversity  # Already 0-1
        )

        if complexity_score < 0.4:
            return "Simple - Easy to read with short sentences and common vocabulary."
        elif complexity_score < 0.6:
            return "Moderate - Balanced sentence length and vocabulary diversity."
        elif complexity_score < 0.8:
            return "Advanced - Longer sentences and diverse vocabulary. May require focus to comprehend."
        else:
            return "Complex - Long sentences with sophisticated vocabulary. Requires high reading comprehension."

    def _format_as_text(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis results as readable text."""
        formatted_text = "TEXT ANALYSIS RESULTS\n\n"

        # Text stats
        stats = results.get("text_stats", {})
        formatted_text += "TEXT STATISTICS\n"
        formatted_text += f"- Character count: {stats.get('character_count', 0)}\n"
        formatted_text += f"- Word count: {stats.get('word_count', 0)}\n"
        formatted_text += f"- Sentence count: {stats.get('sentence_count', 0)}\n"
        formatted_text += f"- Paragraph count: {stats.get('paragraph_count', 0)}\n\n"

        # Process each analysis type
        analysis = results.get("analysis", {})

        # Keywords
        if "keywords" in analysis:
            formatted_text += "KEYWORDS\n"
            for kw in analysis["keywords"]:
                formatted_text += f"- {kw['keyword']}"
                if "relevance" in kw:
                    formatted_text += f" (relevance: {kw['relevance']})"
                if "context" in kw:
                    formatted_text += f": {kw['context']}"
                formatted_text += "\n"
            formatted_text += "\n"

        # Themes
        if "themes" in analysis:
            formatted_text += "THEMES\n"
            for theme in analysis["themes"]:
                formatted_text += f"- {theme['theme']}"
                if "description" in theme:
                    formatted_text += f": {theme['description']}"
                formatted_text += "\n"
            formatted_text += "\n"

        # Sentiment
        if "sentiment" in analysis:
            sentiment = analysis["sentiment"]
            formatted_text += "SENTIMENT ANALYSIS\n"
            formatted_text += (
                f"- Overall sentiment: {sentiment.get('overall_sentiment', 'neutral')}\n"
            )
            formatted_text += f"- Sentiment score: {sentiment.get('sentiment_score', 0.5)}\n"

            if "emotional_tones" in sentiment:
                formatted_text += f"- Emotional tones: {', '.join(sentiment['emotional_tones'])}\n"

            if "emotional_phrases" in sentiment and sentiment["emotional_phrases"]:
                formatted_text += "- Key emotional phrases:\n"
                for phrase in sentiment["emotional_phrases"]:
                    formatted_text += f'  * "{phrase}"\n'

            formatted_text += "\n"

        # Summary
        if "summary" in analysis:
            summary = analysis["summary"]
            formatted_text += "SUMMARY\n"
            formatted_text += f"{summary['text']}\n\n"

            if summary.get("key_takeaway"):
                formatted_text += f"Key Takeaway: {summary['key_takeaway']}\n\n"

        # Add other analysis sections with similar formatting...

        # Add metadata
        metadata = results.get("metadata", {})
        if metadata:
            formatted_text += "ANALYSIS METADATA\n"
            formatted_text += f"- Execution ID: {metadata.get('execution_id', 'unknown')}\n"
            formatted_text += f"- Execution time: {metadata.get('execution_time', 0):.2f} seconds\n"
            formatted_text += f"- Timestamp: {metadata.get('timestamp', 'unknown')}\n"

        # Return with text format instead of JSON structure
        return {"text": formatted_text}

    def _format_as_table(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis results as tables for display."""
        tables = {}

        # Text stats table
        stats = results.get("text_stats", {})
        tables["text_statistics"] = {
            "headers": ["Metric", "Value"],
            "rows": [
                ["Character count", stats.get("character_count", 0)],
                ["Word count", stats.get("word_count", 0)],
                ["Sentence count", stats.get("sentence_count", 0)],
                ["Paragraph count", stats.get("paragraph_count", 0)],
            ],
        }

        # Process each analysis type
        analysis = results.get("analysis", {})

        # Keywords table
        if "keywords" in analysis:
            headers = ["Keyword", "Relevance"]
            if any("context" in kw for kw in analysis["keywords"]):
                headers.append("Context")

            rows = []
            for kw in analysis["keywords"]:
                row = [kw["keyword"], kw.get("relevance", "N/A")]
                if "Context" in headers:
                    row.append(kw.get("context", ""))
                rows.append(row)

            tables["keywords"] = {"headers": headers, "rows": rows}

        # Themes table
        if "themes" in analysis:
            headers = ["Theme", "Description"]
            rows = [[theme["theme"], theme.get("description", "")] for theme in analysis["themes"]]

            tables["themes"] = {"headers": headers, "rows": rows}

        # Sentiment table
        if "sentiment" in analysis:
            sentiment = analysis["sentiment"]
            tables["sentiment"] = {
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Overall sentiment", sentiment.get("overall_sentiment", "neutral")],
                    ["Sentiment score", sentiment.get("sentiment_score", 0.5)],
                ],
            }

            if "emotional_tones" in sentiment:
                tables["sentiment"]["rows"].append(
                    ["Emotional tones", ", ".join(sentiment["emotional_tones"])]
                )

        # Return with table format
        return {"tables": tables, "metadata": results.get("metadata", {})}


class DataAnalyzer(AnalyticalSkill):
    """Advanced data analysis and visualization skill."""

    def get_skill_id(self) -> str:
        return "data_analyzer"

    def get_skill_description(self) -> str:
        return "Analyzes numerical and tabular data to extract insights, identify patterns, generate statistics, and visualize trends."

    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "data": {
                "type": "string",
                "description": "Data input (CSV, JSON, or tabular text)",
                "required": True,
            },
            "format": {
                "type": "string",
                "description": "Format of the input data",
                "enum": ["csv", "json", "table", "auto"],
                "default": "auto",
            },
            "analysis_types": {
                "type": "array",
                "description": "Types of analysis to perform",
                "items": {
                    "type": "string",
                    "enum": [
                        "descriptive",
                        "distribution",
                        "correlation",
                        "comparison",
                        "trends",
                        "outliers",
                        "clustering",
                        "all",
                    ],
                },
                "default": ["descriptive", "trends"],
            },
            "target_columns": {
                "type": "array",
                "description": "Specific columns to focus on (leave empty for all)",
                "items": {"type": "string"},
                "default": [],
            },
            "visualizations": {
                "type": "array",
                "description": "Types of visualizations to generate",
                "items": {"type": "string", "enum": ["none", "charts", "tables", "text", "all"]},
                "default": ["tables", "text"],
            },
            "depth": {
                "type": "string",
                "description": "Analysis depth level",
                "enum": [level.value for level in AnalysisLevel],
                "default": AnalysisLevel.INTERMEDIATE.value,
            },
        }

    def get_skill_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "parameters": {
                    "data": "date,sales,region\n2023-01-01,1200,North\n2023-01-02,1450,South\n...",
                    "format": "csv",
                    "analysis_types": ["descriptive", "trends"],
                    "target_columns": ["sales"],
                },
                "description": "Analyze sales data to generate descriptive statistics and identify trends",
            },
            {
                "parameters": {
                    "data": '[{"product":"A","price":29.99,"units":150},{"product":"B","price":49.99,"units":75}]',
                    "format": "json",
                    "analysis_types": ["comparison", "distribution"],
                    "visualizations": ["charts", "tables"],
                },
                "description": "Compare products by price and units sold with charts and tables",
            },
        ]

    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze data to extract insights, statistics, and visualizations.

        Args:
            data: Data input (CSV, JSON, or tabular text)
            format: Format of the input data
            analysis_types: Types of analysis to perform
            target_columns: Specific columns to focus on
            visualizations: Types of visualizations to generate
            depth: Analysis depth level

        Returns:
            Analysis results and insights
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment(f"skill_{self.get_skill_id()}_executions_total")

        # Emit execution started event
        await self.event_bus.emit(
            SkillExecutionStarted(
                skill_id=self.get_skill_id(), execution_id=execution_id, parameters=kwargs
            )
        )

        # Emit analysis started event
        await self.event_bus.emit(
            AnalysisStarted(
                analysis_id=execution_id,
                analysis_type="data_analysis",
                content_size=len(kwargs.get("data", "")),
            )
        )

        try:
            with self.tracer.trace("data_analysis") if self.tracer else None:
                # Extract parameters with defaults
                data_input = kwargs.get("data", "")
                if not data_input:
                    raise ValueError("Data input is required for analysis")

                data_format = kwargs.get("format", "auto")
                analysis_types = kwargs.get("analysis_types", ["descriptive", "trends"])
                target_columns = kwargs.get("target_columns", [])
                visualizations = kwargs.get("visualizations", ["tables", "text"])
                depth = kwargs.get("depth", AnalysisLevel.INTERMEDIATE.value)

                # Handle "all" analysis type
                if "all" in analysis_types:
                    analysis_types = [
                        "descriptive",
                        "distribution",
                        "correlation",
                        "comparison",
                        "trends",
                        "outliers",
                        "clustering",
                    ]

                # Handle "all" visualizations
                if "all" in visualizations:
                    visualizations = ["charts", "tables", "text"]

                # Parse the data
                df = await self._parse_data(data_input, data_format)

                # Focus on target columns if specified
                if target_columns:
                    available_columns = [col for col in target_columns if col in df.columns]
                    if not available_columns:
                        raise ValueError(
                            f"None of the target columns {target_columns} found in data"
                        )
                    df = df[available_columns]

                # Categorize columns by data type
                column_types = self._categorize_columns(df)

                # Initialize results container
                results = {
                    "data_summary": {
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "column_types": column_types,
                        "column_names": list(df.columns),
                    },
                    "analysis": {},
                }

                # Process each requested analysis type
                analysis_tasks = []

                if "descriptive" in analysis_types:
                    analysis_tasks.append(self._descriptive_analysis(df, depth))

                if "distribution" in analysis_types:
                    analysis_tasks.append(self._distribution_analysis(df, depth))

                if "correlation" in analysis_types:
                    analysis_tasks.append(self._correlation_analysis(df, depth))

                if "comparison" in analysis_types:
                    analysis_tasks.append(self._comparison_analysis(df, depth))

                if "trends" in analysis_types:
                    analysis_tasks.append(self._trend_analysis(df, depth))

                if "outliers" in analysis_types:
                    analysis_tasks.append(self._outlier_analysis(df, depth))

                if "clustering" in analysis_types and depth != AnalysisLevel.BASIC.value:
                    analysis_tasks.append(self._clustering_analysis(df, depth))

                # Execute analysis tasks concurrently
                analysis_results = await asyncio.gather(*analysis_tasks)

                # Merge results
                analysis_data = results["analysis"]
                for result in analysis_results:
                    if result:
                        analysis_data.update(result)

                # Generate visualizations if requested
                if visualizations:
                    visualization_results = await self._generate_visualizations(
                        df, analysis_data, visualizations, depth
                    )
                    results["visualizations"] = visualization_results

                return results

        except Exception as e:
            self.logger.error(f"Error in data analysis: {str(e)}")
            return {"error": str(e), "success": False}
