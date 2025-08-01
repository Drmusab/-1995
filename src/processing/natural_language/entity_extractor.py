"""
Advanced Named Entity Extraction System
Author: Drmusab
Last Modified: 2025-05-27 14:30:00 UTC

This module provides comprehensive named entity recognition and extraction capabilities
for the AI assistant, supporting multiple extraction strategies, custom entity types,
and contextual understanding.
"""

import hashlib
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union

import asyncio

# Third-party imports
import numpy as np
import spacy
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentFailed,
    ComponentInitialized,
    LearningEvent,
    ProcessingCompleted,
    ProcessingFailed,
)
from src.core.health_check import HealthCheck
from src.integrations.cache.cache_strategy import CacheStrategy

# Integration imports
from src.integrations.llm.model_router import ModelRouter

# Learning
from src.learning.feedback_processor import FeedbackProcessor
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Processing imports
from src.processing.natural_language.tokenizer import EnhancedTokenizer, TokenizationStrategy


class EntityType(Enum):
    """Standard entity types supported by the system."""

    # Person-related entities
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"

    # Temporal entities
    DATE = "DATE"
    TIME = "TIME"
    DURATION = "DURATION"

    # Numeric entities
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"

    # Geographic entities
    GPE = "GPE"  # Geopolitical entity
    FACILITY = "FAC"
    LANGUAGE = "LANGUAGE"
    NATIONALITY = "NORP"  # Nationalities, religious groups

    # Product and work entities
    PRODUCT = "PRODUCT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    EVENT = "EVENT"

    # Technical entities
    EMAIL = "EMAIL"
    URL = "URL"
    PHONE = "PHONE"
    IP_ADDRESS = "IP_ADDRESS"
    HASHTAG = "HASHTAG"
    MENTION = "MENTION"

    # Domain-specific entities
    SKILL = "SKILL"
    INTENT = "INTENT"
    EMOTION = "EMOTION"
    SENTIMENT = "SENTIMENT"

    # Custom entities
    CUSTOM = "CUSTOM"
    UNKNOWN = "UNKNOWN"


class ExtractionStrategy(Enum):
    """Named entity extraction strategies."""

    SPACY = "spacy"  # spaCy NLP library
    TRANSFORMERS = "transformers"  # HuggingFace transformers
    REGEX = "regex"  # Regular expression based
    DICTIONARY = "dictionary"  # Dictionary/gazetteer based
    HYBRID = "hybrid"  # Combination of multiple strategies
    ML_CUSTOM = "ml_custom"  # Custom ML models
    RULE_BASED = "rule_based"  # Rule-based extraction
    CONTEXTUAL = "contextual"  # Context-aware extraction


class ExtractionMode(Enum):
    """Entity extraction processing modes."""

    FAST = "fast"  # Quick extraction with basic accuracy
    BALANCED = "balanced"  # Balance between speed and accuracy
    COMPREHENSIVE = "comprehensive"  # Deep extraction with high accuracy
    REAL_TIME = "real_time"  # Optimized for real-time processing
    BATCH = "batch"  # Optimized for batch processing
    STREAMING = "streaming"  # Stream processing mode


class ConfidenceLevel(Enum):
    """Entity extraction confidence levels."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MEDIUM = "medium"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0


class EntityCategory(Enum):
    """High-level entity categories."""

    NAMED = "named"  # Named entities (persons, places, etc.)
    NUMERIC = "numeric"  # Numeric values and quantities
    TEMPORAL = "temporal"  # Time-related entities
    TECHNICAL = "technical"  # Technical identifiers
    CONCEPTUAL = "conceptual"  # Abstract concepts
    DOMAIN_SPECIFIC = "domain_specific"  # Domain-specific entities


@dataclass
class EntityPattern:
    """Pattern definition for entity matching."""

    pattern_id: str
    entity_type: EntityType
    pattern: str  # Regex pattern or spaCy pattern
    confidence_weight: float = 1.0
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    is_active: bool = True


@dataclass
class EntityRule:
    """Rule definition for entity extraction."""

    rule_id: str
    entity_type: EntityType
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    priority: int = 5
    is_enabled: bool = True


@dataclass
class Entity:
    """Represents an extracted named entity."""

    text: str
    entity_type: EntityType
    start_char: int
    end_char: int
    confidence: float

    # Categorization
    category: EntityCategory = EntityCategory.NAMED
    subcategory: Optional[str] = None

    # Normalization and linking
    normalized_value: Optional[Any] = None
    canonical_form: Optional[str] = None
    entity_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None

    # Context and relationships
    context: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[str] = None

    # Metadata
    extraction_method: ExtractionStrategy = ExtractionStrategy.SPACY
    model_used: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    certainty_score: float = 0.0
    consistency_score: float = 0.0

    # Temporal information
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "normalized_value": self.normalized_value,
            "canonical_form": self.canonical_form,
            "entity_id": self.entity_id,
            "knowledge_base_id": self.knowledge_base_id,
            "context": self.context,
            "relationships": self.relationships,
            "sentiment": self.sentiment,
            "extraction_method": self.extraction_method.value,
            "model_used": self.model_used,
            "features": self.features,
            "certainty_score": self.certainty_score,
            "consistency_score": self.consistency_score,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class ExtractionRequest:
    """Request configuration for entity extraction."""

    text: str
    strategy: ExtractionStrategy = ExtractionStrategy.HYBRID
    mode: ExtractionMode = ExtractionMode.BALANCED

    # Entity filtering
    target_entities: Set[EntityType] = field(default_factory=set)
    exclude_entities: Set[EntityType] = field(default_factory=set)
    confidence_threshold: float = 0.5

    # Context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Processing options
    enable_normalization: bool = True
    enable_linking: bool = True
    enable_relationship_extraction: bool = False
    enable_sentiment_analysis: bool = False

    # Performance settings
    max_entities_per_type: int = 50
    timeout_seconds: float = 10.0
    cache_result: bool = True

    # Language and localization
    language: str = "en"
    locale: Optional[str] = None

    # Custom settings
    custom_patterns: List[EntityPattern] = field(default_factory=list)
    custom_rules: List[EntityRule] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class ExtractionResult:
    """Comprehensive entity extraction result."""

    # Extracted entities
    entities: List[Entity]
    entity_count_by_type: Dict[str, int] = field(default_factory=dict)

    # Processing information
    processing_time: float = 0.0
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.HYBRID
    models_used: List[str] = field(default_factory=list)

    # Quality metrics
    overall_confidence: float = 0.0
    coverage_score: float = 0.0  # Percentage of text covered by entities
    consistency_score: float = 0.0

    # Analysis results
    entity_relationships: List[Dict[str, Any]] = field(default_factory=list)
    text_statistics: Dict[str, Any] = field(default_factory=dict)

    # Context and session
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"
    cache_hit: bool = False

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.entities if entity.entity_type == entity_type]

    def get_high_confidence_entities(self, threshold: float = 0.8) -> List[Entity]:
        """Get entities with high confidence scores."""
        return [entity for entity in self.entities if entity.confidence >= threshold]


class EntityExtractionError(Exception):
    """Custom exception for entity extraction operations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        extraction_strategy: Optional[ExtractionStrategy] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.extraction_strategy = extraction_strategy
        self.timestamp = datetime.now(timezone.utc)


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    async def extract(self, request: ExtractionRequest) -> List[Entity]:
        """Extract entities from text."""
        pass

    @abstractmethod
    def get_supported_entity_types(self) -> Set[EntityType]:
        """Get supported entity types."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the extractor."""
        pass

    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold."""
        return 0.5


class SpacyEntityExtractor(EntityExtractor):
    """spaCy-based entity extractor."""

    def __init__(self, model_name: str = "en_core_web_sm", logger=None):
        self.model_name = model_name
        self.logger = logger or get_logger(__name__)
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.custom_patterns = []

        # Entity type mapping
        self.spacy_to_entity_type = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.GPE,
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "QUANTITY": EntityType.QUANTITY,
            "ORDINAL": EntityType.ORDINAL,
            "CARDINAL": EntityType.CARDINAL,
            "FAC": EntityType.FACILITY,
            "LANGUAGE": EntityType.LANGUAGE,
            "NORP": EntityType.NATIONALITY,
            "PRODUCT": EntityType.PRODUCT,
            "WORK_OF_ART": EntityType.WORK_OF_ART,
            "LAW": EntityType.LAW,
            "EVENT": EntityType.EVENT,
        }

    async def initialize(self) -> None:
        """Initialize spaCy model and components."""
        try:
            import spacy

            # Load the spaCy model
            self.nlp = spacy.load(self.model_name)

            # Initialize matchers
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab)

            # Add custom patterns
            await self._setup_custom_patterns()

            self.logger.info(f"SpaCy entity extractor initialized with model: {self.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize spaCy extractor: {str(e)}")
            raise EntityExtractionError(f"SpaCy initialization failed: {str(e)}")

    async def _setup_custom_patterns(self) -> None:
        """Setup custom patterns for additional entity types."""
        # Email pattern
        email_pattern = [{"LIKE_EMAIL": True}]
        self.matcher.add("EMAIL", [email_pattern])

        # URL pattern
        url_pattern = [{"LIKE_URL": True}]
        self.matcher.add("URL", [url_pattern])

        # Phone pattern
        phone_patterns = [
            [{"SHAPE": "ddd-ddd-dddd"}],
            [{"SHAPE": "(ddd) ddd-dddd"}],
            [{"TEXT": {"REGEX": r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"}}],
        ]
        for pattern in phone_patterns:
            self.matcher.add("PHONE", [pattern])

        # Hashtag pattern
        hashtag_pattern = [{"TEXT": {"REGEX": r"#\w+"}}]
        self.matcher.add("HASHTAG", [hashtag_pattern])

        # Mention pattern
        mention_pattern = [{"TEXT": {"REGEX": r"@\w+"}}]
        self.matcher.add("MENTION", [mention_pattern])

    async def extract(self, request: ExtractionRequest) -> List[Entity]:
        """Extract entities using spaCy."""
        try:
            # Process text with spaCy
            doc = self.nlp(request.text)
            entities = []

            # Extract named entities
            for ent in doc.ents:
                entity_type = self.spacy_to_entity_type.get(ent.label_, EntityType.UNKNOWN)

                # Filter based on request criteria
                if request.target_entities and entity_type not in request.target_entities:
                    continue
                if entity_type in request.exclude_entities:
                    continue

                # Calculate confidence (spaCy doesn't provide confidence, so we estimate)
                confidence = await self._calculate_confidence(ent, doc)

                if confidence < request.confidence_threshold:
                    continue

                entity = Entity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=confidence,
                    category=self._get_entity_category(entity_type),
                    extraction_method=ExtractionStrategy.SPACY,
                    model_used=self.model_name,
                    features={
                        "label": ent.label_,
                        "lemma": ent.lemma_,
                        "pos": [token.pos_ for token in ent],
                        "dependency": [token.dep_ for token in ent],
                    },
                )

                # Add normalization if enabled
                if request.enable_normalization:
                    entity.normalized_value = await self._normalize_entity(entity, ent)

                entities.append(entity)

            # Extract custom pattern matches
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                label = self.nlp.vocab.strings[match_id]
                span = doc[start:end]

                entity_type = self._label_to_entity_type(label)
                confidence = 0.9  # High confidence for pattern matches

                if confidence >= request.confidence_threshold:
                    entity = Entity(
                        text=span.text,
                        entity_type=entity_type,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        confidence=confidence,
                        category=self._get_entity_category(entity_type),
                        extraction_method=ExtractionStrategy.SPACY,
                        model_used=self.model_name,
                        features={"pattern_match": label},
                    )
                    entities.append(entity)

            return entities

        except Exception as e:
            self.logger.error(f"spaCy extraction failed: {str(e)}")
            raise EntityExtractionError(f"spaCy extraction failed: {str(e)}")

    async def _calculate_confidence(self, ent, doc) -> float:
        """Calculate confidence score for spaCy entity."""
        # Base confidence from entity properties
        confidence = 0.7

        # Boost confidence for proper nouns
        if any(token.pos_ == "PROPN" for token in ent):
            confidence += 0.2

        # Boost confidence for entities in knowledge base
        if ent.kb_id_:
            confidence += 0.1

        # Reduce confidence for single character entities
        if len(ent.text) == 1:
            confidence -= 0.3

        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))

    async def _normalize_entity(self, entity: Entity, ent) -> Any:
        """Normalize entity value."""
        if entity.entity_type == EntityType.DATE:
            # Parse date entities
            try:
                # Use spaCy's built-in date parsing if available
                return ent._.parse_date() if hasattr(ent._, "parse_date") else ent.text
            except:
                return ent.text
        elif entity.entity_type == EntityType.MONEY:
            # Parse monetary values
            return self._parse_money(ent.text)
        elif entity.entity_type == EntityType.PERCENT:
            # Parse percentages
            return self._parse_percentage(ent.text)
        else:
            return ent.text

    def _parse_money(self, text: str) -> Dict[str, Any]:
        """Parse monetary values."""
        # Simple money parsing - could be enhanced
        amount_match = re.search(r"[\d,]+\.?\d*", text)
        currency_match = re.search(r"[A-Z]{3}|\$|€|£|¥", text)

        return {
            "amount": float(amount_match.group().replace(",", "")) if amount_match else None,
            "currency": currency_match.group() if currency_match else None,
            "original_text": text,
        }

    def _parse_percentage(self, text: str) -> Dict[str, Any]:
        """Parse percentage values."""
        value_match = re.search(r"[\d,]+\.?\d*", text)
        return {
            "value": float(value_match.group().replace(",", "")) if value_match else None,
            "original_text": text,
        }

    def _label_to_entity_type(self, label: str) -> EntityType:
        """Convert label to entity type."""
        label_mapping = {
            "EMAIL": EntityType.EMAIL,
            "URL": EntityType.URL,
            "PHONE": EntityType.PHONE,
            "HASHTAG": EntityType.HASHTAG,
            "MENTION": EntityType.MENTION,
        }
        return label_mapping.get(label, EntityType.UNKNOWN)

    def _get_entity_category(self, entity_type: EntityType) -> EntityCategory:
        """Get entity category for entity type."""
        category_mapping = {
            EntityType.PERSON: EntityCategory.NAMED,
            EntityType.ORGANIZATION: EntityCategory.NAMED,
            EntityType.LOCATION: EntityCategory.NAMED,
            EntityType.GPE: EntityCategory.NAMED,
            EntityType.FACILITY: EntityCategory.NAMED,
            EntityType.DATE: EntityCategory.TEMPORAL,
            EntityType.TIME: EntityCategory.TEMPORAL,
            EntityType.DURATION: EntityCategory.TEMPORAL,
            EntityType.MONEY: EntityCategory.NUMERIC,
            EntityType.PERCENT: EntityCategory.NUMERIC,
            EntityType.QUANTITY: EntityCategory.NUMERIC,
            EntityType.ORDINAL: EntityCategory.NUMERIC,
            EntityType.CARDINAL: EntityCategory.NUMERIC,
            EntityType.EMAIL: EntityCategory.TECHNICAL,
            EntityType.URL: EntityCategory.TECHNICAL,
            EntityType.PHONE: EntityCategory.TECHNICAL,
            EntityType.IP_ADDRESS: EntityCategory.TECHNICAL,
            EntityType.HASHTAG: EntityCategory.TECHNICAL,
            EntityType.MENTION: EntityCategory.TECHNICAL,
        }
        return category_mapping.get(entity_type, EntityCategory.CONCEPTUAL)

    def get_supported_entity_types(self) -> Set[EntityType]:
        """Get supported entity types."""
        return set(self.spacy_to_entity_type.values()) | {
            EntityType.EMAIL,
            EntityType.URL,
            EntityType.PHONE,
            EntityType.HASHTAG,
            EntityType.MENTION,
        }


class RegexEntityExtractor(EntityExtractor):
    """Regular expression-based entity extractor."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.patterns = {}
        self._setup_default_patterns()

    def _setup_default_patterns(self) -> None:
        """Setup default regex patterns."""
        self.patterns = {
            EntityType.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            EntityType.URL: re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            EntityType.PHONE: re.compile(
                r"\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
            ),
            EntityType.IP_ADDRESS: re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            EntityType.HASHTAG: re.compile(r"#\w+"),
            EntityType.MENTION: re.compile(r"@\w+"),
            EntityType.MONEY: re.compile(
                r"\$[0-9,]+\.?[0-9]*|[0-9,]+\.?[0-9]*\s*(?:USD|EUR|GBP|JPY|dollars?|euros?|pounds?)"
            ),
            EntityType.PERCENT: re.compile(r"[0-9]+\.?[0-9]*\s*%"),
            EntityType.DATE: re.compile(
                r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
            ),
            EntityType.TIME: re.compile(
                r"\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\s*(?:AM|PM|am|pm)?\b"
            ),
        }

    async def initialize(self) -> None:
        """Initialize the regex extractor."""
        self.logger.info("Regex entity extractor initialized")

    async def extract(self, request: ExtractionRequest) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, pattern in self.patterns.items():
            # Filter based on request criteria
            if request.target_entities and entity_type not in request.target_entities:
                continue
            if entity_type in request.exclude_entities:
                continue

            matches = pattern.finditer(request.text)
            for match in matches:
                confidence = 0.85  # High confidence for regex matches

                if confidence >= request.confidence_threshold:
                    entity = Entity(
                        text=match.group(),
                        entity_type=entity_type,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=confidence,
                        category=self._get_entity_category(entity_type),
                        extraction_method=ExtractionStrategy.REGEX,
                        features={"pattern": pattern.pattern},
                    )

                    # Add normalization if enabled
                    if request.enable_normalization:
                        entity.normalized_value = await self._normalize_entity(entity)

                    entities.append(entity)

        return entities

    async def _normalize_entity(self, entity: Entity) -> Any:
        """Normalize entity value."""
        if entity.entity_type == EntityType.EMAIL:
            return entity.text.lower()
        elif entity.entity_type == EntityType.URL:
            return entity.text.lower()
        elif entity.entity_type == EntityType.PHONE:
            # Normalize phone number format
            digits = re.sub(r"[^\d]", "", entity.text)
            if len(digits) == 10:
                return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
            elif len(digits) == 11 and digits[0] == "1":
                return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
            return entity.text
        elif entity.entity_type == EntityType.HASHTAG:
            return entity.text.lower()
        elif entity.entity_type == EntityType.MENTION:
            return entity.text.lower()
        else:
            return entity.text

    def _get_entity_category(self, entity_type: EntityType) -> EntityCategory:
        """Get entity category for entity type."""
        category_mapping = {
            EntityType.EMAIL: EntityCategory.TECHNICAL,
            EntityType.URL: EntityCategory.TECHNICAL,
            EntityType.PHONE: EntityCategory.TECHNICAL,
            EntityType.IP_ADDRESS: EntityCategory.TECHNICAL,
            EntityType.HASHTAG: EntityCategory.TECHNICAL,
            EntityType.MENTION: EntityCategory.TECHNICAL,
            EntityType.MONEY: EntityCategory.NUMERIC,
            EntityType.PERCENT: EntityCategory.NUMERIC,
            EntityType.DATE: EntityCategory.TEMPORAL,
            EntityType.TIME: EntityCategory.TEMPORAL,
        }
        return category_mapping.get(entity_type, EntityCategory.CONCEPTUAL)

    def get_supported_entity_types(self) -> Set[EntityType]:
        """Get supported entity types."""
        return set(self.patterns.keys())


class TransformersEntityExtractor(EntityExtractor):
    """HuggingFace transformers-based entity extractor."""

    def __init__(self, model_router: ModelRouter, logger=None):
        self.model_router = model_router
        self.logger = logger or get_logger(__name__)
        self.tokenizer = None
        self.model = None
        self.id2label = {}
        self.label2id = {}

    async def initialize(self) -> None:
        """Initialize the transformers extractor."""
        try:
            # Load default NER model
            model_config = {
                "task": "token-classification",
                "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "aggregation_strategy": "simple",
            }

            # Initialize through model router
            await self.model_router.initialize_model("ner_default", model_config)

            self.logger.info("Transformers entity extractor initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize transformers extractor: {str(e)}")
            raise EntityExtractionError(f"Transformers initialization failed: {str(e)}")

    async def extract(self, request: ExtractionRequest) -> List[Entity]:
        """Extract entities using transformers model."""
        try:
            # Prepare model request
            model_request = {
                "text": request.text,
                "task": "token-classification",
                "model_name": "ner_default",
            }

            # Get predictions from model router
            predictions = await self.model_router.process_request(model_request)

            entities = []
            for pred in predictions.get("entities", []):
                # Map label to entity type
                entity_type = self._map_label_to_entity_type(pred.get("entity_group", ""))

                # Filter based on request criteria
                if request.target_entities and entity_type not in request.target_entities:
                    continue
                if entity_type in request.exclude_entities:
                    continue

                confidence = pred.get("score", 0.0)
                if confidence < request.confidence_threshold:
                    continue

                entity = Entity(
                    text=pred.get("word", ""),
                    entity_type=entity_type,
                    start_char=pred.get("start", 0),
                    end_char=pred.get("end", 0),
                    confidence=confidence,
                    category=self._get_entity_category(entity_type),
                    extraction_method=ExtractionStrategy.TRANSFORMERS,
                    model_used="bert-large-cased-finetuned-conll03",
                    features={
                        "original_label": pred.get("entity_group", ""),
                        "entity_index": pred.get("index", 0),
                    },
                )

                entities.append(entity)

            return entities

        except Exception as e:
            self.logger.error(f"Transformers extraction failed: {str(e)}")
            raise EntityExtractionError(f"Transformers extraction failed: {str(e)}")

    def _map_label_to_entity_type(self, label: str) -> EntityType:
        """Map model label to entity type."""
        label_mapping = {
            "PER": EntityType.PERSON,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "MISC": EntityType.UNKNOWN,
            "GPE": EntityType.GPE,
        }
        return label_mapping.get(label.upper(), EntityType.UNKNOWN)

    def _get_entity_category(self, entity_type: EntityType) -> EntityCategory:
        """Get entity category for entity type."""
        category_mapping = {
            EntityType.PERSON: EntityCategory.NAMED,
            EntityType.ORGANIZATION: EntityCategory.NAMED,
            EntityType.LOCATION: EntityCategory.NAMED,
            EntityType.GPE: EntityCategory.NAMED,
        }
        return category_mapping.get(entity_type, EntityCategory.CONCEPTUAL)

    def get_supported_entity_types(self) -> Set[EntityType]:
        """Get supported entity types."""
        return {EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION, EntityType.GPE}


class HybridEntityExtractor(EntityExtractor):
    """Hybrid extractor combining multiple extraction strategies."""

    def __init__(self, extractors: List[EntityExtractor], logger=None):
        self.extractors = extractors
        self.logger = logger or get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize all extractors."""
        for extractor in self.extractors:
            await extractor.initialize()

        self.logger.info("Hybrid entity extractor initialized")

    async def extract(self, request: ExtractionRequest) -> List[Entity]:
        """Extract entities using multiple strategies and combine results."""
        all_entities = []

        # Run all extractors
        for extractor in self.extractors:
            try:
                entities = await extractor.extract(request)
                all_entities.extend(entities)
            except Exception as e:
                self.logger.warning(f"Extractor {type(extractor).__name__} failed: {str(e)}")

        # Merge and deduplicate entities
        merged_entities = await self._merge_entities(all_entities)

        # Sort by confidence
        merged_entities.sort(key=lambda x: x.confidence, reverse=True)

        return merged_entities

    async def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge overlapping entities and resolve conflicts."""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda x: x.start_char)

        merged = []
        current_entity = entities[0]

        for entity in entities[1:]:
            # Check for overlap
            if self._entities_overlap(current_entity, entity):
                # Merge entities
                current_entity = await self._resolve_entity_conflict(current_entity, entity)
            else:
                merged.append(current_entity)
                current_entity = entity

        merged.append(current_entity)
        return merged

    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap."""
        return not (
            entity1.end_char <= entity2.start_char or entity2.end_char <= entity1.start_char
        )

    async def _resolve_entity_conflict(self, entity1: Entity, entity2: Entity) -> Entity:
        """Resolve conflict between overlapping entities."""
        # Prefer entity with higher confidence
        if entity1.confidence > entity2.confidence:
            return entity1
        elif entity2.confidence > entity1.confidence:
            return entity2

        # If confidence is equal, prefer more specific entity type
        if self._is_more_specific(entity1.entity_type, entity2.entity_type):
            return entity1
        else:
            return entity2

    def _is_more_specific(self, type1: EntityType, type2: EntityType) -> bool:
        """Check if one entity type is more specific than another."""
        # Define specificity hierarchy
        specificity_order = [
            EntityType.UNKNOWN,
            EntityType.CUSTOM,
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
            EntityType.EMAIL,
            EntityType.URL,
            EntityType.PHONE,
        ]

        try:
            return specificity_order.index(type1) > specificity_order.index(type2)
        except ValueError:
            return False

    def get_supported_entity_types(self) -> Set[EntityType]:
        """Get all supported entity types from all extractors."""
        supported_types = set()
        for extractor in self.extractors:
            supported_types.update(extractor.get_supported_entity_types())
        return supported_types


class EntityLinker:
    """Links extracted entities to knowledge base entries."""

    def __init__(self, memory_manager: MemoryManager, logger=None):
        self.memory_manager = memory_manager
        self.logger = logger or get_logger(__name__)
        self.entity_cache = {}

    async def link_entities(self, entities: List[Entity]) -> List[Entity]:
        """Link entities to knowledge base entries."""
        linked_entities = []

        for entity in entities:
            linked_entity = await self._link_single_entity(entity)
            linked_entities.append(linked_entity)

        return linked_entities

    async def _link_single_entity(self, entity: Entity) -> Entity:
        """Link a single entity to knowledge base."""
        # Check cache first
        cache_key = f"{entity.entity_type.value}:{entity.text.lower()}"
        if cache_key in self.entity_cache:
            entity.knowledge_base_id = self.entity_cache[cache_key]
            return entity

        # Search in memory/knowledge base
        try:
            search_results = await self.memory_manager.search_memories(
                query=entity.text, memory_types=["semantic"], max_results=1
            )

            if search_results and search_results[0].confidence > 0.8:
                entity.knowledge_base_id = search_results[0].memory_id
                self.entity_cache[cache_key] = entity.knowledge_base_id

        except Exception as e:
            self.logger.warning(f"Entity linking failed for {entity.text}: {str(e)}")

        return entity


class RelationshipExtractor:
    """Extracts relationships between entities."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)

    async def extract_relationships(
        self, entities: List[Entity], text: str
    ) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []

        # Simple relationship extraction based on proximity and patterns
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                relationship = await self._analyze_entity_pair(entity1, entity2, text)
                if relationship:
                    relationships.append(relationship)

        return relationships

    async def _analyze_entity_pair(
        self, entity1: Entity, entity2: Entity, text: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze relationship between two entities."""
        # Calculate distance between entities
        distance = abs(entity1.start_char - entity2.start_char)

        # Skip if entities are too far apart
        if distance > 100:  # characters
            return None

        # Extract text between entities
        start = min(entity1.end_char, entity2.end_char)
        end = max(entity1.start_char, entity2.start_char)
        between_text = text[start:end].strip()

        # Simple relationship detection
        relationship_type = self._detect_relationship_type(entity1, entity2, between_text)

        if relationship_type:
            return {
                "entity1": entity1.to_dict(),
                "entity2": entity2.to_dict(),
                "relationship_type": relationship_type,
                "confidence": 0.7,  # Simple confidence
                "context": between_text,
            }

        return None

    def _detect_relationship_type(
        self, entity1: Entity, entity2: Entity, context: str
    ) -> Optional[str]:
        """Detect relationship type between entities."""
        # Simple rule-based relationship detection
        context_lower = context.lower()

        if (
            entity1.entity_type == EntityType.PERSON
            and entity2.entity_type == EntityType.ORGANIZATION
        ):
            if any(
                word in context_lower
                for word in ["works at", "employed by", "ceo of", "founder of"]
            ):
                return "works_at"

        elif (
            entity1.entity_type == EntityType.PERSON and entity2.entity_type == EntityType.LOCATION
        ):
            if any(word in context_lower for word in ["lives in", "born in", "from"]):
                return "located_in"

        elif (
            entity1.entity_type == EntityType.ORGANIZATION
            and entity2.entity_type == EntityType.LOCATION
        ):
            if any(
                word in context_lower for word in ["based in", "headquartered in", "located in"]
            ):
                return "headquarters"

        return None


class EntityNormalizer:
    """Normalizes entity values to canonical forms."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.normalization_cache = {}

    async def normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize all entities."""
        normalized_entities = []

        for entity in entities:
            normalized_entity = await self._normalize_single_entity(entity)
            normalized_entities.append(normalized_entity)

        return normalized_entities

    async def _normalize_single_entity(self, entity: Entity) -> Entity:
        """Normalize a single entity."""
        cache_key = f"{entity.entity_type.value}:{entity.text}"

        if cache_key in self.normalization_cache:
            entity.normalized_value = self.normalization_cache[cache_key]
            entity.canonical_form = self.normalization_cache[cache_key]
            return entity

        # Perform normalization based on entity type
        normalized_value = await self._normalize_by_type(entity)

        if normalized_value:
            entity.normalized_value = normalized_value
            entity.canonical_form = str(normalized_value)
            self.normalization_cache[cache_key] = normalized_value

        return entity

    async def _normalize_by_type(self, entity: Entity) -> Any:
        """Normalize entity based on its type."""
        if entity.entity_type == EntityType.PERSON:
            return self._normalize_person_name(entity.text)
        elif entity.entity_type == EntityType.EMAIL:
            return entity.text.lower().strip()
        elif entity.entity_type == EntityType.URL:
            return entity.text.lower().strip()
        elif entity.entity_type == EntityType.PHONE:
            return self._normalize_phone_number(entity.text)
        elif entity.entity_type == EntityType.DATE:
            return self._normalize_date(entity.text)
        elif entity.entity_type == EntityType.MONEY:
            return self._normalize_money(entity.text)
        else:
            return entity.text.strip()

    def _normalize_person_name(self, name: str) -> str:
        """Normalize person names."""
        # Simple name normalization
        words = name.split()
        normalized_words = [word.capitalize() for word in words]
        return " ".join(normalized_words)

    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone numbers."""
        # Extract digits only
        digits = re.sub(r"[^\d]", "", phone)

        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == "1":
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            return phone

    def _normalize_date(self, date_str: str) -> str:
        """Normalize date strings."""
        # Simple date normalization - could be enhanced with proper date parsing
        return date_str.strip()

    def _normalize_money(self, money_str: str) -> Dict[str, Any]:
        """Normalize monetary values."""
        # Extract amount and currency
        amount_match = re.search(r"[\d,]+\.?\d*", money_str)
        currency_match = re.search(r"[A-Z]{3}|\$|€|£|¥", money_str)

        return {
            "amount": float(amount_match.group().replace(",", "")) if amount_match else None,
            "currency": currency_match.group() if currency_match else "USD",
            "original_text": money_str,
        }


class EnhancedEntityExtractor:
    """
    Advanced Named Entity Extraction System for the AI assistant.

    This class provides comprehensive entity extraction capabilities including:
    - Multiple extraction strategies (spaCy, transformers, regex, hybrid)
    - Entity normalization and linking
    - Relationship extraction
    - Caching and performance optimization
    - Integration with the AI assistant's core systems
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced entity extractor.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Core components
        self.memory_manager = container.get(MemoryManager)
        self.tokenizer = container.get(EnhancedTokenizer)
        self.model_router = container.get(ModelRouter)
        self.cache_strategy = container.get(CacheStrategy)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Extractors and processors
        self.extractors: Dict[ExtractionStrategy, EntityExtractor] = {}
        self.entity_linker: Optional[EntityLinker] = None
        self.relationship_extractor: Optional[RelationshipExtractor] = None
        self.entity_normalizer: Optional[EntityNormalizer] = None

        # Configuration
        self._load_configuration()

        # Monitoring and metrics
        self._setup_monitoring()

        # Caching
        self.extraction_cache = {}
        self.cache_ttl = self.config.get("entity_extraction.cache_ttl_seconds", 3600)

        # Performance tracking
        self.extraction_stats = {
            "total_extractions": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "entity_counts_by_type": defaultdict(int),
            "strategy_usage": defaultdict(int),
        }

        # Register health check
        self.health_check.register_component("entity_extractor", self._health_check_callback)

        self.logger.info("EnhancedEntityExtractor initialized")

    def _load_configuration(self) -> None:
        """Load configuration settings."""
        self.default_strategy = ExtractionStrategy(
            self.config.get("entity_extraction.default_strategy", "hybrid")
        )
        self.default_mode = ExtractionMode(
            self.config.get("entity_extraction.default_mode", "balanced")
        )
        self.confidence_threshold = self.config.get("entity_extraction.confidence_threshold", 0.5)
        self.enable_caching = self.config.get("entity_extraction.enable_caching", True)
        self.enable_learning = self.config.get("entity_extraction.enable_learning", True)
        self.max_entities_per_request = self.config.get(
            "entity_extraction.max_entities_per_request", 100
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)

            # Register metrics
            self.metrics.register_counter("entity_extractions_total")
            self.metrics.register_counter("entity_extraction_errors_total")
            self.metrics.register_histogram("entity_extraction_duration_seconds")
            self.metrics.register_gauge("entities_extracted_total")
            self.metrics.register_counter("entity_cache_hits_total")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize the entity extractor and all components."""
        try:
            # Initialize extractors
            await self._initialize_extractors()

            # Initialize supporting components
            await self._initialize_supporting_components()

            # Register event handlers
            await self._register_event_handlers()

            # Start background tasks
            await self._start_background_tasks()

            # Emit initialization event
            await self.event_bus.emit(
                ComponentInitialized(
                    component_name="entity_extractor", initialization_time=time.time()
                )
            )

            self.logger.info("Entity extractor initialization completed")

        except Exception as e:
            self.logger.error(f"Entity extractor initialization failed: {str(e)}")
            await self.event_bus.emit(
                ComponentFailed(component_name="entity_extractor", error_message=str(e))
            )
            raise

    async def _initialize_extractors(self) -> None:
        """Initialize all entity extractors."""
        # spaCy extractor
        if self.config.get("entity_extraction.enable_spacy", True):
            spacy_model = self.config.get("entity_extraction.spacy_model", "en_core_web_sm")
            self.extractors[ExtractionStrategy.SPACY] = SpacyEntityExtractor(
                spacy_model, self.logger
            )
            await self.extractors[ExtractionStrategy.SPACY].initialize()

        # Regex extractor
        if self.config.get("entity_extraction.enable_regex", True):
            self.extractors[ExtractionStrategy.REGEX] = RegexEntityExtractor(self.logger)
            await self.extractors[ExtractionStrategy.REGEX].initialize()

        # Transformers extractor
        if self.config.get("entity_extraction.enable_transformers", True):
            self.extractors[ExtractionStrategy.TRANSFORMERS] = TransformersEntityExtractor(
                self.model_router, self.logger
            )
            await self.extractors[ExtractionStrategy.TRANSFORMERS].initialize()

        # Hybrid extractor
        available_extractors = list(self.extractors.values())
        if len(available_extractors) > 1:
            self.extractors[ExtractionStrategy.HYBRID] = HybridEntityExtractor(
                available_extractors, self.logger
            )
            await self.extractors[ExtractionStrategy.HYBRID].initialize()

    async def _initialize_supporting_components(self) -> None:
        """Initialize supporting components."""
        # Entity linker
        if self.config.get("entity_extraction.enable_linking", True):
            self.entity_linker = EntityLinker(self.memory_manager, self.logger)

        # Relationship extractor
        if self.config.get("entity_extraction.enable_relationships", True):
            self.relationship_extractor = RelationshipExtractor(self.logger)

        # Entity normalizer
        if self.config.get("entity_extraction.enable_normalization", True):
            self.entity_normalizer = EntityNormalizer(self.logger)

    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        if self.enable_learning:
            self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
            self.event_bus.subscribe("session_ended", self._handle_session_ended)

    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        # Cache cleanup task
        asyncio.create_task(self._cache_cleanup_loop())

        # Performance monitoring task
        asyncio.create_task(self._performance_monitoring_loop())

    @handle_exceptions
    async def extract_entities(self, request: ExtractionRequest) -> ExtractionResult:
        """
        Extract entities from text using the specified strategy.

        Args:
            request: Entity extraction request configuration

        Returns:
            ExtractionResult containing extracted entities and metadata
        """
        start_time = time.time()

        # Check cache first
        if self.enable_caching and request.cache_result:
            cache_key = self._generate_cache_key(request)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.extraction_stats["cache_hits"] += 1
                if self.metrics:
                    self.metrics.increment("entity_cache_hits_total")
                return cached_result

        try:
            with self.tracer.trace("entity_extraction") if self.tracer else None:
                # Validate request
                self._validate_request(request)

                # Select extraction strategy
                strategy = (
                    request.strategy
                    if request.strategy != ExtractionStrategy.HYBRID
                    else self.default_strategy
                )
                if strategy not in self.extractors:
                    strategy = (
                        ExtractionStrategy.HYBRID
                        if ExtractionStrategy.HYBRID in self.extractors
                        else list(self.extractors.keys())[0]
                    )

                # Extract entities
                extractor = self.extractors[strategy]
                entities = await extractor.extract(request)

                # Apply post-processing
                entities = await self._post_process_entities(entities, request)

                # Create result
                result = await self._create_extraction_result(
                    entities, request, start_time, strategy
                )

                # Cache result
                if self.enable_caching and request.cache_result:
                    await self._cache_result(cache_key, result)

                # Update metrics
                self._update_metrics(result, time.time() - start_time, strategy)

                # Store for learning
                if self.enable_learning:
                    await self._store_extraction_for_learning(request, result)

                # Emit completion event
                await self.event_bus.emit(
                    ProcessingCompleted(
                        component_name="entity_extractor",
                        processing_time=time.time() - start_time,
                        result_count=len(entities),
                    )
                )

                return result

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")

            # Emit failure event
            await self.event_bus.emit(
                ProcessingFailed(component_name="entity_extractor", error_message=str(e))
            )

            if self.metrics:
                self.metrics.increment("entity_extraction_errors_total")

            # Return empty result with error
            return ExtractionResult(
                entities=[],
                processing_time=time.time() - start_time,
                extraction_strategy=request.strategy,
                errors=[str(e)],
                session_id=request.session_id,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
            )

    def _validate_request(self, request: ExtractionRequest) -> None:
        """Validate extraction request."""
        if not request.text or not request.text.strip():
            raise EntityExtractionError("Empty text provided for extraction")

        if len(request.text) > 50000:  # Reasonable limit
            raise EntityExtractionError("Text too long for extraction")

        if request.confidence_threshold < 0.0 or request.confidence_threshold > 1.0:
            raise EntityExtractionError("Invalid confidence threshold")

    async def _post_process_entities(
        self, entities: List[Entity], request: ExtractionRequest
    ) -> List[Entity]:
        """Apply post-processing to extracted entities."""
        if not entities:
            return entities

        # Remove duplicates and overlaps
        entities = await self._deduplicate_entities(entities)

        # Apply normalization
        if request.enable_normalization and self.entity_normalizer:
            entities = await self.entity_normalizer.normalize_entities(entities)

        # Apply entity linking
        if request.enable_linking and self.entity_linker:
            entities = await self.entity_linker.link_entities(entities)

        # Limit number of entities
        if len(entities) > self.max_entities_per_request:
            entities = entities[: self.max_entities_per_request]

        return entities

    async def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and overlapping entities."""
        if not entities:
            return entities

        # Sort by start position and confidence
        entities.sort(key=lambda x: (x.start_char, -x.confidence))

        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    break

            if not overlaps:
                deduplicated.append(entity)

        return deduplicated

    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap."""
        return not (
            entity1.end_char <= entity2.start_char or entity2.end_char <= entity1.start_char
        )

    async def _create_extraction_result(
        self,
        entities: List[Entity],
        request: ExtractionRequest,
        start_time: float,
        strategy: ExtractionStrategy,
    ) -> ExtractionResult:
        """Create extraction result with comprehensive metadata."""
        processing_time = time.time() - start_time

        # Calculate entity counts by type
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity.entity_type.value] += 1

        # Calculate overall confidence
        overall_confidence = (
            sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        )

        # Calculate coverage score
        total_chars = sum(e.end_char - e.start_char for e in entities)
        coverage_score = total_chars / len(request.text) if request.text else 0.0

        # Extract relationships if enabled
        relationships = []
        if request.enable_relationship_extraction and self.relationship_extractor:
            relationships = await self.relationship_extractor.extract_relationships(
                entities, request.text
            )

        # Calculate text statistics
        text_stats = {
            "character_count": len(request.text),
            "word_count": len(request.text.split()),
            "sentence_count": len(re.split(r"[.!?]+", request.text)),
            "entity_density": (
                len(entities) / len(request.text.split()) if request.text.split() else 0.0
            ),
        }

        return ExtractionResult(
            entities=entities,
            entity_count_by_type=dict(entity_counts),
            processing_time=processing_time,
            extraction_strategy=strategy,
            models_used=[getattr(self.extractors.get(strategy), "model_name", strategy.value)],
            overall_confidence=overall_confidence,
            coverage_score=coverage_score,
            consistency_score=self._calculate_consistency_score(entities),
            entity_relationships=relationships,
            text_statistics=text_stats,
            session_id=request.session_id,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            language=request.language,
        )

    def _calculate_consistency_score(self, entities: List[Entity]) -> float:
        """Calculate consistency score for extracted entities."""
        if len(entities) < 2:
            return 1.0

        # Simple consistency score based on confidence variance
        confidences = [e.confidence for e in entities]
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)

        # Convert variance to consistency score (lower variance = higher consistency)
        consistency_score = max(0.0, 1.0 - variance)
        return consistency_score

    def _generate_cache_key(self, request: ExtractionRequest) -> str:
        """Generate cache key for request."""
        key_components = [
            request.text,
            request.strategy.value,
            request.mode.value,
            str(sorted(request.target_entities)),
            str(request.confidence_threshold),
        ]
