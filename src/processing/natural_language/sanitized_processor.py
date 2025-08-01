"""
Sanitized Natural Language Processor
Author: Drmusab
Last Modified: 2025-07-19 21:43:58 UTC

This module provides input sanitization for all natural language processing operations.
"""

import html
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import asyncio

from src.core.dependency_injection import Container
from src.core.security.sanitization import (
    ContentType,
    SanitizationContext,
    SanitizationResult,
    SanitizationType,
    SecurityLevel,
    SecuritySanitizer,
)
from src.observability.logging.config import get_logger
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer


class SanitizedNLPProcessor:
    """
    Natural language processor with comprehensive input sanitization.
    """

    def __init__(self, container: Container):
        """
        Initialize the sanitized NLP processor.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core components
        self.sanitizer = container.get(SecuritySanitizer)
        self.intent_manager = container.get(IntentManager)
        self.entity_extractor = container.get(EntityExtractor)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)

        # Sanitization patterns
        self.malicious_patterns = {
            "script_injection": re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            "sql_injection": re.compile(
                r"\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b",
                re.IGNORECASE,
            ),
            "command_injection": re.compile(r"[;&|`$]|\$\(|\$\{|<\(|>\("),
            "path_traversal": re.compile(r"\.\.[/\\]"),
            "xss_attempt": re.compile(
                r"(javascript:|onerror=|onload=|onclick=|onmouseover=)", re.IGNORECASE
            ),
            "xxe_injection": re.compile(r"<!ENTITY|SYSTEM|PUBLIC|DOCTYPE", re.IGNORECASE),
        }

        # Safe patterns for preservation
        self.safe_patterns = {
            "url": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "mention": re.compile(r"@[A-Za-z0-9_]+"),
            "hashtag": re.compile(r"#[A-Za-z0-9_]+"),
        }

        # Maximum lengths
        self.max_input_length = 10000
        self.max_word_length = 100
        self.max_entity_length = 200

        self.logger.info("Sanitized NLP processor initialized")

    async def process_text(
        self,
        text: str,
        session_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process text with comprehensive sanitization.

        Args:
            text: Input text to process
            session_id: Session identifier
            user_id: Optional user identifier
            context: Optional processing context

        Returns:
            Processed and sanitized results
        """
        try:
            # Create sanitization context
            sanitization_context = SanitizationContext(
                user_id=user_id,
                session_id=session_id,
                component="nlp_processor",
                content_type=ContentType.TEXT,
                security_level=SecurityLevel.HIGH,
                strict_mode=True,
                preserve_formatting=False,
                max_length=self.max_input_length,
            )

            # Sanitize input
            sanitization_result = await self.sanitizer.sanitize_input(
                content=text, content_type=ContentType.TEXT, context=sanitization_context
            )

            if not sanitization_result.success:
                self.logger.warning(
                    f"Input sanitization failed: {sanitization_result.violations_detected}"
                )
                return {
                    "success": False,
                    "error": "Input validation failed",
                    "violations": sanitization_result.violations_detected,
                }

            # Use sanitized content
            sanitized_text = sanitization_result.sanitized_content

            # Additional NLP-specific sanitization
            sanitized_text = await self._nlp_specific_sanitization(sanitized_text, context)

            # Process with intent detection
            intent_result = await self._safe_intent_detection(sanitized_text, session_id)

            # Extract entities with sanitization
            entity_result = await self._safe_entity_extraction(sanitized_text, session_id)

            # Analyze sentiment safely
            sentiment_result = await self._safe_sentiment_analysis(sanitized_text, session_id)

            # Combine results
            return {
                "success": True,
                "original_length": len(text),
                "sanitized_length": len(sanitized_text),
                "content_modified": text != sanitized_text,
                "intent": intent_result,
                "entities": entity_result,
                "sentiment": sentiment_result,
                "security_flags": sanitization_result.security_flags,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error in sanitized NLP processing: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _nlp_specific_sanitization(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """
        Apply NLP-specific sanitization rules.

        Args:
            text: Text to sanitize
            context: Processing context

        Returns:
            Sanitized text
        """
        # Normalize Unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # Check for malicious patterns
        for pattern_name, pattern in self.malicious_patterns.items():
            if pattern.search(text):
                self.logger.warning(f"Detected {pattern_name} pattern in input")
                text = pattern.sub("", text)

        # Preserve safe patterns
        preserved_content = {}
        for pattern_name, pattern in self.safe_patterns.items():
            matches = pattern.findall(text)
            if matches:
                preserved_content[pattern_name] = matches

        # HTML entity decoding
        text = html.unescape(text)

        # Limit consecutive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove control characters except newlines and tabs
        text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

        # Validate word lengths
        words = text.split()
        sanitized_words = []
        for word in words:
            if len(word) > self.max_word_length:
                self.logger.warning(f"Truncating word exceeding max length: {len(word)}")
                word = word[: self.max_word_length]
            sanitized_words.append(word)

        text = " ".join(sanitized_words)

        # Enforce maximum length
        if len(text) > self.max_input_length:
            text = text[: self.max_input_length]

        return text.strip()

    async def _safe_intent_detection(self, text: str, session_id: str) -> Dict[str, Any]:
        """
        Perform intent detection with safety checks.

        Args:
            text: Sanitized text
            session_id: Session identifier

        Returns:
            Intent detection results
        """
        try:
            # Additional safety check for intent detection
            if len(text) < 2:
                return {"intent": "unknown", "confidence": 0.0, "error": "Text too short"}

            # Detect intent
            intent_result = await self.intent_manager.detect_intent(
                text, context={"session_id": session_id}
            )

            # Sanitize intent name
            if isinstance(intent_result, str):
                intent_result = re.sub(r"[^a-zA-Z0-9_-]", "", intent_result)

            return {
                "intent": intent_result,
                "confidence": 0.85,  # Placeholder confidence
                "sanitized": True,
            }

        except Exception as e:
            self.logger.error(f"Error in safe intent detection: {str(e)}")
            return {"intent": "error", "confidence": 0.0, "error": str(e)}

    async def _safe_entity_extraction(self, text: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Perform entity extraction with safety checks.

        Args:
            text: Sanitized text
            session_id: Session identifier

        Returns:
            Extracted entities
        """
        try:
            # Extract entities
            entities = await self.entity_extractor.extract_entities(text)

            # Sanitize each entity
            sanitized_entities = []
            for entity in entities:
                # Ensure entity text is safe
                entity_text = entity.get("text", "")
                if len(entity_text) > self.max_entity_length:
                    entity_text = entity_text[: self.max_entity_length]

                # Remove any remaining unsafe characters
                entity_text = re.sub(r'[<>"\'`]', "", entity_text)

                sanitized_entity = {
                    "text": entity_text,
                    "type": re.sub(r"[^a-zA-Z0-9_-]", "", entity.get("type", "unknown")),
                    "start": max(0, entity.get("start", 0)),
                    "end": min(len(text), entity.get("end", 0)),
                    "confidence": max(0.0, min(1.0, entity.get("confidence", 0.0))),
                }

                sanitized_entities.append(sanitized_entity)

            return sanitized_entities

        except Exception as e:
            self.logger.error(f"Error in safe entity extraction: {str(e)}")
            return []

    async def _safe_sentiment_analysis(self, text: str, session_id: str) -> Dict[str, Any]:
        """
        Perform sentiment analysis with safety checks.

        Args:
            text: Sanitized text
            session_id: Session identifier

        Returns:
            Sentiment analysis results
        """
        try:
            # Analyze sentiment
            sentiment_result = await self.sentiment_analyzer.analyze(text)

            # Ensure safe output format
            return {
                "sentiment": re.sub(
                    r"[^a-zA-Z0-9_-]", "", str(sentiment_result.get("sentiment", "neutral"))
                ),
                "confidence": max(0.0, min(1.0, float(sentiment_result.get("confidence", 0.0)))),
                "scores": {
                    "positive": max(0.0, min(1.0, float(sentiment_result.get("positive", 0.0)))),
                    "negative": max(0.0, min(1.0, float(sentiment_result.get("negative", 0.0)))),
                    "neutral": max(0.0, min(1.0, float(sentiment_result.get("neutral", 0.0)))),
                },
            }

        except Exception as e:
            self.logger.error(f"Error in safe sentiment analysis: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}

    async def validate_batch_input(
        self, texts: List[str], session_id: str, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and sanitize batch text input.

        Args:
            texts: List of texts to validate
            session_id: Session identifier
            user_id: Optional user identifier

        Returns:
            Validation results
        """
        results = []
        total_violations = 0

        for idx, text in enumerate(texts):
            # Create sanitization context
            context = SanitizationContext(
                user_id=user_id,
                session_id=session_id,
                component=f"batch_nlp_{idx}",
                content_type=ContentType.TEXT,
                security_level=SecurityLevel.HIGH,
            )

            # Sanitize
            result = await self.sanitizer.sanitize_input(
                content=text, content_type=ContentType.TEXT, context=context
            )

            results.append(
                {
                    "index": idx,
                    "success": result.success,
                    "violations": len(result.violations_detected),
                    "content_modified": result.content_modified,
                }
            )

            total_violations += len(result.violations_detected)

        return {
            "total_texts": len(texts),
            "successful": sum(1 for r in results if r["success"]),
            "total_violations": total_violations,
            "results": results,
        }
