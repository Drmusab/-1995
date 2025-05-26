"""
Advanced Intent Management System
Author: Drmusab
Last Modified: 2025-05-26 21:45:00 UTC

This module provides comprehensive intent detection, classification, and management
for the AI assistant, supporting hierarchical intents, context-aware routing,
confidence scoring, and dynamic intent learning.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Union, Tuple, AsyncGenerator
import asyncio
import threading
import time
import re
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import numpy as np
import torch
from abc import ABC, abstractmethod
import uuid
from collections import defaultdict, deque
import weakref

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    IntentDetected, IntentClassified, IntentConfidenceChanged,
    IntentLearned, IntentCategoryUpdated, IntentRoutingCompleted,
    UserBehaviorAnalyzed, ContextUpdated
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# NLP and ML components
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.storage.vector_store import VectorStore
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Skills management
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_factory import SkillFactory

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Session and workflow
from src.assistant.session_manager import SessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator


class IntentType(Enum):
    """Types of intents supported by the system."""
    ACTION = "action"                    # Direct action requests
    QUESTION = "question"                # Information seeking
    TASK = "task"                       # Complex task requests
    CONVERSATION = "conversation"        # Social/conversational
    SKILL_INVOCATION = "skill_invocation"  # Specific skill calls
    WORKFLOW = "workflow"               # Workflow execution
    SYSTEM = "system"                   # System commands
    CLARIFICATION = "clarification"     # Seeking clarification
    FEEDBACK = "feedback"               # User feedback
    PREFERENCE = "preference"           # Preference setting
    EMERGENCY = "emergency"             # Emergency situations
    CREATIVE = "creative"               # Creative requests
    ANALYTICAL = "analytical"           # Analysis requests
    EDUCATIONAL = "educational"         # Learning-related


class IntentCategory(Enum):
    """High-level intent categories."""
    PRODUCTIVITY = "productivity"
    ENTERTAINMENT = "entertainment"
    INFORMATION = "information"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    LEARNING = "learning"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    SYSTEM_CONTROL = "system_control"
    PERSONAL_ASSISTANCE = "personal_assistance"


class IntentScope(Enum):
    """Scope of intent execution."""
    SESSION = "session"                 # Current session only
    USER = "user"                      # User-specific
    GLOBAL = "global"                  # System-wide
    CONTEXTUAL = "contextual"          # Context-dependent
    TEMPORAL = "temporal"              # Time-bound


class ConfidenceLevel(Enum):
    """Intent confidence levels."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0


@dataclass
class IntentSlot:
    """Represents a slot (entity) within an intent."""
    name: str
    value: Any
    entity_type: str
    confidence: float = 0.0
    start_pos: int = -1
    end_pos: int = -1
    normalized_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentPattern:
    """Pattern definition for intent matching."""
    pattern_id: str
    intent_name: str
    patterns: List[str]  # Regex or text patterns
    weight: float = 1.0
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentDefinition:
    """Complete intent definition."""
    intent_name: str
    intent_type: IntentType
    category: IntentCategory
    scope: IntentScope
    
    # Description and documentation
    description: str
    examples: List[str] = field(default_factory=list)
    
    # Patterns and matching
    patterns: List[IntentPattern] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    
    # Slots and entities
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    slot_definitions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Execution configuration
    skill_name: Optional[str] = None
    workflow_id: Optional[str] = None
    component_name: Optional[str] = None
    
    # Context and conditions
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Priority and routing
    priority: int = 5  # 1-10 scale
    timeout_seconds: float = 30.0
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    tags: Set[str] = field(default_factory=set)
    is_active: bool = True


@dataclass
class IntentDetectionResult:
    """Result of intent detection process."""
    intent_name: str
    intent_type: IntentType
    category: IntentCategory
    confidence: float
    confidence_level: ConfidenceLevel
    
    # Detected slots and entities
    slots: List[IntentSlot] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context and metadata
    context_match: bool = True
    pattern_match: Optional[str] = None
    matched_keywords: Set[str] = field(default_factory=set)
    
    # Processing information
    processing_time: float = 0.0
    model_used: Optional[str] = None
    fallback_used: bool = False
    
    # Alternative suggestions
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    # Execution routing
    skill_name: Optional[str] = None
    workflow_id: Optional[str] = None
    component_name: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class IntentError(Exception):
    """Custom exception for intent management operations."""
    
    def __init__(self, message: str, intent_name: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.intent_name = intent_name
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class IntentDetector(ABC):
    """Abstract base class for intent detectors."""
    
    @abstractmethod
    async def detect_intent(self, text: str, context: Dict[str, Any]) -> List[IntentDetectionResult]:
        """Detect intents from text input."""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold."""
        pass
    
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train the detector with new data."""
        pass


class PatternBasedDetector(IntentDetector):
    """Pattern-based intent detector using regex and keyword matching."""
    
    def __init__(self, intent_definitions: Dict[str, IntentDefinition]):
        self.intent_definitions = intent_definitions
        self.confidence_threshold = 0.3
        self.logger = get_logger(__name__)
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns: Dict[str, List[Tuple[re.Pattern, IntentPattern]]] = {}
        
        for intent_def in self.intent_definitions.values():
            patterns = []
            for pattern in intent_def.patterns:
                for pattern_text in pattern.patterns:
                    try:
                        compiled = re.compile(pattern_text, re.IGNORECASE)
                        patterns.append((compiled, pattern))
                    except re.error as e:
                        self.logger.warning(f"Invalid regex pattern {pattern_text}: {e}")
            
            self.compiled_patterns[intent_def.intent_name] = patterns
    
    async def detect_intent(self, text: str, context: Dict[str, Any]) -> List[IntentDetectionResult]:
        """Detect intents using pattern matching."""
        results = []
        
        for intent_name, intent_def in self.intent_definitions.items():
            if not intent_def.is_active:
                continue
            
            confidence = await self._calculate_pattern_confidence(text, intent_def, context)
            
            if confidence >= self.confidence_threshold:
                slots = await self._extract_slots(text, intent_def)
                
                result = IntentDetectionResult(
                    intent_name=intent_name,
                    intent_type=intent_def.intent_type,
                    category=intent_def.category,
                    confidence=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    slots=slots,
                    skill_name=intent_def.skill_name,
                    workflow_id=intent_def.workflow_id,
                    component_name=intent_def.component_name
                )
                
                results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    async def _calculate_pattern_confidence(self, text: str, intent_def: IntentDefinition, 
                                          context: Dict[str, Any]) -> float:
        """Calculate confidence score for pattern matching."""
        scores = []
        
        # Pattern matching score
        pattern_score = 0.0
        for compiled_pattern, pattern in self.compiled_patterns.get(intent_def.intent_name, []):
            if compiled_pattern.search(text):
                pattern_score = max(pattern_score, pattern.weight)
        
        if pattern_score > 0:
            scores.append(pattern_score * 0.6)  # 60% weight for pattern match
        
        # Keyword matching score
        keyword_score = 0.0
        text_lower = text.lower()
        matched_keywords = 0
        
        for keyword in intent_def.keywords:
            if keyword.lower() in text_lower:
                matched_keywords += 1
        
        if intent_def.keywords:
            keyword_score = matched_keywords / len(intent_def.keywords)
            scores.append(keyword_score * 0.3)  # 30% weight for keywords
        
        # Context matching score
        context_score = await self._calculate_context_score(intent_def, context)
        if context_score > 0:
            scores.append(context_score * 0.1)  # 10% weight for context
        
        return sum(scores) if scores else 0.0
    
    async def _calculate_context_score(self, intent_def: IntentDefinition, 
                                     context: Dict[str, Any]) -> float:
        """Calculate context matching score."""
        if not intent_def.context_requirements:
            return 1.0  # No requirements means perfect match
        
        matches = 0
        total = len(intent_def.context_requirements)
        
        for key, expected_value in intent_def.context_requirements.items():
            if key in context and context[key] == expected_value:
                matches += 1
        
        return matches / total if total > 0 else 1.0
    
    async def _extract_slots(self, text: str, intent_def: IntentDefinition) -> List[IntentSlot]:
        """Extract slots from text using patterns."""
        slots = []
        
        for slot_name, slot_def in intent_def.slot_definitions.items():
            pattern = slot_def.get('pattern')
            if not pattern:
                continue
            
            try:
                regex = re.compile(pattern, re.IGNORECASE)
                matches = regex.finditer(text)
                
                for match in matches:
                    slot = IntentSlot(
                        name=slot_name,
                        value=match.group(),
                        entity_type=slot_def.get('type', 'string'),
                        confidence=0.8,  # Pattern-based extraction confidence
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    slots.append(slot)
                    
            except re.error as e:
                self.logger.warning(f"Invalid slot pattern for {slot_name}: {e}")
        
        return slots
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold."""
        return self.confidence_threshold
    
    async def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train pattern-based detector (update patterns from examples)."""
        # For pattern-based detector, training involves updating patterns
        # This is a simplified implementation
        pass


class MLBasedDetector(IntentDetector):
    """Machine learning-based intent detector using neural networks."""
    
    def __init__(self, model_router: ModelRouter, cache_strategy: CacheStrategy):
        self.model_router = model_router
        self.cache_strategy = cache_strategy
        self.confidence_threshold = 0.5
        self.logger = get_logger(__name__)
        self.model_name = "intent_classifier"
    
    async def detect_intent(self, text: str, context: Dict[str, Any]) -> List[IntentDetectionResult]:
        """Detect intents using machine learning models."""
        # Check cache first
        cache_key = f"intent_detection_{hashlib.md5(text.encode()).hexdigest()}"
        cached_result = await self.cache_strategy.get(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            # Prepare input for the model
            model_input = {
                "text": text,
                "context": context,
                "task": "intent_classification"
            }
            
            # Get prediction from model
            prediction = await self.model_router.predict(self.model_name, model_input)
            
            # Parse results
            results = await self._parse_ml_results(prediction, text, context)
            
            # Cache results
            await self.cache_strategy.set(cache_key, results, ttl=300)  # 5 minutes
            
            return results
            
        except Exception as e:
            self.logger.error(f"ML-based intent detection failed: {str(e)}")
            return []
    
    async def _parse_ml_results(self, prediction: Dict[str, Any], text: str, 
                              context: Dict[str, Any]) -> List[IntentDetectionResult]:
        """Parse ML model prediction results."""
        results = []
        
        predictions = prediction.get('intents', [])
        for pred in predictions:
            if pred.get('confidence', 0.0) >= self.confidence_threshold:
                result = IntentDetectionResult(
                    intent_name=pred['intent_name'],
                    intent_type=IntentType(pred.get('type', 'action')),
                    category=IntentCategory(pred.get('category', 'productivity')),
                    confidence=pred['confidence'],
                    confidence_level=self._get_confidence_level(pred['confidence']),
                    entities=pred.get('entities', []),
                    model_used=self.model_name
                )
                results.append(result)
        
        return results
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold."""
        return self.confidence_threshold
    
    async def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train ML model with new data."""
        try:
            await self.model_router.train(self.model_name, training_data)
            self.logger.info(f"Successfully trained ML intent detector with {len(training_data)} samples")
        except Exception as e:
            self.logger.error(f"Failed to train ML intent detector: {str(e)}")


class HybridDetector(IntentDetector):
    """Hybrid detector combining pattern-based and ML-based detection."""
    
    def __init__(self, pattern_detector: PatternBasedDetector, ml_detector: MLBasedDetector):
        self.pattern_detector = pattern_detector
        self.ml_detector = ml_detector
        self.confidence_threshold = 0.4
        self.logger = get_logger(__name__)
    
    async def detect_intent(self, text: str, context: Dict[str, Any]) -> List[IntentDetectionResult]:
        """Detect intents using hybrid approach."""
        # Get results from both detectors
        pattern_results = await self.pattern_detector.detect_intent(text, context)
        ml_results = await self.ml_detector.detect_intent(text, context)
        
        # Combine and rank results
        combined_results = await self._combine_results(pattern_results, ml_results)
        
        # Filter by confidence threshold
        filtered_results = [r for r in combined_results if r.confidence >= self.confidence_threshold]
        
        # Sort by confidence
        filtered_results.sort(key=lambda x: x.confidence, reverse=True)
        
        return filtered_results
    
    async def _combine_results(self, pattern_results: List[IntentDetectionResult], 
                             ml_results: List[IntentDetectionResult]) -> List[IntentDetectionResult]:
        """Combine results from different detectors."""
        combined = {}
        
        # Add pattern results
        for result in pattern_results:
            combined[result.intent_name] = result
        
        # Merge ML results
        for result in ml_results:
            if result.intent_name in combined:
                # Combine confidences (weighted average)
                existing = combined[result.intent_name]
                existing.confidence = (existing.confidence * 0.6 + result.confidence * 0.4)
                existing.confidence_level = self._get_confidence_level(existing.confidence)
                
                # Merge entities and slots
                existing.entities.extend(result.entities)
            else:
                combined[result.intent_name] = result
        
        return list(combined.values())
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level."""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def get_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold."""
        return self.confidence_threshold
    
    async def train(self, training_data: List[Dict[str, Any]]) -> None:
        """Train both detectors."""
        await asyncio.gather(
            self.pattern_detector.train(training_data),
            self.ml_detector.train(training_data)
        )


class IntentRouter:
    """Routes intents to appropriate execution targets."""
    
    def __init__(self, skill_registry: SkillRegistry, workflow_orchestrator: WorkflowOrchestrator):
        self.skill_registry = skill_registry
        self.workflow_orchestrator = workflow_orchestrator
        self.logger = get_logger(__name__)
    
    async def route_intent(self, intent_result: IntentDetectionResult, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Route intent to appropriate execution target."""
        try:
            if intent_result.skill_name:
                return await self._route_to_skill(intent_result, context)
            elif intent_result.workflow_id:
                return await self._route_to_workflow(intent_result, context)
            elif intent_result.component_name:
                return await self._route_to_component(intent_result, context)
            else:
                return await self._route_generic(intent_result, context)
                
        except Exception as e:
            self.logger.error(f"Intent routing failed for {intent_result.intent_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "intent": intent_result.intent_name
            }
    
    async def _route_to_skill(self, intent_result: IntentDetectionResult, 
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Route intent to a specific skill."""
        skill = self.skill_registry.get_skill(intent_result.skill_name)
        if not skill:
            raise IntentError(f"Skill {intent_result.skill_name} not found")
        
        # Prepare skill execution parameters
        skill_params = {}
        for slot in intent_result.slots:
            skill_params[slot.name] = slot.value
        
        # Execute skill
        result = await skill.execute(skill_params, context)
        
        return {
            "success": True,
            "result": result,
            "execution_target": "skill",
            "target_name": intent_result.skill_name
        }
    
    async def _route_to_workflow(self, intent_result: IntentDetectionResult, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Route intent to a workflow."""
        # Prepare workflow input
        workflow_input = {}
        for slot in intent_result.slots:
            workflow_input[slot.name] = slot.value
        
        # Execute workflow
        execution_id = await self.workflow_orchestrator.execute_workflow(
            intent_result.workflow_id,
            workflow_input,
            context.get('session_id'),
            context.get('user_id')
        )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "execution_target": "workflow",
            "target_name": intent_result.workflow_id
        }
    
    async def _route_to_component(self, intent_result: IntentDetectionResult, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Route intent to a system component."""
        # This would integrate with the component manager
        return {
            "success": True,
            "execution_target": "component",
            "target_name": intent_result.component_name,
            "message": "Component routing not implemented"
        }
    
    async def _route_generic(self, intent_result: IntentDetectionResult, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic intent routing."""
        # Default handling for intents without specific routing
        return {
            "success": True,
            "execution_target": "generic",
            "intent_type": intent_result.intent_type.value,
            "category": intent_result.category.value,
            "message": "Generic intent handling"
        }


class IntentLearner:
    """Learns and adapts intent recognition from user interactions."""
    
    def __init__(self, feedback_processor: FeedbackProcessor, 
                 preference_learner: PreferenceLearner):
        self.feedback_processor = feedback_processor
        self.preference_learner = preference_learner
        self.logger = get_logger(__name__)
        self.learning_buffer: deque = deque(maxlen=1000)
    
    async def learn_from_interaction(self, text: str, detected_intent: IntentDetectionResult,
                                   actual_intent: Optional[str], context: Dict[str, Any]) -> None:
        """Learn from user interaction feedback."""
        interaction_data = {
            "text": text,
            "detected_intent": detected_intent.intent_name,
            "detected_confidence": detected_intent.confidence,
            "actual_intent": actual_intent,
            "context": context,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.learning_buffer.append(interaction_data)
        
        # Process feedback if actual intent differs from detected
        if actual_intent and actual_intent != detected_intent.intent_name:
            await self._process_correction(interaction_data)
        
        # Update user preferences
        if context.get('user_id'):
            await self.preference_learner.update_intent_preferences(
                context['user_id'], detected_intent, context
            )
    
    async def _process_correction(self, interaction_data: Dict[str, Any]) -> None:
        """Process intent correction feedback."""
        correction_feedback = {
            "type": "intent_correction",
            "original_intent": interaction_data["detected_intent"],
            "correct_intent": interaction_data["actual_intent"],
            "text": interaction_data["text"],
            "context": interaction_data["context"],
            "timestamp": interaction_data["timestamp"]
        }
        
        await self.feedback_processor.process_feedback(correction_feedback)
        self.logger.info(f"Processed intent correction: {interaction_data['detected_intent']} -> {interaction_data['actual_intent']}")
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data."""
        if not self.learning_buffer:
            return {"message": "No learning data available"}
        
        total_interactions = len(self.learning_buffer)
        corrections = sum(1 for item in self.learning_buffer 
                         if item["actual_intent"] and item["actual_intent"] != item["detected_intent"])
        
        accuracy = (total_interactions - corrections) / total_interactions if total_interactions > 0 else 0
        
        return {
            "total_interactions": total_interactions,
            "corrections": corrections,
            "accuracy": accuracy,
            "learning_buffer_size": len(self.learning_buffer)
        }


class EnhancedIntentManager:
    """
    Advanced Intent Management System for the AI Assistant.
    
    Features:
    - Multi-modal intent detection (pattern-based, ML-based, hybrid)
    - Hierarchical intent classification with categories and types
    - Context-aware intent routing and execution
    - Dynamic intent learning and adaptation
    - Confidence scoring and uncertainty handling
    - Integration with skills, workflows, and components
    - Performance monitoring and analytics
    - Intent caching and optimization
    - User preference learning
    - Comprehensive event emission for system integration
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced intent manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Core dependencies
        self.model_router = container.get(ModelRouter)
        self.cache_strategy = container.get(CacheStrategy)
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.vector_store = container.get(VectorStore)
        
        # Skills and workflow integration
        self.skill_registry = container.get(SkillRegistry)
        self.skill_factory = container.get(SkillFactory)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        
        # Learning components
        self.feedback_processor = container.get(FeedbackProcessor)
        self.preference_learner = container.get(PreferenceLearner)
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Intent management
        self.intent_definitions: Dict[str, IntentDefinition] = {}
        self.intent_detectors: Dict[str, IntentDetector] = {}
        self.intent_router: Optional[IntentRouter] = None
        self.intent_learner: Optional[IntentLearner] = None
        
        # Configuration
        self._detection_mode = self.config.get("intent_manager.detection_mode", "hybrid")
        self._enable_learning = self.config.get("intent_manager.enable_learning", True)
        self._cache_ttl = self.config.get("intent_manager.cache_ttl", 300)
        self._confidence_threshold = self.config.get("intent_manager.confidence_threshold", 0.4)
        
        # Performance tracking
        self._detection_history: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, float] = {}
        
        # State management
        self._initialization_lock = asyncio.Lock()
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.info("EnhancedIntentManager initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            self.metrics = self.container.get(MetricsCollector)
            self.tracer = self.container.get(TraceManager)
            
            # Register intent metrics
            self.metrics.register_counter("intent_detections_total")
            self.metrics.register_counter("intent_routing_total")
            self.metrics.register_histogram("intent_detection_duration_seconds")
            self.metrics.register_histogram("intent_confidence_score")
            self.metrics.register_gauge("intent_definitions_count")
            self.metrics.register_counter("intent_learning_events_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")
            self.metrics = None
            self.tracer = None

    async def initialize(self) -> None:
        """Initialize the intent manager."""
        async with self._initialization_lock:
            try:
                self.logger.info("Initializing intent manager...")
                
                # Load intent definitions
                await self._load_intent_definitions()
                
                # Setup detectors
                await self._setup_detectors()
                
                # Setup router and learner
                self.intent_router = IntentRouter(self.skill_registry, self.workflow_orchestrator)
                
                if self._enable_learning:
                    self.intent_learner = IntentLearner(self.feedback_processor, self.preference_learner)
                
                # Register event handlers
                await self._register_event_handlers()
                
                # Setup health monitoring
                self.health_check.register_component("intent_manager", self._health_check_callback)
                
                # Start background tasks
                await self._start_background_tasks()
                
                self.logger.info(f"Intent manager initialized with {len(self.intent_definitions)} intent definitions")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize intent manager: {str(e)}")
                raise IntentError(f"Intent manager initialization failed: {str(e)}")

    async def _load_intent_definitions(self) -> None:
        """Load intent definitions from configuration."""
        # Load from configuration
        intent_configs = self.config.get("intent_manager.intents", {})
        
        for intent_name, config in intent_configs.items():
            try:
                intent_def = self._parse_intent_definition(intent_name, config)
                self.intent_definitions[intent_name] = intent_def
                
            except Exception as e:
                self.logger.error(f"Failed to load intent definition {intent_name}: {str(e)}")
        
        # Load built-in intents
        await self._load_builtin_intents()
        
        # Update metrics
        if self.metrics:
            self.metrics.set("intent_definitions_count", len(self.intent_definitions))

    def _parse_intent_definition(self, intent_name: str, config: Dict[str, Any]) -> IntentDefinition:
        """Parse intent definition from configuration."""
        # Parse patterns
        patterns = []
        for pattern_config in config.get('patterns', []):
            pattern = IntentPattern(
                pattern_id=pattern_config.get('id', f"{intent_name}_pattern_{len(patterns)}"),
                intent_name=intent_name,
                patterns=pattern_config.get('patterns', []),
                weight=pattern_config.get('weight', 1.0),
                required_slots=pattern_config.get('required_slots', []),
                optional_slots=pattern_config.get('optional_slots', [])
            )
            patterns.append(pattern)
        
        # Create intent definition
        intent_def = IntentDefinition(
            intent_name=intent_name,
            intent_type=IntentType(config.get('type', 'action')),
            category=IntentCategory(config.get('category', 'productivity')),
            scope=IntentScope(config.get('scope', 'session')),
            description=config.get('description', ''),
            examples=config.get('examples', []),
            patterns=patterns,
            keywords=set(config.get('keywords', [])),
            required_slots=config.get('required_slots', []),
            optional_slots=config.get('optional_slots', []),
            slot_definitions=config.get('slot_definitions', {}),
            skill_name=config.get('skill_name'),
            workflow_id=config.get('workflow_id'),
            component_name=config.get('component_name'),
            context_requirements=config.get('context_requirements', {}),
            priority=config.get('priority', 5)
        )
        
        return intent_def

    async def _load_builtin_intents(self) -> None:
        """Load built-in intent definitions."""
        builtin_intents = {
            "greeting": IntentDefinition(
                intent_name="greeting",
                intent_type=IntentType.CONVERSATION,
                category=IntentCategory.COMMUNICATION,
                scope=IntentScope.SESSION,
                description="User greeting",
                examples=["hello", "hi", "good morning"],
                patterns=[
                    IntentPattern(
                        pattern_id="greeting_pattern_1",
                        intent_name="greeting",
                        patterns=[r"\b(hello|hi|hey|good\s+(morning|afternoon|evening))\b"],
                        weight=1.0
                    )
                ],
                keywords={"hello", "hi", "hey", "good morning", "good afternoon", "good evening"}
            ),
            
            "help_request": IntentDefinition(
                intent_name="help_request",
                intent_type=IntentType.QUESTION,
                category=IntentCategory.INFORMATION,
                scope=IntentScope.SESSION,
                description="User requesting help",
                examples=["help", "I need help", "can you help me"],
                patterns=[
                    IntentPattern(
                        pattern_id="help_pattern_1",
                        intent_name="help_request",
                        patterns=[r"\b(help|assist|support)\b"],
                        weight=1.0
                    )
                ],
                keywords={"help", "assist", "support"}
            ),
            
            "task_execution": IntentDefinition(
                intent_name="task_execution",
                intent_type=IntentType.TASK,
                category=IntentCategory.PRODUCTIVITY,
                scope=IntentScope.USER,
                description="Execute a task",
                examples=["create a document", "send an email", "schedule a meeting"],
                patterns=[
                    IntentPattern(
                        pattern_id="task_pattern_1",
                        intent_name="task_execution",
                        patterns=[r"\b(create|make|do|execute|run|perform)\s+(.+)"],
                        weight=1.0
                    )
                ],
                keywords={"create", "make", "do", "execute", "run", "perform"}
            ),
            
            "information_query": IntentDefinition(
                intent_name="information_query",
                intent_type=IntentType.QUESTION,
                category=IntentCategory.INFORMATION,
                scope=IntentScope.GLOBAL,
                description="Query for information",
                examples=["what is", "tell me about", "how do I"],
                patterns=[
                    IntentPattern(
                        pattern_id="query_pattern_1",
                        intent_name="information_query",
                        patterns=[r"\b(what|how|when|where|why|tell\s+me)\b"],
                        weight=1.0
                    )
                ],
                keywords={"what", "how", "when", "where", "why", "tell me"}
            )
        }
        
        # Add to intent definitions
        for intent_name, intent_def in builtin_intents.items():
            if intent_name not in self.intent_definitions:
                self.intent_definitions[intent_name] = intent_def

    async def _setup_detectors(self) -> None:
        """Setup intent detectors based on configuration."""
        # Pattern-based detector
        pattern_detector = PatternBasedDetector(self.intent_definitions)
        self.intent_detectors["pattern"] = pattern_detector
        
        # ML-based detector
        ml_detector = MLBasedDetector(self.model_router, self.cache_strategy)
        self.intent_detectors["ml"] = ml_detector
        
        # Hybrid detector
        hybrid_detector = HybridDetector(pattern_detector, ml_detector)
        self.intent_detectors["hybrid"] = hybrid_detector
        
        self.logger.info(f"Setup {len(self.intent_detectors)} intent detectors")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # User interaction events
        self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
        
        # Skill and workflow events
        self.event_bus.subscribe("skill_executed", self._handle_skill_executed)
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Performance monitoring
        self._background_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        # Learning optimization
        if self._enable_learning:
            self._background_tasks.append(
                asyncio.create_task(self._learning_optimization_loop())
            )
        
        # Intent definition updates
        self._background_tasks.append(
            asyncio.create_task(self._intent_update_loop())
        )

    @handle_exceptions
    async def detect_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[IntentDetectionResult]:
        """
        Detect intents from user input text.
        
        Args:
            text: Input text to analyze
            context: Optional context information
            user_id: Optional user identifier
            session_id: Optional session identifier
            
        Returns:
            List of detected intents sorted by confidence
        """
        start_time = time.time()
        context = context or {}
        
        # Add user and session to context
        if user_id:
            context['user_id'] = user_id
        if session_id:
            context['session_id'] = session_id
        
        try:
            with self.tracer.trace("intent_detection") if self.tracer else None:
                # Get detector
                detector = self.intent_detectors.get(self._detection_mode)
                if not detector:
                    raise IntentError(f"Detector mode {self._detection_mode} not available")
                
                # Detect intents
                results = await detector.detect_intent(text, context)
                
                # Filter by confidence threshold
                filtered_results = [
                    r for r in results 
                    if r.confidence >= self._confidence_threshold
                ]
                
                # Enhance results with context
                for result in filtered_results:
                    result.session_id = session_id
                    result.user_id = user_id
                    result.processing_time = time.time() - start_time
                
                # Record detection history
                detection_record = {
                    "text": text,
                    "detected_intents": [r.intent_name for r in filtered_results],
                    "top_confidence": filtered_results[0].confidence if filtered_results else 0.0,
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now(timezone.utc),
                    "user_id": user_id,
                    "session_id": session_id
                }
                self._detection_history.append(detection_record)
                
                # Emit events
                if filtered_results:
                    await self.event_bus.emit(IntentDetected(
                        intent_name=filtered_results[0].intent_name,
                        confidence=filtered_results[0].confidence,
                        user_id=user_id,
                        session_id=session_id,
                        alternatives=[r.intent_name for r in filtered_results[1:]]
                    ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("intent_detections_total")
                    self.metrics.record("intent_detection_duration_seconds", time.time() - start_time)
                    if filtered_results:
                        self.metrics.record("intent_confidence_score", filtered_results[0].confidence)
                
                self.logger.info(
                    f"Detected {len(filtered_results)} intents for '{text[:50]}...' "
                    f"in {time.time() - start_time:.3f}s"
                )
                
                return filtered_results
                
        except Exception as e:
            self.logger.error(f"Intent detection failed for text '{text[:50]}...': {str(e)}")
            return []

    @handle_exceptions
    async def route_intent(
        self,
        intent_result: IntentDetectionResult,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a detected intent to appropriate execution target.
        
        Args:
            intent_result: Detected intent result
            context: Optional context information
            
        Returns:
            Routing result with execution information
        """
        if not self.intent_router:
            raise IntentError("Intent router not initialized")
        
        start_time = time.time()
        context = context or {}
        
        try:
            with self.tracer.trace("intent_routing") if self.tracer else None:
                # Route intent
                routing_result = await self.intent_router.route_intent(intent_result, context)
                
                # Record routing time
                routing_time = time.time() - start_time
                routing_result['routing_time'] = routing_time
                
                # Emit routing event
                await self.event_bus.emit(IntentRoutingCompleted(
                    intent_name=intent_result.intent_name,
                    execution_target=routing_result.get('execution_target'),
                    target_name=routing_result.get('target_name'),
                    success=routing_result.get('success', False),
                    routing_time=routing_time
                ))
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("intent_routing_total")
                
                self.logger.info(
                    f"Routed intent '{intent_result.intent_name}' to "
                    f"{routing_result.get('execution_target', 'unknown')} in {routing_time:.3f}s"
                )
                
                return routing_result
                
        except Exception as e:
            self.logger.error(f"Intent routing failed for {intent_result.intent_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "intent": intent_result.intent_name
            }

    @handle_exceptions
    async def process_user_input(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        execute_immediately: bool = True
    ) -> Dict[str, Any]:
        """
        Complete intent processing pipeline: detect, route, and optionally execute.
        
        Args:
            text: User input text
            context: Optional context information
            user_id: Optional user identifier
            session_id: Optional session identifier
            execute_immediately: Whether to execute the intent immediately
            
        Returns:
            Complete processing result
        """
        processing_start = time.time()
        
        try:
            # Detect intents
            detected_intents = await self.detect_intent(text, context, user_id, session_id)
            
            if not detected_intents:
                return {
                    "success": False,
                    "message": "No intents detected",
                    "processing_time": time.time() - processing_start
                }
            
            # Use highest confidence intent
            primary_intent = detected_intents[0]
            
            # Route intent
            routing_result = await self.route_intent(primary_intent, context)
            
            result = {
                "success": True,
                "detected_intents": [
                    {
                        "intent_name": intent.intent_name,
                        "confidence": intent.confidence,
                        "type": intent.intent_type.value,
                        "category": intent.category.value
                    }
                    for intent in detected_intents
                ],
                "primary_intent": {
                    "intent_name": primary_intent.intent_name,
                    "confidence": primary_intent.confidence,
                    "type": primary_intent.intent_type.value,
                    "category": primary_intent.category.value,
                    "slots": [
                        {
                            "name": slot.name,
                            "value": slot.value,
                            "type": slot.entity_type,
                            "confidence": slot.confidence
                        }
                        for slot in primary_intent.slots
                    ]
                },
                "routing_result": routing_result,
                "processing_time": time.time() - processing_start
            }
            
            # Learn from interaction
            if self._enable_learning and self.intent_learner:
                await self.intent_learner.learn_from_interaction(
                    text, primary_intent, None, context or {}
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Intent processing failed for '{text[:50]}...': {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - processing_start
            }

    async def register_intent(self, intent_definition: IntentDefinition) -> None:
        """Register a new intent definition."""
        try:
            # Validate intent definition
            self._validate_intent_definition(intent_definition)
            
            # Add to definitions
            self.intent_definitions[intent_definition.intent_name] = intent_definition
            
            # Update detectors
            await self._update_detectors()
            
            # Update metrics
            if self.metrics:
                self.metrics.set("intent_definitions_count", len(self.intent_definitions))
            
            # Emit event
            await self.event_bus.emit(IntentCategoryUpdated(
                intent_name=intent_definition.intent_name,
                category=intent_definition.category.value,
                operation="registered"
            ))
            
            self.logger.info(f"Registered new intent: {intent_definition.intent_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register intent {intent_definition.intent_name}: {str(e)}")
            raise IntentError(f"Intent registration failed: {str(e)}")

    def _validate_intent_definition(self, intent_def: IntentDefinition) -> None:
        """Validate intent definition."""
        if not intent_def.intent_name:
            raise IntentError("Intent name is required")
        
        if not intent_def.description:
            raise IntentError("Intent description is required")
        
        if not intent_def.patterns and not intent_def.keywords:
            raise IntentError("Intent must have patterns or keywords")

    async def _update_detectors(self) -> None:
        """Update detectors with new intent definitions."""
        # Update pattern detector
        if "pattern" in self.intent_detectors:
            pattern_detector = PatternBasedDetector(self.intent_definitions)
            self.intent_detectors["pattern"] = pattern_detector
        
        # Update hybrid detector
        if "hybrid" in self.intent_detectors:
            hybrid_detector = HybridDetector(
                self.intent_detectors["pattern"],
                self.intent_detectors["ml"]
            )
            self.intent_detectors["hybrid"] = hybrid_detector

    async def get_intent_analytics(self) -> Dict[str, Any]:
        """Get intent detection and processing analytics."""
        if not self._detection_history:
            return {"message": "No detection history available"}
        
        total_detections = len(self._detection_history)
        successful_detections = sum(1 for record in self._detection_history 
                                  if record["detected_intents"])
        
        avg_processing_time = np.mean([record["processing_time"] 
                                     for record in self._detection_history])
        
        avg_confidence = np.mean([record["top_confidence"] 
                                for record in self._detection_history 
                                if record["top_confidence"] > 0])
        
        # Intent frequency analysis
        intent_frequency = defaultdict(int)
        for record in self._detection_history:
            for intent in record["detected_intents"]:
                intent_frequency[intent] += 1
        
        return {
            "total_detections": total_detections,
            "successful_detections": successful_detections,
            "success_rate": successful_detections / total_detections if total_detections > 0 else 0,
            "average_processing_time": avg_processing_time,
            "average_confidence": avg_confidence,
            "most_frequent_intents": dict(sorted(intent_frequency.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]),
            "total_intent_definitions": len(self.intent_definitions),
            "detection_mode": self._detection_mode
        }

    def list_intents(self, category: Optional[IntentCategory] = None,
                    intent_type: Optional[IntentType] = None) -> List[Dict[str, Any]]:
        """List available intent definitions with optional filtering."""
        intents = []
        
        for intent_def in self.intent_definitions.values():
            if category and intent_def.category != category:
                continue
            if intent_type and intent_def.intent_type != intent_type:
                continue
            
            intents.append({
                "intent_name": intent_def.intent_name,
                "type": intent_def.intent_type.value,
                "category": intent_def.category.value,
                "scope": intent_def.scope.value,
                "description": intent_def.description,
                "keywords": list(intent_def.keywords),
                "priority": intent_def.priority,
                "is_active": intent_def.is_active,
                "skill_name": intent_def.skill_name,
                "workflow_id": intent_def.workflow_id,
                "examples": intent_def.examples[:3]  # First 3 examples
            })
        
        return sorted(intents, key=lambda x: x["priority"], reverse=True)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Calculate performance metrics
                if self._detection_history:
                    recent_detections = [
                        record for record in self._detection_history
                        if (datetime.now(timezone.utc) - record["timestamp"]).total_seconds() < 300
                    ]
                    
                    if recent_detections:
                        avg_time = np.mean([r["processing_time"] for r in recent_detections])
                        success_rate = sum(1 for r in recent_detections if r["detected_intents"]) / len(recent_detections)
                        
                        self._performance_metrics.update({
                            "avg_detection_time": avg_time,
                            "recent_success_rate": success_rate,
                            "recent_detection_count": len(recent_detections)
                        })
                        
                        # Update metrics
                        if self.metrics:
                            self.metrics.set("intent_manager_avg_detection_time", avg_time)
                            self.metrics.set("intent_manager_success_rate", success_rate)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {str(e)}")

    async def _learning_optimization_loop(self) -> None:
        """Background task for learning optimization."""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                if self.intent_learner:
                    insights = await self.intent_learner.get_learning_insights()
                    
                    # Adjust confidence threshold based on accuracy
                    if insights.get("accuracy", 0) < 0.7:
                        self._confidence_threshold = min(0.8, self._confidence_threshold + 0.05)
                    elif insights.get("accuracy", 0) > 0.9:
                        self._confidence_threshold = max(0.2, self._confidence_threshold - 0.05)
                    
                    self.logger.debug(f"Updated confidence threshold to {self._confidence_threshold}")
                
            except Exception as e:
                self.logger.error(f"Learning optimization error: {str(e)}")

    async def _intent_update_loop(self) -> None:
        """Background task for intent definition updates."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                # Check for configuration updates
                # This would reload intent definitions from external sources
                # For now, just log that we're checking
                self.logger.debug("Checking for intent definition updates")
                
            except Exception as e:
                self.logger.error(f"Intent update check error: {str(e)}")

    async def _handle_session_started(self, event) -> None:
        """Handle session started events."""
        # Initialize session-specific intent context
        pass

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        # Cleanup session-specific data
        pass

    async def _handle_user_feedback(self, event) -> None:
        """Handle user feedback events."""
        if self._enable_learning and self.intent_learner:
            # Process feedback for intent learning
            feedback_data = event.data
            if feedback_data.get("type") == "intent_correction":
                await self.intent_learner.learn_from_interaction(
                    feedback_data.get("text", ""),
                    feedback_data.get("detected_intent"),
                    feedback_data.get("correct_intent"),
                    feedback_data.get("context", {})
                )

    async def _handle_skill_executed(self, event) -> None:
        """Handle skill execution events."""
        # Update intent-skill mapping statistics
        pass

    async def _handle_workflow_completed(self, event) -> None:
        """Handle workflow completion events."""
        # Update intent-workflow mapping statistics
        pass

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the intent manager."""
        try:
            detector_health = {}
            for name, detector in self.intent_detectors.items():
                try:
                    # Basic health check - verify detector is responsive
                    test_result = await detector.detect_intent("hello", {})
                    detector_health[name] = "healthy"
                except Exception as e:
                    detector_health[name] = f"unhealthy: {str(e)}"
            
            return {
                "status": "healthy" if all(h == "healthy" for h in detector_health.values()) else "degraded",
                "intent_definitions": len(self.intent_definitions),
                "detection_mode": self._detection_mode,
                "confidence_threshold": self._confidence_threshold,
                "detector_health": detector_health,
                "recent_detections": len(self._detection_history),
                "performance_metrics": self._performance_metrics
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self) -> None:
        """Cleanup intent manager resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Cleanup detectors
            for detector in self.intent_detectors.values():
                if hasattr(detector, 'cleanup'):
                    await detector.cleanup()
            
            self.logger.info("Intent manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during intent manager cleanup: {str(e)}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_background_tasks') and self._background_tasks:
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()
        except Exception:
            pass  # Ignore cleanup errors in destructor
