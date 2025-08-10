"""
Context awareness and management for the AI assistant.

This module handles multi-layered context tracking, including environmental,
temporal, emotional, task-based, and user-specific contexts across all modalities.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from src.core.events.event_bus import EventBus
from src.core.events.event_types import Event, EventType
from src.core.error_handling import (
    ValidationError
)
from src.core.config.loader import ConfigLoader
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.storage.vector_store import VectorStore
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.processing.multimodal.fusion_strategies import FusionStrategy
from src.integrations.cache.cache_strategy import CacheStrategy
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.vision.detectors.expression_analyzer import ExpressionAnalyzer
from src.processing.speech.emotion_detection import EmotionDetector

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context tracked by the system."""
    TEMPORAL = auto()          # Time-based context
    SPATIAL = auto()           # Location/environment context
    EMOTIONAL = auto()         # Emotional state context
    CONVERSATIONAL = auto()    # Dialogue context
    TASK = auto()              # Current task/activity context
    USER_PREFERENCE = auto()   # User preferences and patterns
    SOCIAL = auto()            # Social/relationship context
    COGNITIVE = auto()         # User's cognitive state
    ENVIRONMENTAL = auto()     # Physical environment context
    MULTIMODAL = auto()        # Cross-modal context


class ContextPriority(Enum):
    """Priority levels for context information."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


@dataclass
class ContextFrame:
    """Represents a single frame of contextual information."""
    frame_id: str
    context_type: ContextType
    timestamp: datetime
    data: Dict[str, Any]
    source: str  # Which component provided this context
    confidence: float = 1.0
    priority: ContextPriority = ContextPriority.MEDIUM
    ttl_seconds: Optional[int] = None  # Time to live
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this context frame has expired."""
        if self.ttl_seconds is None:
            return False
        expiry_time = self.timestamp + timedelta(seconds=self.ttl_seconds)
        return datetime.now(timezone.utc) > expiry_time
    
    def age_seconds(self) -> float:
        """Get the age of this context frame in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()


@dataclass
class ContextStack:
    """Maintains a stack of related contexts with hierarchy."""
    stack_id: str
    context_type: ContextType
    frames: List[ContextFrame] = field(default_factory=list)
    max_depth: int = 10
    
    def push(self, frame: ContextFrame) -> None:
        """Push a new frame onto the stack."""
        if len(self.frames) >= self.max_depth:
            self.frames.pop(0)  # Remove oldest
        self.frames.append(frame)
    
    def peek(self) -> Optional[ContextFrame]:
        """Get the most recent frame without removing it."""
        return self.frames[-1] if self.frames else None
    
    def get_recent(self, n: int = 5) -> List[ContextFrame]:
        """Get the n most recent frames."""
        return self.frames[-n:] if len(self.frames) > n else self.frames


@dataclass
class MultimodalContext:
    """Aggregated context from multiple modalities."""
    context_id: str
    timestamp: datetime
    text_context: Optional[Dict[str, Any]] = None
    audio_context: Optional[Dict[str, Any]] = None
    visual_context: Optional[Dict[str, Any]] = None
    fusion_result: Optional[Dict[str, Any]] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class ContextManager:
    """
    Manages multi-layered context awareness for the AI assistant.
    
    This class maintains and updates various types of context, providing
    a unified interface for context-aware decision making across the system.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        memory_manager: MemoryManager,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        cache_strategy: CacheStrategy,
        sentiment_analyzer: SentimentAnalyzer,
        entity_extractor: EntityExtractor,
        expression_analyzer: ExpressionAnalyzer,
        emotion_detector: EmotionDetector,
        fusion_strategy: FusionStrategy,
        config_loader: ConfigLoader
    ):
        """Initialize the context manager with required dependencies."""
        self.event_bus = event_bus
        self.memory_manager = memory_manager
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.cache = cache_strategy
        self.sentiment_analyzer = sentiment_analyzer
        self.entity_extractor = entity_extractor
        self.expression_analyzer = expression_analyzer
        self.emotion_detector = emotion_detector
        self.fusion_strategy = fusion_strategy
        self.config = config_loader.get_config("context")
        
        # Context storage
        self.context_stacks: Dict[str, Dict[ContextType, ContextStack]] = defaultdict(dict)
        self.active_contexts: Dict[str, List[ContextFrame]] = defaultdict(list)
        self.multimodal_contexts: Dict[str, MultimodalContext] = {}
        
        # Configuration
        self.context_window_size = self.config.get("window_size", 20)
        self.context_decay_rate = self.config.get("decay_rate", 0.95)
        self.min_confidence_threshold = self.config.get("min_confidence", 0.3)
        self.cache_ttl = self.config.get("cache_ttl", 300)
        self.max_contexts_per_session = self.config.get("max_contexts", 1000)
        
        # Context update handlers
        self.context_handlers: Dict[ContextType, List[Callable]] = defaultdict(list)
        
        # Context fusion weights
        self.fusion_weights = self.config.get("fusion_weights", {
            "text": 0.4,
            "audio": 0.3,
            "visual": 0.3
        })
        
        # Initialize context patterns
        self.context_patterns = self._load_context_patterns()
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        # Start background tasks
        asyncio.create_task(self._cleanup_expired_contexts())
        
        logger.info("ContextManager initialized successfully")
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant system events."""
        self.event_bus.subscribe(EventType.USER_INPUT, self._handle_input_event)
        self.event_bus.subscribe(EventType.SENSOR_DATA, self._handle_sensor_event)
        self.event_bus.subscribe(EventType.MEMORY_UPDATED, self._handle_memory_event)
        self.event_bus.subscribe(EventType.SKILL_EXECUTED, self._handle_skill_event)
        self.event_bus.subscribe(EventType.EMOTION_DETECTED, self._handle_emotion_event)
    
    def _load_context_patterns(self) -> Dict[str, Any]:
        """Load predefined context patterns for recognition."""
        return self.config.get("context_patterns", {
            "morning_routine": {
                "time_range": (6, 10),
                "activities": ["wake_up", "breakfast", "exercise", "planning"],
                "mood": "energetic"
            },
            "work_focus": {
                "time_range": (9, 17),
                "activities": ["coding", "meetings", "documentation"],
                "mood": "focused"
            },
            "evening_relaxation": {
                "time_range": (18, 22),
                "activities": ["dinner", "entertainment", "reading"],
                "mood": "relaxed"
            }
        })
    
    async def update_context(
        self,
        session_id: str,
        context_type: ContextType,
        data: Dict[str, Any],
        source: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
        ttl_seconds: Optional[int] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Update context for a session.
        
        Args:
            session_id: Session identifier
            context_type: Type of context being updated
            data: Context data
            source: Source component/system
            priority: Context priority level
            ttl_seconds: Time to live in seconds
            confidence: Confidence score for this context
            
        Returns:
            Context frame ID
        """
        try:
            # Create context frame
            frame = ContextFrame(
                frame_id=str(uuid.uuid4()),
                context_type=context_type,
                timestamp=datetime.now(timezone.utc),
                data=data,
                source=source,
                confidence=confidence,
                priority=priority,
                ttl_seconds=ttl_seconds
            )
            
            # Validate frame
            self._validate_context_frame(frame)
            
            # Add to appropriate stack
            if session_id not in self.context_stacks:
                self.context_stacks[session_id] = {}
            
            if context_type not in self.context_stacks[session_id]:
                self.context_stacks[session_id][context_type] = ContextStack(
                    stack_id=str(uuid.uuid4()),
                    context_type=context_type,
                    max_depth=self.context_window_size
                )
            
            self.context_stacks[session_id][context_type].push(frame)
            
            # Add to active contexts
            self.active_contexts[session_id].append(frame)
            
            # Limit active contexts
            if len(self.active_contexts[session_id]) > self.max_contexts_per_session:
                self.active_contexts[session_id].pop(0)
            
            # Cache for fast retrieval
            await self._cache_context(session_id, frame)
            
            # Process context updates
            await self._process_context_update(session_id, frame)
            
            # Emit context updated event
            await self.event_bus.emit(Event(
                type=EventType.CONTEXT_UPDATED,
                data={
                    "session_id": session_id,
                    "context_type": context_type.name,
                    "frame_id": frame.frame_id,
                    "source": source
                }
            ))
            
            logger.debug(f"Updated {context_type.name} context for session {session_id}")
            return frame.frame_id
            
        except Exception as e:
            logger.error(f"Failed to update context: {str(e)}")
            raise
    
    async def get_current_context(
        self,
        session_id: str,
        context_types: Optional[List[ContextType]] = None,
        include_expired: bool = False
    ) -> Dict[str, Any]:
        """
        Get current context for a session.
        
        Args:
            session_id: Session identifier
            context_types: Specific context types to retrieve
            include_expired: Whether to include expired contexts
            
        Returns:
            Dictionary of current contexts by type
        """
        # Check cache first
        cache_key = f"context:{session_id}:{context_types}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        result = {}
        
        if session_id not in self.context_stacks:
            return result
        
        types_to_check = context_types or list(ContextType)
        
        for context_type in types_to_check:
            if context_type in self.context_stacks[session_id]:
                stack = self.context_stacks[session_id][context_type]
                recent_frames = stack.get_recent()
                
                # Filter expired if needed
                if not include_expired:
                    recent_frames = [f for f in recent_frames if not f.is_expired()]
                
                if recent_frames:
                    # Aggregate recent frames with decay
                    aggregated = self._aggregate_context_frames(recent_frames)
                    result[context_type.name] = aggregated
        
        # Cache result
        await self.cache.set(cache_key, result, ttl=self.cache_ttl)
        
        return result
    
    async def get_multimodal_context(
        self,
        session_id: str,
        modalities: Optional[List[str]] = None
    ) -> MultimodalContext:
        """
        Get fused multimodal context for a session.
        
        Args:
            session_id: Session identifier
            modalities: Specific modalities to include
            
        Returns:
            Fused multimodal context
        """
        # Get individual modality contexts
        current_contexts = await self.get_current_context(session_id)
        
        # Extract modality-specific contexts
        text_context = self._extract_text_context(current_contexts)
        audio_context = self._extract_audio_context(current_contexts)
        visual_context = self._extract_visual_context(current_contexts)
        
        # Apply fusion strategy
        fusion_input = {}
        if not modalities or "text" in modalities:
            fusion_input["text"] = text_context
        if not modalities or "audio" in modalities:
            fusion_input["audio"] = audio_context
        if not modalities or "visual" in modalities:
            fusion_input["visual"] = visual_context
        
        fusion_result = await self.fusion_strategy.fuse(fusion_input, self.fusion_weights)
        
        # Create multimodal context
        multimodal = MultimodalContext(
            context_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            text_context=text_context,
            audio_context=audio_context,
            visual_context=visual_context,
            fusion_result=fusion_result,
            confidence_scores={
                "text": text_context.get("confidence", 0) if text_context else 0,
                "audio": audio_context.get("confidence", 0) if audio_context else 0,
                "visual": visual_context.get("confidence", 0) if visual_context else 0,
                "fusion": fusion_result.get("confidence", 0)
            }
        )
        
        self.multimodal_contexts[session_id] = multimodal
        
        return multimodal
    
    async def infer_context(
        self,
        session_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Infer context from available information.
        
        Args:
            session_id: Session identifier
            input_data: Input data to analyze
            
        Returns:
            Inferred context information
        """
        inferred = {}
        
        # Temporal context inference
        temporal_context = await self._infer_temporal_context(input_data)
        if temporal_context:
            await self.update_context(
                session_id,
                ContextType.TEMPORAL,
                temporal_context,
                source="context_inference",
                confidence=temporal_context.get("confidence", 0.8)
            )
            inferred["temporal"] = temporal_context
        
        # Activity context inference
        activity_context = await self._infer_activity_context(session_id, input_data)
        if activity_context:
            await self.update_context(
                session_id,
                ContextType.TASK,
                activity_context,
                source="context_inference",
                confidence=activity_context.get("confidence", 0.7)
            )
            inferred["activity"] = activity_context
        
        # Emotional context inference
        emotional_context = await self._infer_emotional_context(session_id, input_data)
        if emotional_context:
            await self.update_context(
                session_id,
                ContextType.EMOTIONAL,
                emotional_context,
                source="context_inference",
                confidence=emotional_context.get("confidence", 0.8)
            )
            inferred["emotional"] = emotional_context
        
        # Pattern matching
        matched_patterns = self._match_context_patterns(inferred)
        if matched_patterns:
            inferred["patterns"] = matched_patterns
        
        return inferred
    
    async def get_relevant_context(
        self,
        session_id: str,
        query: str,
        context_types: Optional[List[ContextType]] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get context relevant to a specific query.
        
        Args:
            session_id: Session identifier
            query: Query to match against
            context_types: Specific context types to search
            max_results: Maximum number of results
            
        Returns:
            List of relevant context items
        """
        # Get current contexts
        current_contexts = await self.get_current_context(
            session_id, context_types
        )
        
        # Search in vector store for similar contexts
        query_embedding = await self.vector_store.embed_text(query)
        similar_contexts = await self.vector_store.search(
            query_embedding,
            filter={"session_id": session_id},
            limit=max_results * 2  # Get more for filtering
        )
        
        # Combine and rank results
        relevant_contexts = []
        
        # Add current contexts with high relevance
        for ctx_type, ctx_data in current_contexts.items():
            if self._is_relevant_to_query(ctx_data, query):
                relevant_contexts.append({
                    "type": ctx_type,
                    "data": ctx_data,
                    "relevance": self._calculate_relevance(ctx_data, query),
                    "recency": 1.0  # Current context has max recency
                })
        
        # Add similar historical contexts
        for ctx in similar_contexts:
            if ctx["score"] > self.min_confidence_threshold:
                relevant_contexts.append({
                    "type": ctx.get("metadata", {}).get("type", "unknown"),
                    "data": ctx["data"],
                    "relevance": ctx["score"],
                    "recency": self._calculate_recency(ctx.get("timestamp"))
                })
        
        # Sort by combined score
        relevant_contexts.sort(
            key=lambda x: x["relevance"] * 0.7 + x["recency"] * 0.3,
            reverse=True
        )
        
        return relevant_contexts[:max_results]
    
    async def predict_future_context(
        self,
        session_id: str,
        time_horizon_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Predict future context based on patterns and history.
        
        Args:
            session_id: Session identifier
            time_horizon_minutes: How far to predict into the future
            
        Returns:
            Predicted future contexts
        """
        # Get user's historical patterns
        user_id = await self._get_user_id(session_id)
        historical_patterns = await self.memory_manager.get_user_patterns(user_id)
        
        # Get current context
        current_context = await self.get_current_context(session_id)
        
        predictions = {}
        
        # Temporal predictions
        current_time = datetime.now(timezone.utc)
        future_time = current_time + timedelta(minutes=time_horizon_minutes)
        
        # Activity predictions based on patterns
        predicted_activities = self._predict_activities(
            historical_patterns,
            current_context,
            future_time
        )
        if predicted_activities:
            predictions["activities"] = predicted_activities
        
        # Mood/energy predictions
        predicted_mood = self._predict_mood(
            historical_patterns,
            current_context,
            future_time
        )
        if predicted_mood:
            predictions["mood"] = predicted_mood
        
        # Location predictions
        predicted_location = self._predict_location(
            historical_patterns,
            current_context,
            future_time
        )
        if predicted_location:
            predictions["location"] = predicted_location
        
        return predictions
    
    async def register_context_handler(
        self,
        context_type: ContextType,
        handler: Callable
    ) -> None:
        """Register a handler for context updates."""
        self.context_handlers[context_type].append(handler)
        logger.info(f"Registered handler for {context_type.name} context")
    
    async def get_context_summary(
        self,
        session_id: str,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of context over a time window.
        
        Args:
            session_id: Session identifier
            time_window_minutes: Time window to summarize
            
        Returns:
            Context summary
        """
        summary = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "contexts": {}
        }
        
        # Calculate time boundary
        if time_window_minutes:
            time_boundary = datetime.now(timezone.utc) - timedelta(
                minutes=time_window_minutes
            )
        else:
            time_boundary = None
        
        # Summarize each context type
        if session_id in self.context_stacks:
            for ctx_type, stack in self.context_stacks[session_id].items():
                frames = stack.frames
                
                # Filter by time if specified
                if time_boundary:
                    frames = [f for f in frames if f.timestamp > time_boundary]
                
                if frames:
                    summary["contexts"][ctx_type.name] = {
                        "frame_count": len(frames),
                        "latest": frames[-1].data,
                        "average_confidence": np.mean([f.confidence for f in frames]),
                        "dominant_values": self._extract_dominant_values(frames)
                    }
        
        # Add multimodal summary if available
        if session_id in self.multimodal_contexts:
            multimodal = self.multimodal_contexts[session_id]
            summary["multimodal"] = {
                "fusion_confidence": multimodal.confidence_scores.get("fusion", 0),
                "modality_confidences": multimodal.confidence_scores,
                "timestamp": multimodal.timestamp.isoformat()
            }
        
        return summary
    
    def _validate_context_frame(self, frame: ContextFrame) -> None:
        """Validate a context frame."""
        if not frame.data:
            raise ValidationError("Context frame must contain data")
        
        if frame.confidence < 0 or frame.confidence > 1:
            raise ValidationError("Confidence must be between 0 and 1")
        
        if frame.ttl_seconds and frame.ttl_seconds < 0:
            raise ValidationError("TTL must be positive")
    
    def _aggregate_context_frames(
        self,
        frames: List[ContextFrame]
    ) -> Dict[str, Any]:
        """Aggregate multiple context frames with decay."""
        if not frames:
            return {}
        
        # Sort by timestamp (newest first)
        sorted_frames = sorted(frames, key=lambda f: f.timestamp, reverse=True)
        
        aggregated = {
            "latest": sorted_frames[0].data,
            "timestamp": sorted_frames[0].timestamp.isoformat(),
            "confidence": sorted_frames[0].confidence,
            "source": sorted_frames[0].source
        }
        
        # Apply decay and aggregate values
        if len(sorted_frames) > 1:
            weighted_values = defaultdict(list)
            total_weight = 0
            
            for i, frame in enumerate(sorted_frames):
                weight = self.context_decay_rate ** i
                total_weight += weight
                
                for key, value in frame.data.items():
                    if isinstance(value, (int, float)):
                        weighted_values[key].append((value, weight))
            
            # Calculate weighted averages
            aggregated["aggregated_values"] = {}
            for key, values in weighted_values.items():
                weighted_sum = sum(v * w for v, w in values)
                aggregated["aggregated_values"][key] = weighted_sum / total_weight
        
        return aggregated
    
    async def _cache_context(
        self,
        session_id: str,
        frame: ContextFrame
    ) -> None:
        """Cache context frame for fast retrieval."""
        cache_key = f"context_frame:{session_id}:{frame.context_type.name}:{frame.frame_id}"
        await self.cache.set(
            cache_key,
            {
                "frame_id": frame.frame_id,
                "type": frame.context_type.name,
                "data": frame.data,
                "timestamp": frame.timestamp.isoformat(),
                "confidence": frame.confidence
            },
            ttl=frame.ttl_seconds or self.cache_ttl
        )
    
    async def _process_context_update(
        self,
        session_id: str,
        frame: ContextFrame
    ) -> None:
        """Process context update through registered handlers."""
        # Call registered handlers
        handlers = self.context_handlers.get(frame.context_type, [])
        for handler in handlers:
            try:
                await handler(session_id, frame)
            except Exception as e:
                logger.error(f"Error in context handler: {str(e)}")
        
        # Store in vector store for similarity search
        await self.vector_store.store(
            vector=await self.vector_store.embed_data(frame.data),
            data=frame.data,
            metadata={
                "session_id": session_id,
                "type": frame.context_type.name,
                "source": frame.source,
                "timestamp": frame.timestamp.isoformat()
            }
        )
        
        # Update knowledge graph if applicable
        if frame.context_type in [ContextType.USER_PREFERENCE, ContextType.SOCIAL]:
            await self._update_knowledge_graph(session_id, frame)
    
    async def _update_knowledge_graph(
        self,
        session_id: str,
        frame: ContextFrame
    ) -> None:
        """Update knowledge graph with context information."""
        # Extract entities from context
        entities = await self.entity_extractor.extract(str(frame.data))
        
        # Add to knowledge graph
        for entity in entities:
            await self.knowledge_graph.add_entity(
                entity_type=entity["type"],
                entity_id=entity["id"],
                properties={
                    **entity.get("properties", {}),
                    "session_id": session_id,
                    "context_type": frame.context_type.name
                }
            )
    
    def _extract_text_context(self, contexts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract text-related context."""
        text_context = {}
        
        # Conversational context
        if ContextType.CONVERSATIONAL.name in contexts:
            conv_ctx = contexts[ContextType.CONVERSATIONAL.name]
            text_context.update({
                "topic": conv_ctx.get("latest", {}).get("topic"),
                "intent": conv_ctx.get("latest", {}).get("intent"),
                "entities": conv_ctx.get("latest", {}).get("entities", [])
            })
        
        # Task context
        if ContextType.TASK.name in contexts:
            task_ctx = contexts[ContextType.TASK.name]
            text_context["current_task"] = task_ctx.get("latest", {}).get("task_name")
        
        return text_context if text_context else None
    
    def _extract_audio_context(self, contexts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract audio-related context."""
        audio_context = {}
        
        # Emotional context from audio
        if ContextType.EMOTIONAL.name in contexts:
            emo_ctx = contexts[ContextType.EMOTIONAL.name]
            if "audio_emotion" in emo_ctx.get("latest", {}):
                audio_context["emotion"] = emo_ctx["latest"]["audio_emotion"]
        
        # Environmental audio
        if ContextType.ENVIRONMENTAL.name in contexts:
            env_ctx = contexts[ContextType.ENVIRONMENTAL.name]
            if "ambient_sound" in env_ctx.get("latest", {}):
                audio_context["environment"] = env_ctx["latest"]["ambient_sound"]
        
        return audio_context if audio_context else None
    
    def _extract_visual_context(self, contexts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract visual-related context."""
        visual_context = {}
        
        # Spatial context
        if ContextType.SPATIAL.name in contexts:
            spatial_ctx = contexts[ContextType.SPATIAL.name]
            visual_context["location"] = spatial_ctx.get("latest", {}).get("location")
            visual_context["objects"] = spatial_ctx.get("latest", {}).get("detected_objects", [])
        
        # Emotional context from visual
        if ContextType.EMOTIONAL.name in contexts:
            emo_ctx = contexts[ContextType.EMOTIONAL.name]
            if "facial_expression" in emo_ctx.get("latest", {}):
                visual_context["expression"] = emo_ctx["latest"]["facial_expression"]
        
        return visual_context if visual_context else None
    
    async def _infer_temporal_context(
        self,
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Infer temporal context from input."""
        current_time = datetime.now(timezone.utc)
        
        temporal_context = {
            "timestamp": current_time.isoformat(),
            "time_of_day": self._get_time_of_day(current_time),
            "day_of_week": current_time.strftime("%A"),
            "is_weekend": current_time.weekday() >= 5,
            "confidence": 1.0
        }
        
        # Add any time-related information from input
        if "timestamp" in input_data:
            temporal_context["input_timestamp"] = input_data["timestamp"]
        
        return temporal_context
    
    async def _infer_activity_context(
        self,
        session_id: str,
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Infer activity context from input and history."""
        # Get recent contexts
        recent_contexts = await self.get_current_context(
            session_id,
            [ContextType.TASK, ContextType.CONVERSATIONAL]
        )
        
        activity_context = {}
        confidence = 0.5
        
        # Check for explicit activity mentions
        if "text" in input_data:
            activities = self._extract_activities(input_data["text"])
            if activities:
                activity_context["detected_activities"] = activities
                activity_context["primary_activity"] = activities[0]
                confidence = 0.8
        
        # Infer from recent task context
        if ContextType.TASK.name in recent_contexts:
            task_data = recent_contexts[ContextType.TASK.name].get("latest", {})
            if "task_name" in task_data:
                activity_context["related_task"] = task_data["task_name"]
                confidence = max(confidence, 0.7)
        
        if activity_context:
            activity_context["confidence"] = confidence
            return activity_context
        
        return None
    
    async def _infer_emotional_context(
        self,
        session_id: str,
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Infer emotional context from multimodal input."""
        emotional_context = {}
        confidences = []
        
        # Text sentiment analysis
        if "text" in input_data:
            sentiment_result = await self.sentiment_analyzer.analyze(input_data["text"])
            emotional_context["text_sentiment"] = sentiment_result
            confidences.append(sentiment_result.get("confidence", 0.5))
        
        # Audio emotion detection
        if "audio" in input_data:
            audio_emotion = await self.emotion_detector.detect(input_data["audio"])
            emotional_context["audio_emotion"] = audio_emotion
            confidences.append(audio_emotion.get("confidence", 0.5))
        
        # Visual expression analysis
        if "image" in input_data or "video" in input_data:
            visual_data = input_data.get("image") or input_data.get("video")
            expression = await self.expression_analyzer.analyze(visual_data)
            emotional_context["facial_expression"] = expression
            confidences.append(expression.get("confidence", 0.5))
        
        if emotional_context:
            # Aggregate emotions
            emotional_context["overall_emotion"] = self._aggregate_emotions(
                emotional_context
            )
            emotional_context["confidence"] = np.mean(confidences) if confidences else 0.5
            return emotional_context
        
        return None
    
    def _get_time_of_day(self, dt: datetime) -> str:
        """Get time of day category."""
        hour = dt.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _extract_activities(self, text: str) -> List[str]:
        """Extract activities from text."""
        # Simple keyword-based extraction (would be enhanced with NLP)
        activity_keywords = {
            "working": ["work", "coding", "meeting", "project"],
            "exercising": ["exercise", "workout", "running", "gym"],
            "eating": ["eat", "lunch", "dinner", "breakfast", "cooking"],
            "relaxing": ["relax", "rest", "watch", "read", "music"],
            "studying": ["study", "learn", "research", "homework"]
        }
        
        text_lower = text.lower()
        detected = []
        
        for activity, keywords in activity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(activity)
        
        return detected
    
    def _aggregate_emotions(self, emotional_data: Dict[str, Any]) -> str:
        """Aggregate emotions from multiple sources."""
        emotion_scores = defaultdict(float)
        count = 0
        
        # Aggregate from different sources
        if "text_sentiment" in emotional_data:
            sentiment = emotional_data["text_sentiment"].get("sentiment", "neutral")
            emotion_scores[sentiment] += 1
            count += 1
        
        if "audio_emotion" in emotional_data:
            emotion = emotional_data["audio_emotion"].get("emotion", "neutral")
            emotion_scores[emotion] += 1
            count += 1
        
        if "facial_expression" in emotional_data:
            expression = emotional_data["facial_expression"].get("expression", "neutral")
            emotion_scores[expression] += 1
            count += 1
        
        # Return dominant emotion
        if emotion_scores:
            return max(emotion_scores.items(), key=lambda x: x[1])[0]
        return "neutral"
    
    def _match_context_patterns(self, context: Dict[str, Any]) -> List[str]:
        """Match current context against predefined patterns."""
        matched_patterns = []
        
        for pattern_name, pattern_def in self.context_patterns.items():
            if self._matches_pattern(context, pattern_def):
                matched_patterns.append(pattern_name)
        
        return matched_patterns
    
    def _matches_pattern(
        self,
        context: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> bool:
        """Check if context matches a pattern."""
        # Check time range
        if "temporal" in context and "time_range" in pattern:
            hour = datetime.fromisoformat(
                context["temporal"]["timestamp"]
            ).hour
            start, end = pattern["time_range"]
            if not (start <= hour < end):
                return False
        
        # Check activities
        if "activity" in context and "activities" in pattern:
            detected_activities = context["activity"].get("detected_activities", [])
            if not any(act in pattern["activities"] for act in detected_activities):
                return False
        
        # Check mood
        if "emotional" in context and "mood" in pattern:
            overall_emotion = context["emotional"].get("overall_emotion")
            if overall_emotion != pattern["mood"]:
                return False
        
        return True
    
    def _is_relevant_to_query(self, context_data: Dict[str, Any], query: str) -> bool:
        """Check if context is relevant to a query."""
        # Simple keyword matching (would be enhanced with semantic similarity)
        query_lower = query.lower()
        context_str = str(context_data).lower()
        
        # Check for keyword overlap
        query_words = set(query_lower.split())
        context_words = set(context_str.split())
        
        overlap = len(query_words.intersection(context_words))
        return overlap > 0
    
    def _calculate_relevance(self, context_data: Dict[str, Any], query: str) -> float:
        """Calculate relevance score between context and query."""
        # Simple TF-IDF style relevance (would be enhanced)
        query_words = set(query.lower().split())
        context_str = str(context_data).lower()
        
        relevance = 0.0
        for word in query_words:
            if word in context_str:
                relevance += 1.0 / len(query_words)
        
        return min(relevance, 1.0)
    
    def _calculate_recency(self, timestamp: Optional[str]) -> float:
        """Calculate recency score for a timestamp."""
        if not timestamp:
            return 0.0
        
        try:
            ts = datetime.fromisoformat(timestamp)
            age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            # Exponential decay over 24 hours
            return np.exp(-age_hours / 24)
        except:
            return 0.0
    
    def _predict_activities(
        self,
        patterns: Dict[str, Any],
        current_context: Dict[str, Any],
        future_time: datetime
    ) -> List[Dict[str, Any]]:
        """Predict future activities based on patterns."""
        predictions = []
        
        # Extract hour for future time
        future_hour = future_time.hour
        future_day = future_time.strftime("%A")
        
        # Check historical patterns
        if "activity_patterns" in patterns:
            for pattern in patterns["activity_patterns"]:
                if self._pattern_matches_time(pattern, future_hour, future_day):
                    predictions.append({
                        "activity": pattern["activity"],
                        "probability": pattern.get("probability", 0.7),
                        "typical_duration": pattern.get("duration", 60)
                    })
        
        return sorted(predictions, key=lambda x: x["probability"], reverse=True)
    
    def _predict_mood(
        self,
        patterns: Dict[str, Any],
        current_context: Dict[str, Any],
        future_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """Predict future mood based on patterns."""
        # Simplified mood prediction
        time_of_day = self._get_time_of_day(future_time)
        
        mood_patterns = patterns.get("mood_patterns", {})
        if time_of_day in mood_patterns:
            return {
                "predicted_mood": mood_patterns[time_of_day]["typical_mood"],
                "confidence": mood_patterns[time_of_day].get("confidence", 0.6),
                "factors": mood_patterns[time_of_day].get("factors", [])
            }
        
        return None
    
    def _predict_location(
        self,
        patterns: Dict[str, Any],
        current_context: Dict[str, Any],
        future_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """Predict future location based on patterns."""
        # Check location patterns
        location_patterns = patterns.get("location_patterns", [])
        
        future_hour = future_time.hour
        future_day = future_time.strftime("%A")
        
        for pattern in location_patterns:
            if self._pattern_matches_time(pattern, future_hour, future_day):
                return {
                    "predicted_location": pattern["location"],
                    "probability": pattern.get("probability", 0.7),
                    "typical_duration": pattern.get("duration", 120)
                }
        
        return None
    
    def _pattern_matches_time(
        self,
        pattern: Dict[str, Any],
        hour: int,
        day: str
    ) -> bool:
        """Check if a pattern matches the given time."""
        # Check hour range
        if "hour_range" in pattern:
            start, end = pattern["hour_range"]
            if not (start <= hour < end):
                return False
        
        # Check days
        if "days" in pattern:
            if day not in pattern["days"]:
                return False
        
        return True
    
    def _extract_dominant_values(
        self,
        frames: List[ContextFrame]
    ) -> Dict[str, Any]:
        """Extract dominant values from a list of frames."""
        value_counts = defaultdict(lambda: defaultdict(int))
        
        for frame in frames:
            for key, value in frame.data.items():
                if isinstance(value, (str, int, bool)):
                    value_counts[key][str(value)] += 1
        
        dominant = {}
        for key, counts in value_counts.items():
            if counts:
                dominant[key] = max(counts.items(), key=lambda x: x[1])[0]
        
        return dominant
    
    async def _get_user_id(self, session_id: str) -> str:
        """Get user ID for a session."""
        # This would typically query a session store
        # For now, return a placeholder
        return f"user_{session_id[:8]}"
    
    async def _cleanup_expired_contexts(self) -> None:
        """Background task to clean up expired contexts."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                for session_id in list(self.active_contexts.keys()):
                    # Remove expired frames
                    self.active_contexts[session_id] = [
                        frame for frame in self.active_contexts[session_id]
                        if not frame.is_expired()
                    ]
                    
                    # Remove empty sessions
                    if not self.active_contexts[session_id]:
                        del self.active_contexts[session_id]
                        if session_id in self.context_stacks:
                            del self.context_stacks[session_id]
                
                logger.debug("Cleaned up expired contexts")
                
            except Exception as e:
                logger.error(f"Error in context cleanup: {str(e)}")
    
    async def _handle_input_event(self, event: Event) -> None:
        """Handle input events to update context."""
        try:
            session_id = event.data.get("session_id")
            input_data = event.data.get("input")
            
            if session_id and input_data:
                # Infer and update context
                inferred = await self.infer_context(session_id, input_data)
                logger.debug(f"Inferred context from input: {inferred}")
                
        except Exception as e:
            logger.error(f"Error handling input event: {str(e)}")
    
    async def _handle_sensor_event(self, event: Event) -> None:
        """Handle sensor data events."""
        try:
            session_id = event.data.get("session_id")
            sensor_type = event.data.get("sensor_type")
            sensor_data = event.data.get("data")
            
            if session_id and sensor_type and sensor_data:
                # Update environmental context
                await self.update_context(
                    session_id,
                    ContextType.ENVIRONMENTAL,
                    {
                        "sensor_type": sensor_type,
                        "sensor_data": sensor_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    source=f"sensor_{sensor_type}"
                )
                
        except Exception as e:
            logger.error(f"Error handling sensor event: {str(e)}")
    
    async def _handle_memory_event(self, event: Event) -> None:
        """Handle memory update events."""
        try:
            session_id = event.data.get("session_id")
            memory_type = event.data.get("memory_type")
            
            if session_id and memory_type == "user_preference":
                # Update user preference context
                preferences = event.data.get("data", {})
                await self.update_context(
                    session_id,
                    ContextType.USER_PREFERENCE,
                    preferences,
                    source="memory_system",
                    ttl_seconds=3600  # Cache for 1 hour
                )
                
        except Exception as e:
            logger.error(f"Error handling memory event: {str(e)}")
    
    async def _handle_skill_event(self, event: Event) -> None:
        """Handle skill execution events."""
        try:
            session_id = event.data.get("session_id")
            skill_name = event.data.get("skill_name")
            skill_result = event.data.get("result", {})
            
            if session_id and skill_name:
                # Update task context
                await self.update_context(
                    session_id,
                    ContextType.TASK,
                    {
                        "skill_executed": skill_name,
                        "skill_result": skill_result,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    source="skill_system"
                )
                
        except Exception as e:
            logger.error(f"Error handling skill event: {str(e)}")
    
    async def _handle_emotion_event(self, event: Event) -> None:
        """Handle emotion detection events."""
        try:
            session_id = event.data.get("session_id")
            emotion_data = event.data.get("emotion")
            source = event.data.get("source", "unknown")
            
            if session_id and emotion_data:
                # Update emotional context
                await self.update_context(
                    session_id,
                    ContextType.EMOTIONAL,
                    {
                        f"{source}_emotion": emotion_data,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    source=f"emotion_{source}",
                    confidence=emotion_data.get("confidence", 0.7)
                )
                
        except Exception as e:
            logger.error(f"Error handling emotion event: {str(e)}")
