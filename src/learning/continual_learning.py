"""
Advanced Continual Learning System for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 03:00:49 UTC

This module provides comprehensive continual learning capabilities for the AI assistant,
enabling the system to learn from interactions, adapt to user preferences, and improve
performance over time without catastrophic forgetting.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import json
import pickle
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    LearningEventOccurred, ModelAdaptationStarted, ModelAdaptationCompleted,
    LearningMetricsUpdated, FeedbackReceived, UserPreferenceUpdated,
    SystemStateChanged, ComponentHealthChanged, ErrorOccurred,
    ContinualLearningStarted, ContinualLearningCompleted,
    KnowledgeDistillationStarted, KnowledgeDistillationCompleted,
    CatastrophicForgettingDetected, LearningRateAdapted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.encryption import EncryptionManager

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory
from src.memory.storage.vector_store import VectorStore

# Assistant components
from src.assistant.session_manager import SessionManager
from src.assistant.component_manager import ComponentManager
from src.assistant.core_engine import ProcessingResult, MultimodalInput, ProcessingContext

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class LearningMode(Enum):
    """Continual learning modes."""
    ONLINE = "online"                    # Real-time learning from interactions
    BATCH = "batch"                      # Periodic batch learning
    REPLAY = "replay"                    # Experience replay-based learning
    ELASTIC = "elastic"                  # Elastic weight consolidation
    PROGRESSIVE = "progressive"          # Progressive neural networks
    META = "meta"                        # Meta-learning approach
    DISTILLATION = "distillation"        # Knowledge distillation
    REHEARSAL = "rehearsal"              # Rehearsal-based learning
    REGULARIZATION = "regularization"    # Regularization-based methods
    ARCHITECTURE = "architecture"       # Architecture-based methods


class LearningStrategy(Enum):
    """Learning strategy selection."""
    NAIVE = "naive"                      # Simple fine-tuning
    EWC = "ewc"                         # Elastic Weight Consolidation
    L2 = "l2"                           # L2 regularization
    PACKNET = "packnet"                 # PackNet pruning
    PROGRESSIVE_NETS = "progressive_nets" # Progressive Neural Networks
    AGEM = "agem"                       # Averaged Gradient Episodic Memory
    GEM = "gem"                         # Gradient Episodic Memory
    MAS = "mas"                         # Memory Aware Synapses
    ICARL = "icarl"                     # Incremental Classifier and Representation Learning
    LWOE = "lwoe"                       # Learning without Forgetting


class LearningDomain(Enum):
    """Domains for continual learning."""
    LANGUAGE_UNDERSTANDING = "language_understanding"
    SPEECH_PROCESSING = "speech_processing"
    VISION_PROCESSING = "vision_processing"
    MULTIMODAL_FUSION = "multimodal_fusion"
    USER_PREFERENCES = "user_preferences"
    SKILL_EXECUTION = "skill_execution"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    CONTEXT_AWARENESS = "context_awareness"
    REASONING = "reasoning"
    DECISION_MAKING = "decision_making"


class ForgettingMeasure(Enum):
    """Methods to measure catastrophic forgetting."""
    ACCURACY_DROP = "accuracy_drop"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    KNOWLEDGE_RETENTION = "knowledge_retention"
    SIMILARITY_METRICS = "similarity_metrics"
    ACTIVATION_PATTERNS = "activation_patterns"


@dataclass
class LearningConfiguration:
    """Configuration for continual learning."""
    # Core settings
    learning_mode: LearningMode = LearningMode.ONLINE
    learning_strategy: LearningStrategy = LearningStrategy.EWC
    learning_domains: Set[LearningDomain] = field(default_factory=lambda: {LearningDomain.LANGUAGE_UNDERSTANDING})
    
    # Learning parameters
    base_learning_rate: float = 0.001
    adaptive_learning_rate: bool = True
    max_learning_rate: float = 0.01
    min_learning_rate: float = 0.0001
    learning_rate_decay: float = 0.95
    
    # Memory management
    replay_buffer_size: int = 10000
    episodic_memory_size: int = 50000
    working_memory_window: int = 100
    long_term_memory_threshold: float = 0.8
    
    # Forgetting prevention
    ewc_lambda: float = 400.0
    mas_lambda: float = 1.0
    l2_lambda: float = 0.01
    knowledge_distillation_alpha: float = 0.7
    temperature: float = 4.0
    
    # Update scheduling
    batch_size: int = 32
    update_frequency: int = 100
    consolidation_interval: int = 3600
    evaluation_interval: int = 300
    
    # Quality control
    min_confidence_threshold: float = 0.6
    performance_drop_threshold: float = 0.1
    catastrophic_forgetting_threshold: float = 0.2
    min_samples_for_update: int = 10
    
    # Model management
    model_versioning: bool = True
    checkpoint_interval: int = 1000
    max_model_versions: int = 10
    auto_rollback: bool = True
    
    # Security and privacy
    differential_privacy: bool = False
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.1
    federated_learning: bool = False
    
    # Resource management
    max_memory_usage_mb: float = 1024.0
    max_cpu_usage_percent: float = 50.0
    gpu_enabled: bool = True
    parallel_processing: bool = True


@dataclass
class LearningExperience:
    """Represents a learning experience."""
    experience_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Input data
    input_data: Dict[str, Any] = field(default_factory=dict)
    multimodal_input: Optional[MultimodalInput] = None
    processing_context: Optional[ProcessingContext] = None
    
    # Output data
    predicted_output: Dict[str, Any] = field(default_factory=dict)
    actual_output: Dict[str, Any] = field(default_factory=dict)
    processing_result: Optional[ProcessingResult] = None
    
    # Learning metadata
    domain: LearningDomain = LearningDomain.LANGUAGE_UNDERSTANDING
    confidence_score: float = 0.0
    feedback_score: Optional[float] = None
    success: bool = True
    
    # Context
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    environmental_context: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    response_time: float = 0.0
    accuracy: float = 0.0
    relevance: float = 0.0
    user_satisfaction: Optional[float] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None
    learned_at: Optional[datetime] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    importance_weight: float = 1.0
    priority: int = 1


@dataclass
class LearningUpdate:
    """Represents a learning update."""
    update_id: str
    experience_ids: List[str]
    domain: LearningDomain
    strategy: LearningStrategy
    
    # Update parameters
    learning_rate: float
    batch_size: int
    gradient_norm: float = 0.0
    loss_value: float = 0.0
    
    # Performance metrics
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Dict[str, float] = field(default_factory=dict)
    
    # Forgetting analysis
    forgetting_metrics: Dict[str, float] = field(default_factory=dict)
    retention_score: float = 1.0
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    
    # Timing
    update_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LearningError(Exception):
    """Custom exception for continual learning operations."""
    
    def __init__(self, message: str, domain: Optional[LearningDomain] = None,
                 error_code: Optional[str] = None, experience_id: Optional[str] = None):
        super().__init__(message)
        self.domain = domain
        self.error_code = error_code
        self.experience_id = experience_id
        self.timestamp = datetime.now(timezone.utc)


class ReplayBuffer:
    """Experience replay buffer for continual learning."""
    
    def __init__(self, capacity: int, logger):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.domain_buffers: Dict[LearningDomain, deque] = defaultdict(lambda: deque(maxlen=capacity // 10))
        self.importance_weights: deque = deque(maxlen=capacity)
        self.logger = logger
        self._lock = threading.Lock()
    
    def add_experience(self, experience: LearningExperience) -> None:
        """Add experience to replay buffer."""
        with self._lock:
            self.buffer.append(experience)
            self.domain_buffers[experience.domain].append(experience)
            self.importance_weights.append(experience.importance_weight)
    
    def sample_batch(self, batch_size: int, 
                    domain: Optional[LearningDomain] = None,
                    strategy: str = "uniform") -> List[LearningExperience]:
        """Sample a batch of experiences."""
        with self._lock:
            if domain:
                source_buffer = list(self.domain_buffers[domain])
            else:
                source_buffer = list(self.buffer)
            
            if len(source_buffer) == 0:
                return []
            
            if strategy == "uniform":
                indices = np.random.choice(len(source_buffer), 
                                         min(batch_size, len(source_buffer)), 
                                         replace=False)
            elif strategy == "importance":
                weights = np.array([exp.importance_weight for exp in source_buffer])
                weights = weights / weights.sum()
                indices = np.random.choice(len(source_buffer), 
                                         min(batch_size, len(source_buffer)), 
                                         replace=False, p=weights)
            else:
                indices = list(range(min(batch_size, len(source_buffer))))
            
            return [source_buffer[i] for i in indices]
    
    def get_domain_distribution(self) -> Dict[LearningDomain, int]:
        """Get distribution of experiences by domain."""
        with self._lock:
            return {domain: len(buffer) for domain, buffer in self.domain_buffers.items()}
    
    def clear_domain(self, domain: LearningDomain) -> None:
        """Clear experiences from a specific domain."""
        with self._lock:
            self.domain_buffers[domain].clear()


class ForgettingDetector:
    """Detects and measures catastrophic forgetting."""
    
    def __init__(self, logger, config: LearningConfiguration):
        self.logger = logger
        self.config = config
        self.baseline_metrics: Dict[LearningDomain, Dict[str, float]] = {}
        self.performance_history: Dict[LearningDomain, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def set_baseline(self, domain: LearningDomain, metrics: Dict[str, float]) -> None:
        """Set baseline performance metrics for a domain."""
        self.baseline_metrics[domain] = metrics.copy()
        self.logger.info(f"Set baseline for {domain}: {metrics}")
    
    def detect_forgetting(self, domain: LearningDomain, 
                         current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Detect catastrophic forgetting in a domain."""
        if domain not in self.baseline_metrics:
            return {"forgetting_detected": False, "reason": "No baseline set"}
        
        baseline = self.baseline_metrics[domain]
        forgetting_scores = {}
        significant_drops = []
        
        for metric, current_value in current_metrics.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if baseline_value > 0:
                    drop_ratio = (baseline_value - current_value) / baseline_value
                    forgetting_scores[metric] = drop_ratio
                    
                    if drop_ratio > self.config.catastrophic_forgetting_threshold:
                        significant_drops.append({
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "drop_ratio": drop_ratio
                        })
        
        # Update performance history
        self.performance_history[domain].append({
            "timestamp": datetime.now(timezone.utc),
            "metrics": current_metrics.copy(),
            "forgetting_scores": forgetting_scores
        })
        
        forgetting_detected = len(significant_drops) > 0
        overall_forgetting_score = np.mean(list(forgetting_scores.values())) if forgetting_scores else 0.0
        
        return {
            "forgetting_detected": forgetting_detected,
            "overall_forgetting_score": overall_forgetting_score,
            "forgetting_scores": forgetting_scores,
            "significant_drops": significant_drops,
            "affected_metrics": len(significant_drops),
            "domain": domain
        }
    
    def get_forgetting_trend(self, domain: LearningDomain) -> Dict[str, Any]:
        """Get forgetting trend for a domain over time."""
        if domain not in self.performance_history:
            return {"trend": "no_data"}
        
        history = list(self.performance_history[domain])
        if len(history) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend over recent history
        recent_scores = [entry.get("forgetting_scores", {}) for entry in history[-10:]]
        if not recent_scores:
            return {"trend": "no_forgetting_data"}
        
        # Analyze trend
        trend_analysis = {
            "trend": "stable",
            "direction": "neutral",
            "severity": "low",
            "recent_average": 0.0
        }
        
        # Implementation would analyze the trend
        return trend_analysis


class KnowledgeDistillator:
    """Performs knowledge distillation for continual learning."""
    
    def __init__(self, logger, config: LearningConfiguration):
        self.logger = logger
        self.config = config
        self.teacher_models: Dict[LearningDomain, Any] = {}
        self.student_models: Dict[LearningDomain, Any] = {}
    
    async def distill_knowledge(self, domain: LearningDomain,
                               teacher_model: Any, student_model: Any,
                               experiences: List[LearningExperience]) -> Dict[str, float]:
        """Perform knowledge distillation."""
        try:
            distillation_metrics = {
                "distillation_loss": 0.0,
                "knowledge_retention": 0.0,
                "compression_ratio": 0.0,
                "transfer_efficiency": 0.0
            }
            
            # Simplified knowledge distillation process
            for batch_start in range(0, len(experiences), self.config.batch_size):
                batch = experiences[batch_start:batch_start + self.config.batch_size]
                
                # Extract inputs from batch
                batch_inputs = [exp.input_data for exp in batch]
                
                # Get teacher predictions (would be actual model inference)
                teacher_outputs = await self._get_teacher_outputs(teacher_model, batch_inputs)
                
                # Get student predictions
                student_outputs = await self._get_student_outputs(student_model, batch_inputs)
                
                # Calculate distillation loss (simplified)
                batch_loss = self._calculate_distillation_loss(teacher_outputs, student_outputs)
                distillation_metrics["distillation_loss"] += batch_loss
            
            # Normalize metrics
            if len(experiences) > 0:
                distillation_metrics["distillation_loss"] /= len(experiences)
            
            return distillation_metrics
            
        except Exception as e:
            self.logger.error(f"Knowledge distillation failed for {domain}: {str(e)}")
            raise LearningError(f"Knowledge distillation failed: {str(e)}", domain)
    
    async def _get_teacher_outputs(self, teacher_model: Any, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get outputs from teacher model."""
        # Placeholder implementation
        return [{"prediction": 0.8, "confidence": 0.9} for _ in inputs]
    
    async def _get_student_outputs(self, student_model: Any, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get outputs from student model."""
        # Placeholder implementation
        return [{"prediction": 0.7, "confidence": 0.8} for _ in inputs]
    
    def _calculate_distillation_loss(self, teacher_outputs: List[Dict[str, Any]], 
                                   student_outputs: List[Dict[str, Any]]) -> float:
        """Calculate knowledge distillation loss."""
        # Simplified distillation loss calculation
        if len(teacher_outputs) != len(student_outputs):
            return float('inf')
        
        total_loss = 0.0
        for teacher_out, student_out in zip(teacher_outputs, student_outputs):
            # Simplified KL divergence
            teacher_pred = teacher_out.get("prediction", 0.0)
            student_pred = student_out.get("prediction", 0.0)
            loss = abs(teacher_pred - student_pred)
            total_loss += loss
        
        return total_loss / len(teacher_outputs) if teacher_outputs else 0.0


class ContinualLearner:
    """
    Advanced Continual Learning System for the AI Assistant.
    
    This system enables the AI assistant to continuously learn and adapt from
    user interactions while preventing catastrophic forgetting. It integrates
    with all core system components to provide seamless learning capabilities.
    
    Features:
    - Multiple continual learning strategies (EWC, Progressive Networks, etc.)
    - Catastrophic forgetting detection and prevention
    - Multi-domain learning with domain-specific adaptations
    - Experience replay and rehearsal mechanisms
    - Knowledge distillation for model compression
    - Adaptive learning rate scheduling
    - Performance monitoring and quality control
    - Integration with memory systems and user preferences
    - Real-time and batch learning modes
    - Security and privacy-aware learning
    """
    
    def __init__(self, container: Container):
        """
        Initialize the continual learner.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        self.vector_store = container.get(VectorStore)
        
        # Assistant components
        self.session_manager = container.get(SessionManager)
        self.component_manager = container.get(ComponentManager)
        
        # Processing components
        self.intent_manager = container.get(IntentManager)
        self.language_chain = container.get(LanguageChain)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)
        self.model_router = container.get(ModelRouter)
        
        # Storage and caching
        self.database = container.get(DatabaseManager)
        self.cache_strategy = container.get(CacheStrategy)
        
        # Security (optional)
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.encryption_manager = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Configuration
        self.config = LearningConfiguration()
        self._load_configuration()
        
        # Learning infrastructure
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.logger)
        self.forgetting_detector = ForgettingDetector(self.logger, self.config)
        self.knowledge_distillator = KnowledgeDistillator(self.logger, self.config)
        
        # State management
        self.domain_models: Dict[LearningDomain, Any] = {}
        self.learning_statistics: Dict[LearningDomain, Dict[str, Any]] = defaultdict(dict)
        self.active_learning_tasks: Dict[str, asyncio.Task] = {}
        self.learning_locks: Dict[LearningDomain, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Performance tracking
        self.update_history: deque = deque(maxlen=1000)
        self.experience_buffer: deque = deque(maxlen=10000)
        self.learning_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="continual_learning")
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("continual_learner", self._health_check_callback)
        
        self.logger.info("ContinualLearner initialized successfully")

    def _load_configuration(self) -> None:
        """Load continual learning configuration."""
        try:
            learning_config = self.config_loader.get("learning", {})
            
            # Update configuration from loaded settings
            if "continual_learning_enabled" in learning_config:
                self.enabled = learning_config["continual_learning_enabled"]
            else:
                self.enabled = True
            
            # Learning parameters
            self.config.base_learning_rate = learning_config.get("learning_rate", 0.001)
            self.config.batch_size = learning_config.get("batch_size", 32)
            self.config.update_frequency = learning_config.get("update_frequency", 100)
            
            # Memory settings
            self.config.replay_buffer_size = learning_config.get("replay_buffer_size", 10000)
            self.config.episodic_memory_size = learning_config.get("episodic_memory_size", 50000)
            
            # Quality control
            self.config.min_confidence_threshold = learning_config.get("min_confidence_threshold", 0.6)
            self.config.performance_drop_threshold = learning_config.get("performance_drop_threshold", 0.1)
            
            self.logger.info("Continual learning configuration loaded")
            
        except Exception as e:
            self.logger.warning(f"Failed to load learning configuration: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register learning metrics
            self.metrics.register_counter("learning_experiences_total")
            self.metrics.register_counter("learning_updates_total")
            self.metrics.register_counter("learning_updates_successful")
            self.metrics.register_counter("learning_updates_failed")
            self.metrics.register_histogram("learning_update_duration_seconds")
            self.metrics.register_histogram("experience_processing_time_seconds")
            self.metrics.register_gauge("replay_buffer_size")
            self.metrics.register_gauge("active_learning_tasks")
            self.metrics.register_counter("catastrophic_forgetting_detected")
            self.metrics.register_histogram("knowledge_distillation_loss")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the continual learner."""
        if not self.enabled:
            self.logger.info("Continual learning is disabled")
            return
        
        try:
            # Initialize domain models
            await self._initialize_domain_models()
            
            # Load existing learning state
            await self._load_learning_state()
            
            # Start background tasks
            asyncio.create_task(self._learning_update_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._forgetting_detection_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("ContinualLearner initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ContinualLearner: {str(e)}")
            raise LearningError(f"Initialization failed: {str(e)}")

    async def _initialize_domain_models(self) -> None:
        """Initialize models for each learning domain."""
        for domain in self.config.learning_domains:
            try:
                # Initialize domain-specific models
                # This would load or create models for each domain
                self.domain_models[domain] = await self._create_domain_model(domain)
                self.learning_statistics[domain] = {
                    "total_experiences": 0,
                    "successful_updates": 0,
                    "failed_updates": 0,
                    "last_update": None,
                    "performance_metrics": {}
                }
                
                self.logger.debug(f"Initialized model for domain: {domain}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize model for domain {domain}: {str(e)}")

    async def _create_domain_model(self, domain: LearningDomain) -> Any:
        """Create a model for a specific learning domain."""
        # Placeholder implementation - would create actual models
        return {
            "domain": domain,
            "model_type": "transformer",
            "parameters": {},
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc)
        }

    async def _load_learning_state(self) -> None:
        """Load existing learning state from storage."""
        try:
            # Load replay buffer
            if self.database:
                # Load experiences from database
                experiences = await self._load_experiences_from_database()
                for experience in experiences:
                    self.replay_buffer.add_experience(experience)
            
            # Load baseline metrics for forgetting detection
            baseline_metrics = await self._load_baseline_metrics()
            for domain, metrics in baseline_metrics.items():
                self.forgetting_detector.set_baseline(domain, metrics)
            
            self.logger.info("Learning state loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load learning state: {str(e)}")

    async def _load_experiences_from_database(self) -> List[LearningExperience]:
        """Load learning experiences from database."""
        experiences = []
        try:
            # Simplified database query
            results = await self.database.fetch_all(
                "SELECT * FROM learning_experiences ORDER BY created_at DESC LIMIT ?",
                (self.config.replay_buffer_size,)
            )
            
            for row in results:
                # Convert database row to LearningExperience
                experience = self._row_to_experience(row)
                experiences.append(experience)
            
        except Exception as e:
            self.logger.warning(f"Failed to load experiences from database: {str(e)}")
        
        return experiences

    def _row_to_experience(self, row: Dict[str, Any]) -> LearningExperience:
        """Convert database row to LearningExperience object."""
        return LearningExperience(
            experience_id=row.get("experience_id", str(uuid.uuid4())),
            session_id=row.get("session_id", ""),
            user_id=row.get("user_id"),
            input_data=json.loads(row.get("input_data", "{}")),
            predicted_output=json.loads(row.get("predicted_output", "{}")),
            actual_output=json.loads(row.get("actual_output", "{}")),
            domain=LearningDomain(row.get("domain", "language_understanding")),
            confidence_score=row.get("confidence_score", 0.0),
            success=row.get("success", True),
            created_at=datetime.fromisoformat(row.get("created_at", datetime.now(timezone.utc).isoformat()))
        )

    async def _load_baseline_metrics(self) -> Dict[LearningDomain, Dict[str, float]]:
        """Load baseline performance metrics."""
        baseline_metrics = {}
        try:
            if self.database:
                results = await self.database.fetch_all(
                    "SELECT domain, metrics FROM baseline_performance_metrics"
                )
                
                for row in results:
                    domain = LearningDomain(row["domain"])
                    metrics = json.loads(row["metrics"])
                    baseline_metrics[domain] = metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to load baseline metrics: {str(e)}")
        
        return baseline_metrics

    async def _register_event_handlers(self) -> None:
        """Register event handlers for learning events."""
        # Feedback events
        self.event_bus.subscribe("feedback_received", self._handle_feedback_event)
        
        # User preference events
        self.event_bus.subscribe("user_preference_updated", self._handle_preference_event)
        
        # System events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        self.event_bus.subscribe("system_state_changed", self._handle_system_state_change)

    @handle_exceptions
    async def learn_from_interaction(
        self,
        input_data: MultimodalInput,
        processing_result: ProcessingResult,
        context: ProcessingContext,
        feedback_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Learn from a user interaction.
        
        Args:
            input_data: Multimodal input data
            processing_result: Processing result from the interaction
            context: Processing context
            feedback_data: Optional feedback data
            
        Returns:
            Experience ID
        """
        if not self.enabled:
            return ""
        
        start_time = time.time()
        
        try:
            with self.tracer.trace("learning_from_interaction") as span:
                span.set_attributes({
                    "session_id": context.session_id,
                    "user_id": context.user_id or "anonymous",
                    "request_id": context.request_id
                })
                
                # Create learning experience
                experience = await self._create_learning_experience(
                    input_data, processing_result, context, feedback_data
                )
                
                # Add to experience buffer and replay buffer
                self.experience_buffer.append(experience)
                self.replay_buffer.add_experience(experience)
                
                # Store in persistent storage
                if self.database:
                    await self._store_experience(experience)
                
                # Trigger learning update if conditions are met
                if self._should_trigger_update(experience.domain):
                    task_id = f"learning_update_{experience.domain.value}_{int(time.time())}"
                    self.active_learning_tasks[task_id] = asyncio.create_task(
                        self._perform_learning_update(experience.domain, task_id)
                    )
                
                # Update metrics
                self.metrics.increment("learning_experiences_total")
                self.metrics.record("experience_processing_time_seconds", time.time() - start_time)
                self.metrics.set("replay_buffer_size", len(self.replay_buffer.buffer))
                
                # Emit learning event
                await self.event_bus.emit(LearningEventOccurred(
                    event_type="interaction_learning",
                    data={
                        "experience_id": experience.experience_id,
                        "domain": experience.domain.value,
                        "confidence": experience.confidence_score,
                        "success": experience.success
                    }
                ))
                
                self.logger.debug(f"Created learning experience: {experience.experience_id}")
                return experience.experience_id
                
        except Exception as e:
            self.logger.error(f"Failed to learn from interaction: {str(e)}")
            raise LearningError(f"Learning from interaction failed: {str(e)}")

    async def _create_learning_experience(
        self,
        input_data: MultimodalInput,
        processing_result: ProcessingResult,
        context: ProcessingContext,
        feedback_data: Optional[Dict[str, Any]]
    ) -> LearningExperience:
        """Create a learning experience from interaction data."""
        experience_id = str(uuid.uuid4())
        
        # Determine learning domain
        domain = await self._determine_learning_domain(input_data, processing_result, context)
        
        # Extract relevant data
        input_dict = {
            "text": input_data.text,
            "modalities": [k for k, v in asdict(input_data).items() if v is not None and k != 'context'],
            "context_data": context.__dict__ if context else {}
        }
        
        predicted_output = {
            "response_text": processing_result.response_text,
            "confidence": processing_result.overall_confidence,
            "modality_confidences": processing_result.modality_confidences,
            "processing_time": processing_result.processing_time
        }
        
        # Calculate confidence score
        confidence_score = processing_result.overall_confidence
        
        # Determine success based on various factors
        success = (
            processing_result.success and
            confidence_score >= self.config.min_confidence_threshold and
            not processing_result.errors
        )
        
        # Extract feedback score if available
        feedback_score = None
        if feedback_data:
            feedback_score = feedback_data.get("rating", feedback_data.get("score"))
        
        experience = LearningExperience(
            experience_id=experience_id,
            session_id=context.session_id,
            user_id=context.user_id,
            input_data=input_dict,
            predicted_output=predicted_output,
            multimodal_input=input_data,
            processing_result=processing_result,
            processing_context=context,
            domain=domain,
            confidence_score=confidence_score,
            success=success,
            feedback_score=feedback_score,
            response_time=processing_result.processing_time,
            accuracy=confidence_score,  # Simplified
            processed_at=datetime.now(timezone.utc)
        )
        
        return experience

    async def _determine_learning_domain(
        self,
        input_data: MultimodalInput,
        processing_result: ProcessingResult,
        context: ProcessingContext
    ) -> LearningDomain:
        """Determine the primary learning domain for an experience."""
        # Analyze input modalities
        has_text = input_data.text is not None
        has_audio = input_data.audio is not None
        has_image = input_data.image is not None
        has_video = input_data.video is not None
        
        # Count modalities
        modality_count = sum([has_text, has_audio, has_image, has_video])
        
        if modality_count > 1:
            return LearningDomain.MULTIMODAL_FUSION
        elif has_audio:
            return LearningDomain.SPEECH_PROCESSING
        elif has_image or has_video:
            return LearningDomain.VISION_PROCESSING
        elif has_text:
            return LearningDomain.LANGUAGE_UNDERSTANDING
        else:
            return LearningDomain.LANGUAGE_UNDERSTANDING  # Default

    async def _store_experience(self, experience: LearningExperience) -> None:
        """Store learning experience in persistent storage."""
        try:
            await self.database.execute(
                """
                INSERT INTO learning_experiences (
                    experience_id, session_id, user_id, input_data, predicted_output,
                    actual_output, domain, confidence_score, feedback_score, success,
                    response_time, accuracy, created_at, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.experience_id,
                    experience.session_id,
                    experience.user_id,
                    json.dumps(experience.input_data),
                    json.dumps(experience.predicted_output),
                    json.dumps(experience.actual_output),
                    experience.domain.value,
                    experience.confidence_score,
                    experience.feedback_score,
                    experience.success,
                    experience.response_time,
                    experience.accuracy,
                    experience.created_at.isoformat(),
                    experience.processed_at.isoformat() if experience.processed_at else None
                )
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store experience {experience.experience_id}: {str(e)}")

    def _should_trigger_update(self, domain: LearningDomain) -> bool:
        """Determine if a learning update should be triggered."""
        if self.config.learning_mode == LearningMode.ONLINE:
            # Check if enough experiences have accumulated
            domain_experiences = len(self.replay_buffer.domain_buffers[domain])
            return domain_experiences >= self.config.min_samples_for_update
        
        elif self.config.learning_mode == LearningMode.BATCH:
            # Check update frequency
            stats = self.learning_statistics[domain]
            last_update = stats.get("last_update")
            
            if last_update is None:
                return True
            
            time_since_update = (datetime.now(timezone.utc) - last_update).total_seconds()
            return time_since_update >= self.config.consolidation_interval
        
        return False

    async def _perform_learning_update(self, domain: LearningDomain, task_id: str) -> None:
        """Perform a learning update for a specific domain."""
        async with self.learning_locks[domain]:
            start_time = time.time()
            
            try:
                # Emit update started event
                await self.event_bus.emit(ModelAdaptationStarted(
                    model_id=f"{domain.value}_model",
                    adaptation_type="continual_learning"
                ))
                
                # Sample experiences for update
                experiences = self.replay_buffer.sample_batch(
                    self.config.batch_size,
                    domain=domain,
                    strategy="importance"
                )
                
                if not experiences:
                    self.logger.warning(f"No experiences available for domain {domain}")
                    return
                
                # Get current performance baseline
                baseline_metrics = await self._evaluate_current_performance(domain)
                
                # Perform the learning update
                update_result = await self._execute_learning_update(domain, experiences, baseline_metrics)
                
                # Evaluate post-update performance
                post_update_metrics = await self._evaluate_current_performance(domain)
                
                # Check for catastrophic forgetting
                forgetting_analysis = self.forgetting_detector.detect_forgetting(
                    domain, post_update_metrics
                )
                
                if forgetting_analysis["forgetting_detected"]:
                    await self._handle_catastrophic_forgetting(domain, forgetting_analysis, update_result)
                
                # Update statistics
                self.learning_statistics[domain].update({
                    "last_update": datetime.now(timezone.utc),
                    "successful_updates": self.learning_statistics[domain].get("successful_updates", 0) + 1,
                    "performance_metrics": post_update_metrics
                })
                
                # Store update result
                self.update_history.append(update_result)
                
                # Update metrics
                update_time = time.time() - start_time
                self.metrics.increment("learning_updates_total")
                self.metrics.increment("learning_updates_successful")
                self.metrics.record("learning_update_duration_seconds", update_time)
                
                # Emit completion event
                await self.event_bus.emit(ModelAdaptationCompleted(
                    model_id=f"{domain.value}_model",
                    adaptation_success=update_result.success,
                    adaptation_time=update_time
                ))
                
                self.logger.info(f"Learning update completed for {domain} in {update_time:.2f}s")
                
            except Exception as e:
                # Handle update failure
                self.learning_statistics[domain]["failed_updates"] = (
                    self.learning_statistics[domain].get("failed_updates", 0) + 1
                )
                
                self.metrics.increment("learning_updates_failed")
                
                self.logger.error(f"Learning update failed for {domain}: {str(e)}")
                
                # Create failed update result
                failed_update = LearningUpdate(
                    update_id=str(uuid.uuid4()),
                    experience_ids=[],
                    domain=domain,
                    strategy=self.config.learning_strategy,
                    learning_rate=self.config.base_learning_rate,
                    batch_size=0,
                    success=False,
                    error_message=str(e),
                    update_time=time.time() - start_time
                )
                
                self.update_history.append(failed_update)
                
                # Emit failure event
                await self.event_bus.emit(ModelAdaptationCompleted(
                    model_id=f"{domain.value}_model",
                    adaptation_success=False,
                    adaptation_time=time.time() - start_time
                ))
                
            finally:
                # Remove from active tasks
                self.active_learning_tasks.pop(task_id, None)
                self.metrics.set("active_learning_tasks", len(self.active_learning_tasks))

    async def _evaluate_current_performance(self, domain: LearningDomain) -> Dict[str, float]:
        """Evaluate current performance for a domain."""
        # Placeholder implementation - would evaluate actual models
        performance_metrics = {
            "accuracy": 0.85 + np.random.normal(0, 0.05),
            "precision": 0.82 + np.random.normal(0, 0.05),
            "recall": 0.88 + np.random.normal(0, 0.05),
            "f1_score": 0.84 + np.random.normal(0, 0.05),
            "response_time": 0.5 + np.random.normal(0, 0.1),
            "confidence": 0.9 + np.random.normal(0, 0.03)
        }
        
        # Ensure values are within valid ranges
        for key, value in performance_metrics.items():
            if key != "response_time":
                performance_metrics[key] = max(0.0, min(1.0, value))
            else:
                performance_metrics[key] = max(0.1, value)
        
        return performance_metrics

    async def _execute_learning_update(
        self,
        domain: LearningDomain,
        experiences: List[LearningExperience],
        baseline_metrics: Dict[str, float]
    ) -> LearningUpdate:
        """Execute the actual learning update."""
        update_id = str(uuid.uuid4())
        experience_ids = [exp.experience_id for exp in experiences]
        
        # Determine learning rate (adaptive)
        learning_rate = await self._calculate_adaptive_learning_rate(domain, experiences)
        
        # Simulate learning update (in practice, this would update actual models)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Calculate mock gradients and loss
        gradient_norm = np.random.uniform(0.1, 2.0)
        loss_value = np.random.uniform(0.1, 1.0)
        
        # Simulate post-update metrics
        improvement = {
            metric: np.random.normal(0.01, 0.005) for metric in baseline_metrics.keys()
        }
        
        after_metrics = {
            metric: baseline_metrics[metric] + improvement[metric]
            for metric in baseline_metrics.keys()
        }
        
        # Ensure metrics are within valid ranges
        for key, value in after_metrics.items():
            if key != "response_time":
                after_metrics[key] = max(0.0, min(1.0, value))
            else:
                after_metrics[key] = max(0.1, value)
        
        update_result = LearningUpdate(
            update_id=update_id,
            experience_ids=experience_ids,
            domain=domain,
            strategy=self.config.learning_strategy,
            learning_rate=learning_rate,
            batch_size=len(experiences),
            gradient_norm=gradient_norm,
            loss_value=loss_value,
            before_metrics=baseline_metrics,
            after_metrics=after_metrics,
            improvement=improvement,
            success=True
        )
        
        return update_result

    async def _calculate_adaptive_learning_rate(
        self,
        domain: LearningDomain,
        experiences: List[LearningExperience]
    ) -> float:
        """Calculate adaptive learning rate based on experiences and performance."""
        base_rate = self.config.base_learning_rate
        
        if not self.config.adaptive_learning_rate:
            return base_rate
        
        # Factors that influence learning rate
        factors = {
            "confidence": np.mean([exp.confidence_score for exp in experiences]),
            "success_rate": np.mean([exp.success for exp in experiences]),
            "feedback_quality": np.mean([
                exp.feedback_score for exp in experiences 
                if exp.feedback_score is not None
            ]) if any(exp.feedback_score is not None for exp in experiences) else 0.8
        }
        
        # Calculate adaptation multiplier
        adaptation_multiplier = 1.0
        
        # Lower learning rate for high-confidence, successful experiences
        if factors["confidence"] > 0.8 and factors["success_rate"] > 0.8:
            adaptation_multiplier *= 0.5
        
        # Increase learning rate for low-confidence experiences
        elif factors["confidence"] < 0.6:
            adaptation_multiplier *= 1.5
        
        # Consider feedback quality
        if factors["feedback_quality"] < 0.5:
            adaptation_multiplier *= 0.8
        
        adapted_rate = base_rate * adaptation_multiplier
        
        # Ensure within bounds
        adapted_rate = max(self.config.min_learning_rate, 
                          min(self.config.max_learning_rate, adapted_rate))
        
        return adapted_rate

    async def _handle_catastrophic_forgetting(
        self,
        domain: LearningDomain,
        forgetting_analysis: Dict[str, Any],
        update_result: LearningUpdate
    ) -> None:
        """Handle detected catastrophic forgetting."""
        self.logger.warning(f"Catastrophic forgetting detected in {domain}: {forgetting_analysis}")
        
        # Emit forgetting detection event
        await self.event_bus.emit(CatastrophicForgettingDetected(
            domain=domain.value,
            forgetting_score=forgetting_analysis["overall_forgetting_score"],
            affected_metrics=forgetting_analysis["affected_metrics"]
        ))
        
        # Increment forgetting counter
        self.metrics.increment("catastrophic_forgetting_detected")
        
        # Apply mitigation strategies based on configuration
        if self.config.learning_strategy == LearningStrategy.EWC:
            await self._apply_ewc_regularization(domain, forgetting_analysis)
        
        elif self.config.learning_strategy == LearningStrategy.REPLAY:
            await self._increase_replay_importance(domain, forgetting_analysis)
        
        elif self.config.auto_rollback:
            await self._rollback_model_update(domain, update_result)
        
        # Update forgetting-specific metrics
        self.learning_statistics[domain]["catastrophic_forgetting_events"] = (
            self.learning_statistics[domain].get("catastrophic_forgetting_events", 0) + 1
        )

    async def _apply_ewc_regularization(self, domain: LearningDomain, forgetting_analysis: Dict[str, Any]) -> None:
        """Apply Elastic Weight Consolidation regularization."""
        self.logger.info(f"Applying EWC regularization for {domain}")
        # Placeholder implementation
        # In practice, this would adjust model parameters with EWC constraints

    async def _increase_replay_importance(self, domain: LearningDomain, forgetting_analysis: Dict[str, Any]) -> None:
        """Increase importance of replay samples for affected metrics."""
        self.logger.info(f"Increasing replay importance for {domain}")
        # Placeholder implementation
        # In practice, this would adjust replay buffer sampling weights

    async def _rollback_model_update(self, domain: LearningDomain, update_result: LearningUpdate) -> None:
        """Rollback model to previous state."""
        self.logger.warning(f"Rolling back model update for {domain}")
        # Placeholder implementation
        # In practice, this would restore model from checkpoint

    async def learn_from_workflow_execution(self, workflow_data: Dict[str, Any]) -> None:
        """Learn from workflow execution data."""
        if not self.enabled:
            return
        
        try:
            # Extract learning signals from workflow execution
            experience = await self._create_workflow_experience(workflow_data)
            
            if experience:
                self.experience_buffer.append(experience)
                self.replay_buffer.add_experience(experience)
                
                if self.database:
                    await self._store_experience(experience)
                
                self.logger.debug(f"Created workflow learning experience: {experience.experience_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to learn from workflow execution: {str(e)}")

    async def _create_workflow_experience(self, workflow_data: Dict[str, Any]) -> Optional[LearningExperience]:
        """Create learning experience from workflow execution data."""
        if not workflow_data.get("success", False):
            return None
        
        experience_id = str(uuid.uuid4())
        
        # Extract relevant workflow information
        input_data = {
            "workflow_id": workflow_data.get("workflow_id"),
            "execution_time": workflow_data.get("execution_time", 0.0),
            "steps_completed": workflow_data.get("completed_steps", 0),
            "context": workflow_data.get("context", {})
        }
        
        predicted_output = {
            "expected_duration": workflow_data.get("expected_duration", 0.0),
            "success_probability": workflow_data.get("success_probability", 0.5)
        }
        
        actual_output = {
            "actual_duration": workflow_data.get("execution_time", 0.0),
            "actual_success": workflow_data.get("success", False)
        }
        
        # Calculate confidence based on execution success and timing
        confidence_score = 0.8 if workflow_data.get("success", False) else 0.3
        
        experience = LearningExperience(
            experience_id=experience_id,
            session_id=workflow_data.get("session_id", ""),
            user_id=workflow_data.get("user_id"),
            input_data=input_data,
            predicted_output=predicted_output,
            actual_output=actual_output,
            domain=LearningDomain.WORKFLOW_OPTIMIZATION,
            confidence_score=confidence_score,
            success=workflow_data.get("success", False),
            response_time=workflow_data.get("execution_time", 0.0)
        )
        
        return experience

    async def incorporate_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Incorporate user feedback into learning."""
        if not self.enabled:
            return
        
        try:
            # Create feedback-based learning experience
            experience = await self._create_feedback_experience(feedback_data)
            
            if experience:
                # Add high importance weight for feedback-based experiences
                experience.importance_weight = 2.0
                
                self.experience_buffer.append(experience)
                self.replay_buffer.add_experience(experience)
                
                if self.database:
                    await self._store_experience(experience)
                
                # Trigger immediate learning update for high-value feedback
                if experience.feedback_score and experience.feedback_score > 0.8:
                    task_id = f"feedback_update_{experience.domain.value}_{int(time.time())}"
                    self.active_learning_tasks[task_id] = asyncio.create_task(
                        self._perform_learning_update(experience.domain, task_id)
                    )
                
                self.logger.debug(f"Incorporated feedback: {experience.experience_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to incorporate feedback: {str(e)}")

    async def _create_feedback_experience(self, feedback_data: Dict[str, Any]) -> Optional[LearningExperience]:
        """Create learning experience from feedback data."""
        if "interaction_id" not in feedback_data:
            return None
        
        experience_id = str(uuid.uuid4())
        
        # Extract feedback information
        input_data = {
            "interaction_id": feedback_data["interaction_id"],
            "feedback_type": feedback_data.get("feedback_type", "rating"),
            "original_response": feedback_data.get("original_response", "")
        }
        
        predicted_output = {
            "expected_satisfaction": 0.7  # Default expectation
        }
        
        actual_output = {
            "user_satisfaction": feedback_data.get("rating", feedback_data.get("score", 0.5)),
            "feedback_text": feedback_data.get("feedback_text", "")
        }
        
        # Determine success based on feedback
        feedback_score = feedback_data.get("rating", feedback_data.get("score", 0.5))
        success = feedback_score >= 0.6 if feedback_score is not None else True
        
        experience = LearningExperience(
            experience_id=experience_id,
            session_id=feedback_data.get("session_id", ""),
            user_id=feedback_data.get("user_id"),
            input_data=input_data,
            predicted_output=predicted_output,
            actual_output=actual_output,
            domain=LearningDomain.USER_PREFERENCES,
            confidence_score=abs(feedback_score - 0.5) * 2 if feedback_score is not None else 0.5,
            feedback_score=feedback_score,
            success=success
        )
        
        return experience

    async def periodic_update(self) -> None:
        """Perform periodic learning updates."""
        if not self.enabled:
            return
        
        try:
            for domain in self.config.learning_domains:
                if self._should_trigger_update(domain):
                    task_id = f"periodic_update_{domain.value}_{int(time.time())}"
                    self.active_learning_tasks[task_id] = asyncio.create_task(
                        self._perform_learning_update(domain, task_id)
                    )
            
        except Exception as e:
            self.logger.error(f"Periodic update failed: {str(e)}")

    async def _learning_update_loop(self) -> None:
        """Background task for learning updates."""
        while True:
            try:
                if self.enabled and self.config.learning_mode == LearningMode.BATCH:
                    await self.periodic_update()
                
                await asyncio.sleep(self.config.consolidation_interval)
                
            except Exception as e:
                self.logger.error(f"Learning update loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                # Update learning metrics
                for domain in self.config.learning_domains:
                    try:
                        current_metrics = await self._evaluate_current_performance(domain)
                        
                        # Store metrics for trend analysis
                        for metric_name, value in current_metrics.items():
                            self.learning_metrics[f"{domain.value}_{metric_name}"].append(value)
                            
                            # Update Prometheus metrics
                            self.metrics.set(
                                f"learning_performance_{metric_name}",
                                value,
                                tags={"domain": domain.value}
                            )
                        
                        # Check for performance degradation
                        if domain in self.forgetting_detector.baseline_metrics:
                            forgetting_analysis = self.forgetting_detector.detect_forgetting(
                                domain, current_metrics
                            )
                            
                            if forgetting_analysis["forgetting_detected"]:
                                await self._handle_catastrophic_forgetting(
                                    domain, forgetting_analysis, None
                                )
                    
                    except Exception as e:
                        self.logger.warning(f"Performance monitoring failed for {domain}: {str(e)}")
                
                await asyncio.sleep(self.config.evaluation_interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitoring loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _forgetting_detection_loop(self) -> None:
        """Background task for forgetting detection."""
        while True:
            try:
                for domain in self.config.learning_domains:
                    try:
                        # Get current performance
                        current_metrics = await self._evaluate_current_performance(domain)
                        
                        # Check for forgetting
                        forgetting_analysis = self.forgetting_detector.detect_forgetting(
                            domain, current_metrics
                        )
                        
                        if forgetting_analysis["forgetting_detected"]:
                            await self._handle_catastrophic_forgetting(
                                domain, forgetting_analysis, None
                            )
                        
                        # Update trend analysis
                        trend_analysis = self.forgetting_detector.get_forgetting_trend(domain)
                        if trend_analysis["trend"] == "degrading":
                            self.logger.warning(f"Performance degradation trend detected for {domain}")
                    
                    except Exception as e:
                        self.logger.warning(f"Forgetting detection failed for {domain}: {str(e)}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Forgetting detection loop error: {str(e)}")
                await asyncio.sleep(300)

    async def _handle_feedback_event(self, event) -> None:
        """Handle feedback received events."""
        try:
            await self.incorporate_feedback(event.feedback)
        except Exception as e:
            self.logger.error(f"Error handling feedback event: {str(e)}")
