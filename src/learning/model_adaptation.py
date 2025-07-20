"""
Advanced Model Adaptation System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 03:21:22 UTC

This module provides comprehensive model adaptation capabilities for the AI assistant,
including dynamic parameter adjustment, fine-tuning management, performance optimization,
personalization, and domain adaptation across different model architectures.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar, Tuple
import asyncio
import threading
import time
import json
import pickle
import hashlib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ModelAdaptationStarted, ModelAdaptationCompleted, ModelAdaptationFailed,
    ModelFineTuneStarted, ModelFineTuneCompleted, ModelParametersUpdated,
    PersonalizationUpdated, DomainAdaptationCompleted, ModelOptimized,
    AdaptationPolicyChanged, PerformanceThresholdExceeded, ModelRollback,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager

# Assistant components
from src.assistant.core_engine import EnhancedCoreEngine
from src.assistant.component_manager import EnhancedComponentManager
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.redis_cache import RedisCache

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class AdaptationType(Enum):
    """Types of model adaptation."""
    PARAMETER_UPDATE = "parameter_update"
    FINE_TUNING = "fine_tuning"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    ARCHITECTURE_SEARCH = "architecture_search"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    DOMAIN_ADAPTATION = "domain_adaptation"
    PERSONALIZATION = "personalization"


class AdaptationScope(Enum):
    """Scope of adaptation application."""
    GLOBAL = "global"           # System-wide adaptation
    USER_SPECIFIC = "user_specific"     # User-specific adaptation
    SESSION_SPECIFIC = "session_specific"   # Session-specific adaptation
    DOMAIN_SPECIFIC = "domain_specific"     # Domain-specific adaptation
    TASK_SPECIFIC = "task_specific"       # Task-specific adaptation
    TEMPORARY = "temporary"     # Temporary adaptation


class AdaptationStrategy(Enum):
    """Adaptation strategies."""
    CONSERVATIVE = "conservative"   # Minimal changes, high stability
    BALANCED = "balanced"          # Balanced approach
    AGGRESSIVE = "aggressive"      # Maximum adaptation, fast learning
    ADAPTIVE = "adaptive"          # Strategy adapts based on context
    EXPERIMENTAL = "experimental"  # Experimental strategies


class ModelType(Enum):
    """Types of models that can be adapted."""
    LANGUAGE_MODEL = "language_model"
    VISION_MODEL = "vision_model"
    SPEECH_MODEL = "speech_model"
    MULTIMODAL_MODEL = "multimodal_model"
    EMBEDDING_MODEL = "embedding_model"
    CLASSIFIER = "classifier"
    REGRESSION = "regression"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class AdaptationStatus(Enum):
    """Status of adaptation processes."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"


@dataclass
class AdaptationConfiguration:
    """Configuration for model adaptation."""
    adaptation_type: AdaptationType
    adaptation_scope: AdaptationScope
    strategy: AdaptationStrategy = AdaptationStrategy.BALANCED
    
    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    patience: int = 5
    min_delta: float = 0.001
    
    # Resource constraints
    max_memory_mb: float = 1024.0
    max_gpu_memory_mb: float = 2048.0
    max_training_time_seconds: float = 3600.0
    max_cpu_cores: int = 4
    
    # Optimization settings
    optimizer_type: str = "adam"
    scheduler_type: str = "cosine"
    gradient_clipping: float = 1.0
    weight_decay: float = 0.01
    
    # Adaptation behavior
    auto_save_checkpoints: bool = True
    enable_early_stopping: bool = True
    validation_split: float = 0.2
    adaptation_frequency: str = "daily"  # immediate, hourly, daily, weekly
    
    # Quality gates
    min_performance_threshold: float = 0.8
    max_performance_degradation: float = 0.1
    require_validation: bool = True
    
    # Security and privacy
    enable_differential_privacy: bool = True
    privacy_epsilon: float = 1.0
    enable_federated_learning: bool = False
    
    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSnapshot:
    """Snapshot of model state for rollback purposes."""
    snapshot_id: str
    model_id: str
    timestamp: datetime
    
    # Model state
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    adaptation_config: Optional[AdaptationConfiguration] = None
    
    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    file_path: Optional[Path] = None
    checksum: Optional[str] = None


@dataclass
class AdaptationTask:
    """Represents an adaptation task."""
    task_id: str
    model_id: str
    adaptation_config: AdaptationConfiguration
    
    # Task state
    status: AdaptationStatus = AdaptationStatus.PENDING
    progress: float = 0.0
    
    # Data and context
    training_data: Optional[Any] = None
    validation_data: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    initial_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # Resource usage
    memory_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    cpu_usage: float = 0.0
    training_time: float = 0.0


class ModelAdaptationError(Exception):
    """Custom exception for model adaptation operations."""
    
    def __init__(self, message: str, model_id: Optional[str] = None, 
                 task_id: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.model_id = model_id
        self.task_id = task_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    @abstractmethod
    def get_supported_types(self) -> List[ModelType]:
        """Get list of supported model types."""
        pass
    
    @abstractmethod
    async def adapt_model(self, model: Any, task: AdaptationTask) -> Any:
        """Adapt the model according to the task configuration."""
        pass
    
    @abstractmethod
    async def validate_adaptation(self, model: Any, task: AdaptationTask) -> Dict[str, float]:
        """Validate the adapted model and return metrics."""
        pass
    
    @abstractmethod
    def create_snapshot(self, model: Any) -> ModelSnapshot:
        """Create a snapshot of the current model state."""
        pass
    
    @abstractmethod
    async def restore_snapshot(self, model: Any, snapshot: ModelSnapshot) -> Any:
        """Restore model from snapshot."""
        pass


class LanguageModelAdapter(BaseModelAdapter):
    """Adapter for language models."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def get_supported_types(self) -> List[ModelType]:
        """Get supported model types."""
        return [ModelType.LANGUAGE_MODEL, ModelType.EMBEDDING_MODEL]
    
    async def adapt_model(self, model: Any, task: AdaptationTask) -> Any:
        """Adapt language model using various techniques."""
        config = task.adaptation_config
        
        if config.adaptation_type == AdaptationType.FINE_TUNING:
            return await self._fine_tune_model(model, task)
        elif config.adaptation_type == AdaptationType.PROMPT_OPTIMIZATION:
            return await self._optimize_prompts(model, task)
        elif config.adaptation_type == AdaptationType.PARAMETER_UPDATE:
            return await self._update_parameters(model, task)
        else:
            raise ModelAdaptationError(f"Unsupported adaptation type: {config.adaptation_type}")
    
    async def _fine_tune_model(self, model: Any, task: AdaptationTask) -> Any:
        """Fine-tune the language model."""
        config = task.adaptation_config
        
        # Setup training
        if hasattr(model, 'train'):
            model.train()
        
        # Setup optimizer
        optimizer = self._create_optimizer(model, config)
        scheduler = self._create_scheduler(optimizer, config)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Training step
            if task.training_data:
                for batch in self._create_data_loader(task.training_data, config.batch_size):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if config.gradient_clipping > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Update progress
                    task.progress = (epoch + batch_count / len(task.training_data)) / config.num_epochs
            
            # Validation
            if task.validation_data:
                val_loss = await self._validate_model(model, task.validation_data)
                
                # Early stopping
                if config.enable_early_stopping:
                    if val_loss < best_loss - config.min_delta:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config.patience:
                            self.logger.info(f"Early stopping at epoch {epoch}")
                            break
            
            # Step scheduler
            if scheduler:
                scheduler.step()
            
            # Update adaptation history
            task.adaptation_history.append({
                'epoch': epoch,
                'train_loss': epoch_loss / max(batch_count, 1),
                'val_loss': val_loss if task.validation_data else None,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Set to eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        return model
    
    async def _optimize_prompts(self, model: Any, task: AdaptationTask) -> Any:
        """Optimize prompts for the model."""
        # Implement prompt optimization logic
        # This would involve techniques like prompt tuning, prefix tuning, etc.
        return model
    
    async def _update_parameters(self, model: Any, task: AdaptationTask) -> Any:
        """Update specific model parameters."""
        # Implement parameter update logic
        return model
    
    def _create_optimizer(self, model: Any, config: AdaptationConfiguration) -> optim.Optimizer:
        """Create optimizer for training."""
        if config.optimizer_type.lower() == "adam":
            return optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optimizer_type.lower() == "sgd":
            return optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def _create_scheduler(self, optimizer: optim.Optimizer, config: AdaptationConfiguration) -> Optional[Any]:
        """Create learning rate scheduler."""
        if config.scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        elif config.scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=config.num_epochs // 3)
        return None
    
    def _create_data_loader(self, data: Any, batch_size: int) -> Any:
        """Create data loader for training."""
        # This would create appropriate data loaders based on the data format
        return data  # Placeholder
    
    async def _validate_model(self, model: Any, validation_data: Any) -> float:
        """Validate the model and return loss."""
        if hasattr(model, 'eval'):
            model.eval()
        
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in validation_data:
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                batch_count += 1
        
        return total_loss / max(batch_count, 1)
    
    async def validate_adaptation(self, model: Any, task: AdaptationTask) -> Dict[str, float]:
        """Validate the adapted model."""
        metrics = {}
        
        if task.validation_data:
            metrics['validation_loss'] = await self._validate_model(model, task.validation_data)
        
        # Add perplexity calculation if applicable
        if hasattr(model, 'config') and 'vocab_size' in model.config:
            metrics['perplexity'] = np.exp(metrics.get('validation_loss', 0))
        
        return metrics
    
    def create_snapshot(self, model: Any) -> ModelSnapshot:
        """Create model snapshot."""
        snapshot_id = str(uuid.uuid4())
        
        # Create state dict
        state_dict = model.state_dict() if hasattr(model, 'state_dict') else {}
        
        return ModelSnapshot(
            snapshot_id=snapshot_id,
            model_id=getattr(model, 'model_id', 'unknown'),
            timestamp=datetime.now(timezone.utc),
            model_state_dict=state_dict,
            model_config=getattr(model, 'config', {})
        )
    
    async def restore_snapshot(self, model: Any, snapshot: ModelSnapshot) -> Any:
        """Restore model from snapshot."""
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(snapshot.model_state_dict)
        
        return model


class VisionModelAdapter(BaseModelAdapter):
    """Adapter for vision models."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def get_supported_types(self) -> List[ModelType]:
        """Get supported model types."""
        return [ModelType.VISION_MODEL, ModelType.CLASSIFIER]
    
    async def adapt_model(self, model: Any, task: AdaptationTask) -> Any:
        """Adapt vision model."""
        # Implement vision model adaptation
        return model
    
    async def validate_adaptation(self, model: Any, task: AdaptationTask) -> Dict[str, float]:
        """Validate vision model adaptation."""
        return {}
    
    def create_snapshot(self, model: Any) -> ModelSnapshot:
        """Create vision model snapshot."""
        return ModelSnapshot(
            snapshot_id=str(uuid.uuid4()),
            model_id=getattr(model, 'model_id', 'unknown'),
            timestamp=datetime.now(timezone.utc),
            model_state_dict=model.state_dict() if hasattr(model, 'state_dict') else {}
        )
    
    async def restore_snapshot(self, model: Any, snapshot: ModelSnapshot) -> Any:
        """Restore vision model from snapshot."""
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(snapshot.model_state_dict)
        return model


class PersonalizationEngine:
    """Engine for personalized model adaptations."""
    
    def __init__(self, preference_learner: PreferenceLearner, logger):
        self.preference_learner = preference_learner
        self.logger = logger
        self.user_adaptations: Dict[str, List[str]] = defaultdict(list)
    
    async def create_personalized_adaptation(
        self,
        user_id: str,
        model_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> AdaptationConfiguration:
        """Create personalized adaptation configuration."""
        # Get user preferences
        user_prefs = await self.preference_learner.get_user_preferences(user_id)
        
        # Analyze interaction patterns
        patterns = self._analyze_interaction_patterns(interaction_history)
        
        # Create adaptation config
        config = AdaptationConfiguration(
            adaptation_type=AdaptationType.PERSONALIZATION,
            adaptation_scope=AdaptationScope.USER_SPECIFIC,
            strategy=AdaptationStrategy.CONSERVATIVE,
            learning_rate=0.0001,  # Conservative for personalization
            num_epochs=5,
            description=f"Personalized adaptation for user {user_id}",
            tags={"personalization", "user_specific"},
            metadata={
                'user_id': user_id,
                'patterns': patterns,
                'preferences': user_prefs
            }
        )
        
        return config
    
    def _analyze_interaction_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        patterns = {
            'avg_session_length': 0,
            'preferred_topics': [],
            'interaction_frequency': 0,
            'response_preferences': {},
            'error_patterns': []
        }
        
        if not history:
            return patterns
        
        # Calculate average session length
        session_lengths = [item.get('session_length', 0) for item in history]
        patterns['avg_session_length'] = sum(session_lengths) / len(session_lengths)
        
        # Extract preferred topics
        topics = [item.get('topic') for item in history if item.get('topic')]
        patterns['preferred_topics'] = list(set(topics))
        
        # Calculate interaction frequency
        patterns['interaction_frequency'] = len(history)
        
        return patterns


class DomainAdaptationEngine:
    """Engine for domain-specific adaptations."""
    
    def __init__(self, knowledge_graph, logger):
        self.knowledge_graph = knowledge_graph
        self.logger = logger
        self.domain_models: Dict[str, str] = {}  # domain -> model_id
    
    async def adapt_to_domain(
        self,
        model_id: str,
        domain: str,
        domain_data: Optional[Any] = None
    ) -> AdaptationConfiguration:
        """Create domain adaptation configuration."""
        # Analyze domain requirements
        domain_analysis = await self._analyze_domain(domain, domain_data)
        
        # Determine adaptation strategy
        strategy = self._determine_domain_strategy(domain_analysis)
        
        config = AdaptationConfiguration(
            adaptation_type=AdaptationType.DOMAIN_ADAPTATION,
            adaptation_scope=AdaptationScope.DOMAIN_SPECIFIC,
            strategy=strategy,
            learning_rate=0.001,
            num_epochs=15,
            description=f"Domain adaptation for {domain}",
            tags={"domain_adaptation", domain},
            metadata={
                'domain': domain,
                'analysis': domain_analysis
            }
        )
        
        return config
    
    async def _analyze_domain(self, domain: str, data: Optional[Any]) -> Dict[str, Any]:
        """Analyze domain characteristics."""
        analysis = {
            'domain': domain,
            'data_characteristics': {},
            'complexity': 'medium',
            'specialization_required': False
        }
        
        # Use knowledge graph to understand domain
        if self.knowledge_graph:
            domain_info = await self.knowledge_graph.get_domain_info(domain)
            analysis.update(domain_info)
        
        return analysis
    
    def _determine_domain_strategy(self, analysis: Dict[str, Any]) -> AdaptationStrategy:
        """Determine optimal adaptation strategy for domain."""
        complexity = analysis.get('complexity', 'medium')
        
        if complexity == 'high':
            return AdaptationStrategy.AGGRESSIVE
        elif complexity == 'low':
            return AdaptationStrategy.CONSERVATIVE
        else:
            return AdaptationStrategy.BALANCED


class PerformanceOptimizer:
    """Optimizer for model performance."""
    
    def __init__(self, metrics_collector: MetricsCollector, logger):
        self.metrics = metrics_collector
        self.logger = logger
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def optimize_model_performance(
        self,
        model_id: str,
        performance_metrics: Dict[str, float],
        resource_constraints: Dict[str, float]
    ) -> Optional[AdaptationConfiguration]:
        """Create optimization configuration based on performance metrics."""
        # Analyze current performance
        bottlenecks = self._identify_bottlenecks(performance_metrics, resource_constraints)
        
        if not bottlenecks:
            return None
        
        # Determine optimization techniques
        techniques = self._select_optimization_techniques(bottlenecks, resource_constraints)
        
        if not techniques:
            return None
        
        # Create optimization config
        config = AdaptationConfiguration(
            adaptation_type=techniques[0],  # Primary technique
            adaptation_scope=AdaptationScope.GLOBAL,
            strategy=AdaptationStrategy.BALANCED,
            description=f"Performance optimization for {model_id}",
            tags={"performance", "optimization"},
            metadata={
                'bottlenecks': bottlenecks,
                'techniques': techniques,
                'constraints': resource_constraints
            }
        )
        
        return config
    
    def _identify_bottlenecks(
        self,
        metrics: Dict[str, float],
        constraints: Dict[str, float]
    ) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Memory usage
        if metrics.get('memory_usage', 0) > constraints.get('max_memory', 1000) * 0.9:
            bottlenecks.append('memory')
        
        # Response time
        if metrics.get('response_time', 0) > 5.0:  # 5 second threshold
            bottlenecks.append('latency')
        
        # Accuracy
        if metrics.get('accuracy', 1.0) < 0.8:  # 80% threshold
            bottlenecks.append('accuracy')
        
        return bottlenecks
    
    def _select_optimization_techniques(
        self,
        bottlenecks: List[str],
        constraints: Dict[str, float]
    ) -> List[AdaptationType]:
        """Select appropriate optimization techniques."""
        techniques = []
        
        if 'memory' in bottlenecks:
            techniques.extend([AdaptationType.QUANTIZATION, AdaptationType.PRUNING])
        
        if 'latency' in bottlenecks:
            techniques.extend([AdaptationType.QUANTIZATION, AdaptationType.KNOWLEDGE_DISTILLATION])
        
        if 'accuracy' in bottlenecks:
            techniques.extend([AdaptationType.FINE_TUNING, AdaptationType.HYPERPARAMETER_TUNING])
        
        return techniques


class ModelAdaptationManager:
    """
    Advanced Model Adaptation System for the AI Assistant.
    
    This manager provides comprehensive model adaptation capabilities including:
    - Dynamic parameter adjustment based on user interactions
    - Automated fine-tuning workflows for different model types
    - Performance optimization with resource awareness
    - User-specific personalization and domain adaptation
    - Comprehensive monitoring and rollback capabilities
    - Integration with all core system components
    """
    
    def __init__(self, container: Container):
        """
        Initialize the model adaptation manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Assistant components
        self.core_engine = container.get(EnhancedCoreEngine)
        self.component_manager = container.get(EnhancedComponentManager)
        self.session_manager = container.get(EnhancedSessionManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        
        # Memory and learning
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Integrations
        self.model_router = container.get(ModelRouter)
        try:
            self.database = container.get(DatabaseManager)
            self.redis_cache = container.get(RedisCache)
        except Exception:
            self.database = None
            self.redis_cache = None
        
        # Security
        try:
            self.encryption = container.get(EncryptionManager)
        except Exception:
            self.encryption = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Adaptation infrastructure
        self.model_adapters: Dict[ModelType, BaseModelAdapter] = {}
        self.adaptation_tasks: Dict[str, AdaptationTask] = {}
        self.model_snapshots: Dict[str, List[ModelSnapshot]] = defaultdict(list)
        self.active_adaptations: Dict[str, str] = {}  # model_id -> task_id
        
        # Specialized engines
        self.personalization_engine = PersonalizationEngine(self.preference_learner, self.logger)
        self.domain_adaptation_engine = DomainAdaptationEngine(
            getattr(self, 'knowledge_graph', None), self.logger
        )
        self.performance_optimizer = PerformanceOptimizer(self.metrics, self.logger)
        
        # Execution infrastructure
        self.adaptation_semaphore = asyncio.Semaphore(3)  # Max concurrent adaptations
        self.thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adaptation")
        
        # Configuration
        self.auto_adaptation_enabled = self.config.get("adaptation.auto_enabled", True)
        self.max_snapshots_per_model = self.config.get("adaptation.max_snapshots", 10)
        self.adaptation_frequency = self.config.get("adaptation.frequency", "daily")
        self.performance_threshold = self.config.get("adaptation.performance_threshold", 0.8)
        
        # State management
        self.adaptation_policies: Dict[str, Dict[str, Any]] = {}
        self.user_adaptation_history: Dict[str, List[str]] = defaultdict(list)
        
        # Initialize components
        self._setup_model_adapters()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("model_adaptation", self._health_check_callback)
        
        self.logger.info("ModelAdaptationManager initialized successfully")

    def _setup_model_adapters(self) -> None:
        """Setup model adapters for different model types."""
        try:
            # Language model adapter
            lang_adapter = LanguageModelAdapter(self.logger)
            for model_type in lang_adapter.get_supported_types():
                self.model_adapters[model_type] = lang_adapter
            
            # Vision model adapter
            vision_adapter = VisionModelAdapter(self.logger)
            for model_type in vision_adapter.get_supported_types():
                self.model_adapters[model_type] = vision_adapter
            
            self.logger.info(f"Initialized {len(self.model_adapters)} model adapters")
            
        except Exception as e:
            self.logger.error(f"Failed to setup model adapters: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register adaptation metrics
            self.metrics.register_counter("model_adaptations_total")
            self.metrics.register_counter("model_adaptations_successful")
            self.metrics.register_counter("model_adaptations_failed")
            self.metrics.register_histogram("model_adaptation_duration_seconds")
            self.metrics.register_gauge("active_adaptations")
            self.metrics.register_counter("model_snapshots_created")
            self.metrics.register_counter("model_rollbacks_total")
            self.metrics.register_histogram("model_performance_improvement")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the model adaptation manager."""
        try:
            # Load adaptation policies
            await self._load_adaptation_policies()
            
            # Start background tasks
            asyncio.create_task(self._adaptation_monitor_loop())
            asyncio.create_task(self._performance_monitor_loop())
            asyncio.create_task(self._cleanup_old_snapshots_loop())
            
            if self.auto_adaptation_enabled:
                asyncio.create_task(self._auto_adaptation_loop())
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("ModelAdaptationManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ModelAdaptationManager: {str(e)}")
            raise ModelAdaptationError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User feedback events
        self.event_bus.subscribe("feedback_received", self._handle_feedback_event)
        
        # Performance events
        self.event_bus.subscribe("performance_threshold_exceeded", self._handle_performance_event)
        
        # Session events
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Model events
        self.event_bus.subscribe("model_loaded", self._handle_model_loaded)
        self.event_bus.subscribe("model_performance_degraded", self._handle_performance_degradation)

    @handle_exceptions
    async def create_adaptation_task(
        self,
        model_id: str,
        model_type: ModelType,
        adaptation_config: AdaptationConfiguration,
        training_data: Optional[Any] = None,
        validation_data: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new model adaptation task.
        
        Args:
            model_id: Identifier of the model to adapt
            model_type: Type of the model
            adaptation_config: Configuration for the adaptation
            training_data: Optional training data
            validation_data: Optional validation data
            context: Optional context information
            
        Returns:
            Task ID for the adaptation
        """
        # Validate inputs
        if model_type not in self.model_adapters:
            raise ModelAdaptationError(f"No adapter available for model type: {model_type}")
        
        # Check if model is already being adapted
        if model_id in self.active_adaptations:
            raise ModelAdaptationError(f"Model {model_id} is already being adapted")
        
        # Create adaptation task
        task = AdaptationTask(
            task_id=str(uuid.uuid4()),
            model_id=model_id,
            adaptation_config=adaptation_config,
            training_data=training_data,
            validation_data=validation_data,
            context=context or {}
        )
        
        # Store task
        self.adaptation_tasks[task.task_id] = task
        self.active_adaptations[model_id] = task.task_id
        
        # Emit task creation event
        await self.event_bus.emit(ModelAdaptationStarted(
            model_id=model_id,
            task_id=task.task_id,
            adaptation_type=adaptation_config.adaptation_type.value,
            adaptation_scope=adaptation_config.adaptation_scope.value
        ))
        
        # Start adaptation in background
        asyncio.create_task(self._execute_adaptation_task(task, model_type))
        
        self.logger.info(f"Created adaptation task: {task.task_id} for model: {model_id}")
        return task.task_id

    async def _execute_adaptation_task(self, task: AdaptationTask, model_type: ModelType) -> None:
        """Execute an adaptation task asynchronously."""
        async with self.adaptation_semaphore:
            start_time = time.time()
            task.started_at = datetime.now(timezone.utc)
            task.status = AdaptationStatus.IN_PROGRESS
            
            try:
                with self.tracer.trace("model_adaptation") as span:
                    span.set_attributes({
                        "model_id": task.model_id,
                        "task_id": task.task_id,
                        "adaptation_type": task.adaptation_config.adaptation_type.value,
                        "adaptation_scope": task.adaptation_config.adaptation_scope.value
                    })
                    
                    # Get model from router
                    model = await self.model_router.get_model(task.model_id)
                    if not model:
                        raise ModelAdaptationError(f"Model {task.model_id} not found")
                    
                    # Create snapshot before adaptation
                    snapshot = await self._create_model_snapshot(model, model_type, task)
                    
                    # Get adapter
                    adapter = self.model_adapters[model_type]
                    
                    # Get initial metrics
                    task.initial_metrics = await adapter.validate_adaptation(model, task)
                    
                    # Perform adaptation
                    adapted_model = await adapter.adapt_model(model, task)
                    
                    # Validate adaptation
                    task.final_metrics = await adapter.validate_adaptation(adapted_model, task)
                    
                    # Check if adaptation improved performance
                    improvement = self._calculate_improvement(task.initial_metrics, task.final_metrics)
                    
                    if improvement < task.adaptation_config.min_performance_threshold:
                        # Rollback if performance didn't improve sufficiently
                        await self._rollback_adaptation(model, snapshot, adapter)
                        task.status = AdaptationStatus.FAILED
                        task.error_message = f"Adaptation didn't meet performance threshold: {improvement}"
                    else:
                        # Update model in router
                        await self.model_router.update_model(task.model_id, adapted_model)
                        task.status = AdaptationStatus.COMPLETED
                        
                        # Store adaptation results
                        await self._store_adaptation_results(task, improvement)
                    
                    # Calculate execution time
                    task.training_time = time.time() - start_time
                    task.completed_at = datetime.now(timezone.utc)
                    
                    # Update metrics
                    self.metrics.increment("model_adaptations_total")
                    if task.status == AdaptationStatus.COMPLETED:
                        self.metrics.increment("model_adaptations_successful")
                        self.metrics.record("model_performance_improvement", improvement)
                    else:
                        self.metrics.increment("model_adaptations_failed")
                    
                    self.metrics.record("model_adaptation_duration_seconds", task.training_time)
                    
                    # Emit completion event
                    if task.status == AdaptationStatus.COMPLETED:
                        await self.event_bus.emit(ModelAdaptationCompleted(
                            model_id=task.model_id,
                            task_id=task.task_id,
                            adaptation_type=task.adaptation_config.adaptation_type.value,
                            performance_improvement=improvement,
                            execution_time=task.training_time
                        ))
                    else:
                        await self.event_bus.emit(ModelAdaptationFailed(
                            model_id=task.model_id,
                            task_id=task.task_id,
                            error_message=task.error_message or "Unknown error",
                            execution_time=task.training_time
                        ))
                    
                    self.logger.info(
                        f"Adaptation task {task.task_id} completed with status: {task.status.value} "
                        f"in {task.training_time:.2f}s"
                    )
                    
            except Exception as e:
                # Handle adaptation failure
                task.status = AdaptationStatus.FAILED
                task.error_message = str(e)
                task.training_time = time.time() - start_time
                task.completed_at = datetime.now(timezone.utc)
                
                await self.event_bus.emit(ModelAdaptationFailed(
                    model_id=task.model_id,
                    task_id=task.task_id,
                    error_message=str(e),
                    execution_time=task.training_time
                ))
                
                self.metrics.increment("model_adaptations_failed")
                self.logger.error(f"Adaptation task {task.task_id} failed: {str(e)}")
                
            finally:
                # Cleanup
                self.active_adaptations.pop(task.model_id, None)

    async def _create_model_snapshot(
        self,
        model: Any,
        model_type: ModelType,
        task: AdaptationTask
    ) -> ModelSnapshot:
        """Create a snapshot of the model before adaptation."""
        adapter = self.model_adapters[model_type]
        snapshot = adapter.create_snapshot(model)
        
        # Add task-specific metadata
        snapshot.description = f"Pre-adaptation snapshot for task {task.task_id}"
        snapshot.tags.add("pre_adaptation")
        snapshot.adaptation_config = task.adaptation_config
        
        # Store snapshot
        self.model_snapshots[task.model_id].append(snapshot)
        
        # Limit number of snapshots
        if len(self.model_snapshots[task.model_id]) > self.max_snapshots_per_model:
            # Remove oldest snapshot
            oldest = self.model_snapshots[task.model_id].pop(0)
            await self._cleanup_snapshot(oldest)
        
        # Save snapshot to disk if possible
        if self.database:
            await self._save_snapshot_to_storage(snapshot)
        
        self.metrics.increment("model_snapshots_created")
        return snapshot

    def _calculate_improvement(
        self,
        initial_metrics: Dict[str, float],
        final_metrics: Dict[str, float]
    ) -> float:
        """Calculate performance improvement from adaptation."""
        if not initial_metrics or not final_metrics:
            return 0.0
        
        improvements = []
        
        # Calculate improvement for each metric
        for metric_name in initial_metrics:
            if metric_name in final_metrics:
                initial = initial_metrics[metric_name]
                final = final_metrics[metric_name]
                
                if initial > 0:
                    # For metrics where higher is better (accuracy, f1, etc.)
                    if metric_name in ['accuracy', 'f1_score', 'precision', 'recall']:
                        improvement = (final - initial) / initial
                    # For metrics where lower is better (loss, error, etc.)
                    else:
                        improvement = (initial - final) / initial
                    
                    improvements.append(improvement)
        
        # Return average improvement
        return sum(improvements) / len(improvements) if improvements else 0.0

    async def _rollback_adaptation(
        self,
        model: Any,
        snapshot: ModelSnapshot,
        adapter: BaseModelAdapter
    ) -> None:
        """Rollback model to previous snapshot."""
        try:
            await adapter.restore_snapshot(model, snapshot)
            
            await self.event_bus.emit(ModelRollback(
                model_id=snapshot.model_id,
                snapshot_id=snapshot.snapshot_id,
                reason="Performance degradation"
            ))
            
            self.metrics.increment("model_rollbacks_total")
            self.logger.info(f"Rolled back model {snapshot.model_id} to snapshot {snapshot.snapshot_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to rollback model {snapshot.model_id}: {str(e)}")

    async def _store_adaptation_results(self, task: AdaptationTask, improvement: float) -> None:
        """Store adaptation results for learning."""
        try:
            results = {
                'task_id': task.task_id,
                'model_id': task.model_id,
                'adaptation_type': task.adaptation_config.adaptation_type.value,
                'adaptation_scope': task.adaptation_config.adaptation_scope.value,
                'initial_metrics': task.initial_metrics,
                'final_metrics': task.final_metrics,
                'improvement': improvement,
                'training_time': task.training_time,
                'context': task.context,
                'timestamp': task.completed_at.isoformat() if task.completed_at else None
            }
            
            # Store in memory for continual learning
            await self.continual_learner.learn_from_adaptation(results)
            
            # Store in episodic memory
            await self.memory_manager.store_episodic_memory(
                event_type="model_adaptation",
                data=results,
                session_id=task.context.get('session_id')
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store adaptation results: {str(e)}")

    @handle_exceptions
    async def create_personalized_adaptation(
        self,
        user_id: str,
        model_id: str,
        model_type: ModelType,
        session_id: Optional[str] = None
    ) -> str:
        """
        Create a personalized adaptation for a specific user.
        
        Args:
            user_id: User identifier
            model_id: Model identifier
            model_type: Type of the model
            session_id: Optional session identifier
            
        Returns:
            Task ID for the personalized adaptation
        """
        # Get user interaction history
        interaction_history = await self._get_user_interaction_history(user_id, session_id)
        
        # Create personalized configuration
        config = await self.personalization_engine.create_personalized_adaptation(
            user_id, model_id, interaction_history
        )
        
        # Extract training data from interactions
        training_data = self._extract_training_data_from_interactions(interaction_history)
        
        # Create adaptation task
        task_id = await self.create_adaptation_task(
            model_id=model_id,
            model_type=model_type,
            adaptation_config=config,
            training_data=training_data,
            context={'user_id': user_id, 'session_id': session_id}
        )
        
        # Track user adaptation
        self.user_adaptation_history[user_id].append(task_id)
        
        return task_id

    @handle_exceptions
    async def create_domain_adaptation(
        self,
        model_id: str,
        model_type: ModelType,
        domain: str,
        domain_data: Optional[Any] = None
    ) -> str:
        """
        Create a domain-specific adaptation.
        
        Args:
            model_id: Model identifier
            model_type: Type of the model
            domain: Target domain
            domain_data: Optional domain-specific data
            
        Returns:
            Task ID for the domain adaptation
        """
        # Create domain adaptation configuration
        config = await self.domain_adaptation_engine.adapt_to_domain(
            model_id, domain, domain_data
        )
        
        # Create adaptation task
        task_id = await self.create_adaptation_task(
            model_id=model_id,
            model_type=model_type,
            adaptation_config=config,
            training_data=domain_data,
            context={'domain': domain}
        )
        
        return task_id

    @handle_exceptions
    async def optimize_model_performance(
        self,
        model_id: str,
        model_type: ModelType,
        performance_metrics: Dict[str, float],
        resource_constraints: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """
        Optimize model performance based on current metrics.
        
        Args:
            model_id: Model identifier
            model_type: Type of the model
            performance_metrics: Current performance metrics
            resource_constraints: Optional resource constraints
            
        Returns:
            Task ID for the optimization or None if no optimization needed
        """
        constraints = resource_constraints or {
            'max_memory': 2048.0,
            'max_training_time': 1800.0
        }
        
        # Get optimization configuration
        config = await self.performance_optimizer.optimize_model_performance(
            model_id, performance_metrics, constraints
        )
        
        if not config:
            return None
        
        # Create optimization task
        task_id = await self.create_adaptation_task(
            model_id=model_id,
            model_type=model_type,
            adaptation_config=config,
            context={'optimization': True, 'metrics': performance_metrics}
        )
        
        return task_id

    async def get_adaptation_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of an adaptation task."""
        if task_id not in self.adaptation_tasks:
            raise ModelAdaptationError(f"Adaptation task {task_id} not found")
        
        task = self.adaptation_tasks[task_id]
        
        return {
            'task_id': task.task_id,
            'model_id': task.model_id,
            'status': task.status.value,
            'progress': task.progress,
            'adaptation_type': task.adaptation_config.adaptation_type.value,
            'adaptation_scope': task.adaptation_config.adaptation_scope.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'training_time': task.training_time,
            'initial_metrics': task.initial_metrics,
            'final_metrics': task.final_metrics,
            'error_message': task.error_message
        }

    async def cancel_adaptation(self, task_id: str) -> None:
        """Cancel an active adaptation task."""
        if task_id not in self.adaptation_tasks:
            raise ModelAdaptationError(f"Adaptation task {task_id} not found")
        
        task = self.adaptation_tasks[task_id]
        
        if task.status == AdaptationStatus.IN_PROGRESS:
            task.status = AdaptationStatus.CANCELLED
            self.active_adaptations.pop(task.model_id, None)
            
            self.logger.info(f"Cancelled adaptation task: {task_id}")

    async def rollback_model(self, model_id: str, snapshot_id: Optional[str] = None) -> None:
        """Rollback model to a previous snapshot."""
        if model_id not in self.model_snapshots:
            raise ModelAdaptationError(f"No snapshots found for model {model_id}")
        
        snapshots = self.model_snapshots[model_id]
        
        if snapshot_id:
            # Find specific snapshot
            snapshot = None
            for s in snapshots:
                if s.snapshot_id == snapshot_id:
                    snapshot = s
                    break
            
            if not snapshot:
                raise ModelAdaptationError(f"Snapshot {snapshot_id} not found")
        else:
            # Use latest snapshot
            snapshot = snapshots[-1]
        
        # Get model and adapter
        model = await self.model_router.get_model(model_id)
        if not model:
            raise ModelAdaptationError(f"Model {model_id} not found")
        
        # Determine model type (this would need to be stored with the model)
        model_type = ModelType.LANGUAGE_MODEL  # Default, should be determined properly
        adapter = self.model_adapters[model_type]
        
        # Perform rollback
        await self._rollback_adaptation(model, snapshot, adapter)

    def list_adaptations(self, model_id: Optional[str] = None, status: Optional[AdaptationStatus] = None) -> List[Dict[str, Any]]:
        """List adaptation tasks with optional filtering."""
        tasks = []
        
        for task in self.adaptation_tasks.values():
            if model_id and task.model_id != model_id:
                continue
            
            if status and task.status != status:
                continue
            
            tasks.append({
                'task_id': task.task_id,
                'model_id': task.model_id,
                'status': task.status.value,
                'adaptation_type': task.adaptation_config.adaptation_type.value,
                'adaptation_scope': task.adaptation_config.adaptation_scope.value,
                'created_at': task.created_at.isoformat(),
                'training_time': task.training_time,
                'progress': task.progress
            })
        
        return tasks

    def get_model_snapshots(self, model_id: str) -> List[Dict[str, Any]]:
        """Get snapshots for a specific model."""
        if model_id not in self.model_snapshots:
            return []
        
        snapshots = []
        for snapshot in self.model_snapshots[model_id]:
            snapshots.append({
                'snapshot_id': snapshot.snapshot_id,
                'model_id': snapshot.model_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'description': snapshot.description,
                'tags': list(snapshot.tags),
                'performance_metrics': snapshot.performance_metrics
            })
        
        return snapshots

    async def _get_user_interaction_history(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get user interaction history for personalization."""
        try:
            # Get from episodic memory
            history = await self.memory_manager.get_user_episodes(
                user_id=user_id,
                session_id=session_id,
                limit=100
            )
            
            return history
            
        except Exception as e:
            self.logger.warning(f"Failed to get user interaction history: {str(e)}")
            return []

    def _extract_training_data_from_interactions(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Optional[Any]:
        """Extract training data from user interactions."""
        if not interactions:
            return None
        
        # This would implement extraction logic based on interaction format
        # For now, return a simplified representation
        training_examples = []
        
        for interaction in interactions:
            if 'user_input' in interaction and 'assistant_response' in interaction:
                training_examples.append({
                    'input': interaction['user_input'],
                    'output': interaction['assistant_response'],
                    'feedback': interaction.get('user_feedback'),
                    'context': interaction.get('context', {})
                })
        
        return training_examples if training_examples else None

    async def _load_adaptation_policies(self) -> None:
        """Load adaptation policies from configuration."""
        try:
            policies = self.config.get("adaptation.policies", {})
            self.adaptation_policies.update(policies)
            
            self.logger.info(f"Loaded {len(self.adaptation_policies)} adaptation policies")
            
        except Exception as e:
            self.logger.warning(f"Failed to load adaptation policies: {str(e)}")

    async def _save_snapshot_to_storage(self, snapshot: ModelSnapshot) -> None:
        """Save snapshot to persistent storage."""
        try:
            if self.database:
                snapshot_data = {
                    'snapshot_id': snapshot.snapshot_id,
                    'model_id': snapshot.model_id,
                    'timestamp': snapshot.timestamp.isoformat(),
                    'model_state': pickle.dumps(snapshot.model_state_dict),
                    'performance_metrics': json.dumps(snapshot.performance_metrics),
                    'description': snapshot.description,
                    'tags': json.dumps(list(snapshot.tags))
                }
                
                if self.encryption:
                    snapshot_data['model_state'] = await self.encryption.encrypt(snapshot_data['model_state'])
                
                await self.database.execute(
                    """
                    INSERT INTO model_snapshots 
                    (snapshot_id, model_id, timestamp, model_state, performance_metrics, description, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    tuple(snapshot_data.values())
                )
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot to storage: {str(e)}")

    async def _cleanup_snapshot(self, snapshot: ModelSnapshot) -> None:
        """Cleanup snapshot files and data."""
        try:
            # Remove from database
            if self.database:
                await self.database.execute(
                    "DELETE FROM model_snapshots WHERE snapshot_id = ?",
                    (snapshot.snapshot_id,)
                )
            
            # Remove file if it exists
            if snapshot.file_path and snapshot.file_path.exists():
                snapshot.file_path.unlink()
            
        except Exception as e:
            self.logger.warning(f"Failed to cleanup snapshot {snapshot.snapshot_id}: {str(e)}")

    async def _adaptation_monitor_loop(self) -> None:
        """Background task to monitor active adaptations."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check for timed out adaptations
                for task_id, task in list(self.adaptation_tasks.items()):
                    if task.status == AdaptationStatus.IN_PROGRESS:
                        # Check for timeout
                        if task.started_at:
                            runtime = (current_time - task.started_at).total_seconds()
                            max_time = task.adaptation_config.max_training_time_seconds
                            
                            if runtime > max_time:
                                task.status = AdaptationStatus.FAILED
                                task.error_message = f"Adaptation timed out after {runtime:.2f}s"
                                self.active_adaptations.pop(task.model_id, None)
                                
                                self.logger.warning(f"Adaptation task {task_id} timed out")
                
                # Update metrics
                self.metrics.set("active_adaptations", len(self.active_adaptations))
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in adaptation monitor: {str(e)}")
                await asyncio.sleep(30)

    async def _performance_monitor_loop(self) -> None:
        """Background task to monitor model performance."""
        while True:
            try:
                # Check performance of all models
                model_ids = await self.model_router.list_models()
                
                for model_id in model_ids:
                    try:
                        # Get current performance metrics
                        metrics = await self._get_model_performance_metrics(model_id)
                        
                        # Check if performance is below threshold
                        if self._is_performance_below_threshold(metrics):
                            await self._trigger_performance_optimization(model_id, metrics)
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to check performance for model {model_id}: {str(e)}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {str(e)}")
                await asyncio.sleep(300)

    async def _auto_adaptation_loop(self) -> None:
        """Background task for automatic adaptations."""
        while True:
            try:
                if not self.auto_adaptation_enabled:
                    await asyncio.sleep(3600)  # Sleep for an hour if disabled
                    continue
                
                # Check for users who need personalized adaptations
                await self._check_personalization_needs()
                
                # Check for domain adaptations
                await self._check_domain_adaptation_needs()
                
                # Determine sleep interval based on frequency setting
                if self.adaptation_frequency == "hourly":
                    sleep_time = 3600
                elif self.adaptation_frequency == "daily":
                    sleep_time = 86400
                else:
                    sleep_time = 3600  # Default to hourly
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in auto adaptation loop: {str(e)}")
                await asyncio.sleep(3600)  # Sleep before retrying
