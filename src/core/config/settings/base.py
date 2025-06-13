"""
Base Configuration Settings for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 10:56:43 UTC

This module provides the foundational configuration settings for the AI assistant,
including core system settings, component configurations, service registrations,
dependency injection container setup, and integration with all core subsystems.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Type, Callable, ClassVar
from dataclasses import dataclass, field
from enum import Enum
import importlib
import inspect
from datetime import datetime, timezone, timedelta
import uuid
import platform
import multiprocessing
import asyncio
from contextlib import asynccontextmanager

# Core imports
from src.core.dependency_injection import Container, Singleton, Transient, Scoped
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus, EnhancedEventBus
from src.core.error_handling import ErrorHandler
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger, setup_logging

# Security components
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager
from src.core.security.sanitization import SecuritySanitizer

# Assistant core components
from src.assistant.core_engine import EnhancedCoreEngine, EngineConfiguration
from src.assistant.component_manager import EnhancedComponentManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.interaction_handler import InteractionHandler
from src.assistant.plugin_manager import EnhancedPluginManager

# Processing components
from src.processing.natural_language.intent_manager import EnhancedIntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EnhancedEntityExtractor
from src.processing.natural_language.tokenizer import EnhancedTokenizer

# Speech processing
from src.processing.speech.audio_pipeline import EnhancedAudioPipeline
from src.processing.speech.speech_to_text import EnhancedWhisperTranscriber
from src.processing.speech.text_to_speech import EnhancedTextToSpeech
from src.processing.speech.emotion_detection import EnhancedEmotionDetector
from src.processing.speech.speaker_recognition import EnhancedSpeakerRecognition

# Vision processing
from src.processing.vision.vision_processor import VisionProcessor
from src.processing.vision.image_analyzer import ImageAnalyzer
from src.processing.vision.ocr_engine import OCREngine
from src.processing.vision.face_recognition import FaceRecognition

# Multimodal processing
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy
from src.processing.multimodal.cross_modal_attention import CrossModalAttention

# Reasoning components
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.reasoning.inference_engine import InferenceEngine
from src.reasoning.planning.task_planner import TaskPlanner
from src.reasoning.decision_making.decision_tree import DecisionTree

# Skills management
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.storage.vector_store import VectorStore
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.model_adaptation import ModelAdapter

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.database import DatabaseManager
from src.integrations.storage.backup_manager import BackupManager


class Environment(Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentLifecycle(Enum):
    """Component lifecycle management types."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    LAZY = "lazy"


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///data/assistant.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_pre_ping: bool = True
    pool_recycle: int = 300
    echo: bool = False
    echo_pool: bool = False
    migration_dir: str = "migrations"
    backup_enabled: bool = True
    backup_interval: int = 3600  # seconds


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    redis_url: str = "redis://localhost:6379/0"
    default_ttl: int = 3600
    max_connections: int = 10
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: float = 30.0
    enabled: bool = True


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    encryption_enabled: bool = True
    authentication_required: bool = True
    authorization_enabled: bool = True
    session_timeout: int = 3600
    max_login_attempts: int = 5
    password_min_length: int = 8
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    sanitization_enabled: bool = True
    audit_logging: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    profiling_enabled: bool = False
    health_check_interval: float = 30.0
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    jaeger_endpoint: Optional[str] = None
    prometheus_pushgateway: Optional[str] = None
    log_sampling_rate: float = 1.0


@dataclass
class ProcessingConfig:
    """Processing pipeline configuration."""
    enable_speech_processing: bool = True
    enable_vision_processing: bool = True
    enable_multimodal_fusion: bool = True
    enable_reasoning: bool = True
    enable_learning: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 10
    default_timeout: float = 30.0
    request_queue_size: int = 100
    
    # Quality settings
    default_quality: str = "balanced"  # fast, balanced, quality
    adaptive_quality: bool = True
    
    # Speech settings
    default_voice: str = "neural"
    speech_model: str = "whisper-base"
    tts_model: str = "tacotron2"
    
    # Vision settings
    vision_model: str = "clip-vit-base"
    ocr_enabled: bool = True
    face_recognition_enabled: bool = False


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    working_memory_size: int = 1000
    context_window_size: int = 4096
    episodic_memory_retention: int = 30  # days
    semantic_memory_threshold: float = 0.7
    consolidation_interval: int = 3600  # seconds
    vector_store_type: str = "faiss"  # faiss, pinecone, weaviate
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    memory_compression: bool = True


@dataclass
class LearningConfig:
    """Learning and adaptation configuration."""
    continual_learning_enabled: bool = True
    preference_learning_enabled: bool = True
    feedback_processing_enabled: bool = True
    model_adaptation_enabled: bool = False
    
    # Learning rates and parameters
    learning_rate: float = 0.001
    adaptation_threshold: float = 0.1
    feedback_weight: float = 0.5
    preference_weight: float = 0.3
    
    # Update intervals
    learning_update_interval: int = 300  # seconds
    model_save_interval: int = 3600  # seconds


@dataclass
class PluginConfig:
    """Plugin system configuration."""
    enabled: bool = True
    auto_discovery: bool = True
    hot_reload: bool = False
    security_validation: bool = True
    max_plugins: int = 100
    plugin_directories: List[str] = field(default_factory=lambda: [
        "plugins/",
        "src/plugins/",
        "data/plugins/"
    ])
    sandbox_enabled: bool = True
    plugin_timeout: float = 30.0


class BaseSettings:
    """
    Base configuration settings for the AI Assistant system.
    
    This class provides the foundational configuration and dependency injection
    setup for the entire AI assistant system, including:
    
    - Core system configuration
    - Component registration and lifecycle management
    - Service dependency resolution
    - Environment-specific settings
    - Integration with all subsystems
    - Performance and monitoring settings
    - Security and authentication configuration
    """

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        """
        Initialize base settings.
        
        Args:
            environment: Application environment
        """
        self.environment = environment
        self.logger = get_logger(__name__)
        
        # Core settings
        self.app_name = "AI Assistant"
        self.app_version = "1.0.0"
        self.app_description = "Advanced AI Assistant with Multimodal Capabilities"
        
        # System information
        self.system_info = self._get_system_info()
        
        # Load environment variables
        self._load_environment_variables()
        
        # Initialize configurations
        self.database = DatabaseConfig()
        self.cache = CacheConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.processing = ProcessingConfig()
        self.memory = MemoryConfig()
        self.learning = LearningConfig()
        self.plugins = PluginConfig()
        
        # Initialize dependency injection container
        self.container = Container()
        
        # Setup logging
        self._setup_logging()
        
        # Register core services
        self._register_core_services()
        self._register_assistant_components()
        self._register_processing_components()
        self._register_reasoning_components()
        self._register_memory_systems()
        self._register_learning_systems()
        self._register_integrations()
        
        self.logger.info(f"BaseSettings initialized for {environment.value} environment")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": multiprocessing.cpu_count(),
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine()
        }

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        # Database
        self.database.url = os.getenv("DATABASE_URL", self.database.url)
        
        # Cache
        self.cache.redis_url = os.getenv("REDIS_URL", self.cache.redis_url)
        
        # Security
        if secret_key := os.getenv("SECRET_KEY"):
            self.security.secret_key = secret_key
        
        # Processing
        self.processing.max_concurrent_requests = int(
            os.getenv("MAX_CONCURRENT_REQUESTS", self.processing.max_concurrent_requests)
        )
        
        # Environment-specific overrides
        if self.environment == Environment.PRODUCTION:
            self.database.echo = False
            self.monitoring.profiling_enabled = False
            self.plugins.hot_reload = False
            self.security.authentication_required = True
        elif self.environment == Environment.DEVELOPMENT:
            self.database.echo = True
            self.monitoring.profiling_enabled = True
            self.plugins.hot_reload = True
            self.security.authentication_required = False

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = LogLevel.DEBUG if self.environment == Environment.DEVELOPMENT else LogLevel.INFO
        
        setup_logging(
            level=log_level.value,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_file=f"data/logs/{self.environment.value}.log",
            max_bytes=10*1024*1024,  # 10MB
            backup_count=5
        )

    def _register_core_services(self) -> None:
        """Register core system services."""
        # Configuration loader
        self.container.register(
            ConfigLoader,
            factory=lambda: ConfigLoader(
                config_dir="configs/",
                environment=self.environment.value
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Event bus
        self.container.register(
            EventBus,
            EnhancedEventBus,
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Error handler
        self.container.register(
            ErrorHandler,
            factory=lambda: ErrorHandler(
                max_retries=3,
                retry_delays=[1, 2, 4],
                enable_circuit_breaker=True
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Health check
        self.container.register(
            HealthCheck,
            factory=lambda: HealthCheck(
                check_interval=self.monitoring.health_check_interval,
                timeout=5.0
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )

    def _register_assistant_components(self) -> None:
        """Register core assistant components."""
        # Component manager
        self.container.register(
            EnhancedComponentManager,
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Session manager
        self.container.register(
            EnhancedSessionManager,
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Workflow orchestrator
        self.container.register(
            WorkflowOrchestrator,
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Interaction handler
        self.container.register(
            InteractionHandler,
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Plugin manager
        if self.plugins.enabled:
            self.container.register(
                EnhancedPluginManager,
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
        
        # Core engine (main orchestrator)
        self.container.register(
            EnhancedCoreEngine,
            factory=lambda container: EnhancedCoreEngine(
                container,
                EngineConfiguration(
                    enable_speech_processing=self.processing.enable_speech_processing,
                    enable_vision_processing=self.processing.enable_vision_processing,
                    enable_multimodal_fusion=self.processing.enable_multimodal_fusion,
                    enable_reasoning=self.processing.enable_reasoning,
                    enable_learning=self.processing.enable_learning,
                    max_concurrent_requests=self.processing.max_concurrent_requests,
                    default_timeout_seconds=self.processing.default_timeout
                )
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )

    def _register_processing_components(self) -> None:
        """Register processing pipeline components."""
        # Natural language processing
        self.container.register(EnhancedTokenizer, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(EnhancedIntentManager, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(LanguageChain, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(SentimentAnalyzer, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(EnhancedEntityExtractor, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Speech processing
        if self.processing.enable_speech_processing:
            self.container.register(EnhancedAudioPipeline, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(EnhancedWhisperTranscriber, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(EnhancedTextToSpeech, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(EnhancedEmotionDetector, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(EnhancedSpeakerRecognition, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Vision processing
        if self.processing.enable_vision_processing:
            self.container.register(VisionProcessor, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(ImageAnalyzer, lifecycle=ComponentLifecycle.SINGLETON.value)
            
            if self.processing.ocr_enabled:
                self.container.register(OCREngine, lifecycle=ComponentLifecycle.SINGLETON.value)
            
            if self.processing.face_recognition_enabled:
                self.container.register(FaceRecognition, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Multimodal processing
        if self.processing.enable_multimodal_fusion:
            self.container.register(MultimodalFusionStrategy, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(CrossModalAttention, lifecycle=ComponentLifecycle.SINGLETON.value)

    def _register_reasoning_components(self) -> None:
        """Register reasoning and planning components."""
        if self.processing.enable_reasoning:
            self.container.register(LogicEngine, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(KnowledgeGraph, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(InferenceEngine, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(TaskPlanner, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(DecisionTree, lifecycle=ComponentLifecycle.SINGLETON.value)

    def _register_memory_systems(self) -> None:
        """Register memory management systems."""
        # Core memory components
        self.container.register(MemoryManager, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(ContextManager, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Memory storage
        self.container.register(
            VectorStore,
            factory=lambda: VectorStore(
                store_type=self.memory.vector_store_type,
                embedding_model=self.memory.embedding_model,
                dimension=384  # Default for MiniLM
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Memory types
        self.container.register(
            WorkingMemory,
            factory=lambda: WorkingMemory(
                max_size=self.memory.working_memory_size,
                ttl=3600
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        self.container.register(
            EpisodicMemory,
            factory=lambda: EpisodicMemory(
                retention_days=self.memory.episodic_memory_retention
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        self.container.register(
            SemanticMemory,
            factory=lambda: SemanticMemory(
                similarity_threshold=self.memory.semantic_memory_threshold
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )

    def _register_learning_systems(self) -> None:
        """Register learning and adaptation systems."""
        if self.processing.enable_learning:
            self.container.register(
                ContinualLearner,
                factory=lambda: ContinualLearner(
                    learning_rate=self.learning.learning_rate,
                    adaptation_threshold=self.learning.adaptation_threshold
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
            
            self.container.register(
                PreferenceLearner,
                factory=lambda: PreferenceLearner(
                    preference_weight=self.learning.preference_weight
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
            
            self.container.register(
                FeedbackProcessor,
                factory=lambda: FeedbackProcessor(
                    feedback_weight=self.learning.feedback_weight
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
            
            if self.learning.model_adaptation_enabled:
                self.container.register(ModelAdapter, lifecycle=ComponentLifecycle.SINGLETON.value)

    def _register_integrations(self) -> None:
        """Register integration components."""
        # Model router for LLM integration
        self.container.register(ModelRouter, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Cache strategy
        self.container.register(
            CacheStrategy,
            factory=lambda: CacheStrategy(
                default_ttl=self.cache.default_ttl,
                enabled=self.cache.enabled
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Redis cache
        if self.cache.enabled:
            self.container.register(
                RedisCache,
                factory=lambda: RedisCache(
                    url=self.cache.redis_url,
                    max_connections=self.cache.max_connections,
                    socket_timeout=self.cache.socket_timeout
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
        
        # Database manager
        self.container.register(
            DatabaseManager,
            factory=lambda: DatabaseManager(
                url=self.database.url,
                pool_size=self.database.pool_size,
                max_overflow=self.database.max_overflow,
                echo=self.database.echo
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value
        )
        
        # Backup manager
        if self.database.backup_enabled:
            self.container.register(
                BackupManager,
                factory=lambda: BackupManager(
                    backup_interval=self.database.backup_interval
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
        
        # Skills management
        self.container.register(SkillFactory, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(SkillRegistry, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(SkillValidator, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Security components
        if self.security.authentication_required:
            self.container.register(
                AuthenticationManager,
                factory=lambda: AuthenticationManager(
                    secret_key=self.security.secret_key,
                    jwt_algorithm=self.security.jwt_algorithm,
                    jwt_expiration=self.security.jwt_expiration
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
        
        if self.security.authorization_enabled:
            self.container.register(AuthorizationManager, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        if self.security.encryption_enabled:
            self.container.register(
                EncryptionManager,
                factory=lambda: EncryptionManager(
                    secret_key=self.security.secret_key
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
        
        if self.security.sanitization_enabled:
            self.container.register(SecuritySanitizer, lifecycle=ComponentLifecycle.SINGLETON.value)
        
        # Observability
        if self.monitoring.metrics_enabled:
            self.container.register(
                MetricsCollector,
                factory=lambda: MetricsCollector(
                    port=self.monitoring.metrics_port,
                    path=self.monitoring.metrics_path
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )
        
        if self.monitoring.tracing_enabled:
            self.container.register(
                TraceManager,
                factory=lambda: TraceManager(
                    jaeger_endpoint=self.monitoring.jaeger_endpoint,
                    sampling_rate=self.monitoring.log_sampling_rate
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value
            )

    @asynccontextmanager
    async def application_lifespan(self):
        """
        Application lifespan context manager for proper startup and shutdown.
        
        This ensures all components are properly initialized and cleaned up.
        """
        try:
            # Initialize the system
            await self.initialize_system()
            yield
        finally:
            # Cleanup the system
            await self.cleanup_system()

    async def initialize_system(self) -> None:
        """Initialize the complete AI assistant system."""
        try:
            self.logger.info("Starting AI assistant system initialization...")
            
            # Get core engine and initialize
            core_engine = self.container.get(EnhancedCoreEngine)
            await core_engine.initialize()
            
            # Initialize component manager
            component_manager = self.container.get(EnhancedComponentManager)
            await component_manager.initialize_all()
            
            # Initialize plugin manager if enabled
            if self.plugins.enabled:
                plugin_manager = self.container.get(EnhancedPluginManager)
                await plugin_manager.initialize()
            
            self.logger.info("AI assistant system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise

    async def cleanup_system(self) -> None:
        """Cleanup the AI assistant system."""
        try:
            self.logger.info("Starting AI assistant system cleanup...")
            
            # Shutdown core engine
            core_engine = self.container.get(EnhancedCoreEngine)
            await core_engine.shutdown()
            
            # Shutdown component manager
            component_manager = self.container.get(EnhancedComponentManager)
            await component_manager.shutdown_all()
            
            # Shutdown plugin manager if enabled
            if self.plugins.enabled:
                plugin_manager = self.container.get(EnhancedPluginManager)
                await plugin_manager.shutdown()
            
            self.logger.info("AI assistant system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during system cleanup: {str(e)}")

    def get_component(self, component_type: Type[T]) -> T:
        """
        Get a component from the dependency injection container.
        
        Args:
            component_type: Type of component to retrieve
            
        Returns:
            Component instance
        """
        return self.container.get(component_type)

    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            "app": {
                "name": self.app_name,
                "version": self.app_version,
                "description": self.app_description,
                "environment": self.environment.value
            },
            "system": self.system_info,
            "database": self.database.__dict__,
            "cache": self.cache.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if k != "secret_key"},
            "monitoring": self.monitoring.__dict__,
            "processing": self.processing.__dict__,
            "memory": self.memory.__dict__,
            "learning": self.learning.__dict__,
            "plugins": self.plugins.__dict__
        }

    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate database URL
        if not self.database.url:
            errors.append("Database URL is required")
        
        # Validate Redis URL if cache is enabled
        if self.cache.enabled and not self.cache.redis_url:
            errors.append("Redis URL is required when cache is enabled")
        
        # Validate secret key
        if len(self.security.secret_key) < 32:
            errors.append("Secret key must be at least 32 characters long")
        
        # Validate processing settings
        if self.processing.max_concurrent_requests <= 0:
            errors.append("Max concurrent requests must be positive")
        
        # Validate memory settings
        if self.memory.working_memory_size <= 0:
            errors.append("Working memory size must be positive")
        
        return errors

    def __str__(self) -> str:
        """String representation of settings."""
        return f"BaseSettings(environment={self.environment.value}, app={self.app_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"BaseSettings("
            f"environment={self.environment.value}, "
            f"app_name='{self.app_name}', "
            f"version='{self.app_version}', "
            f"components_registered={len(self.container._services)}"
            f")"
        )


# Environment-specific settings factories
def create_development_settings() -> BaseSettings:
    """Create development environment settings."""
    return BaseSettings(Environment.DEVELOPMENT)


def create_testing_settings() -> BaseSettings:
    """Create testing environment settings."""
    settings = BaseSettings(Environment.TESTING)
    
    # Testing-specific overrides
    settings.database.url = "sqlite:///:memory:"
    settings.cache.enabled = False
    settings.monitoring.profiling_enabled = False
    settings.security.authentication_required = False
    settings.plugins.enabled = False
    
    return settings


def create_staging_settings() -> BaseSettings:
    """Create staging environment settings."""
    settings = BaseSettings(Environment.STAGING)
    
    # Staging-specific overrides
    settings.monitoring.profiling_enabled = True
    settings.security.authentication_required = True
    settings.plugins.hot_reload = False
    
    return settings


def create_production_settings() -> BaseSettings:
    """Create production environment settings."""
    settings = BaseSettings(Environment.PRODUCTION)
    
    # Production-specific overrides
    settings.database.echo = False
    settings.monitoring.profiling_enabled = False
    settings.plugins.hot_reload = False
    settings.plugins.security_validation = True
    settings.security.authentication_required = True
    settings.security.authorization_enabled = True
    settings.security.audit_logging = True
    
    return settings


# Global settings factory
def get_settings(environment: Optional[str] = None) -> BaseSettings:
    """
    Get settings for the specified environment.
    
    Args:
        environment: Environment name (defaults to ENV environment variable)
        
    Returns:
        BaseSettings instance for the environment
    """
    if environment is None:
        environment = os.getenv("ENV", "development").lower()
    
    factories = {
        "development": create_development_settings,
        "testing": create_testing_settings,
        "staging": create_staging_settings,
        "production": create_production_settings
    }
    
    factory = factories.get(environment, create_development_settings)
    return factory()
