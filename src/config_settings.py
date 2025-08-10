"""
Base Configuration Settings for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 10:56:43 UTC

This module provides the foundational configuration settings for the AI assistant,
including core system settings, component configurations, service registrations,
dependency injection container setup, and integration with all core subsystems.
"""

import importlib
import inspect
import logging
import multiprocessing
import os
import platform
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union

import asyncio

from src.core.config.loader import ConfigLoader

# Import YAML configuration system
try:
    from src.core.config.yaml_loader import get_config, get_config_section, YamlConfigLoader
    YAML_CONFIG_AVAILABLE = True
except ImportError:
    YAML_CONFIG_AVAILABLE = False
    YamlConfigLoader = None

# Core imports
from src.core.dependency_injection import Container, LifecycleScope
from src.core.error_handling import ErrorHandler
from src.core.events.event_bus import EnhancedEventBus, EventBus
from src.core.health_check import HealthCheck

# Security components
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager
from src.core.security.sanitization import SecuritySanitizer
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.redis_cache import RedisCache

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.backup_manager import BackupManager
from src.integrations.storage.database import DatabaseManager

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.model_adaptation import ModelAdapter
from src.learning.preference_learning import PreferenceLearner

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.memory.storage.vector_store import VectorStore
from src.observability.logging.config import get_logger, setup_logging

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.processing.multimodal.cross_modal_attention import CrossModalAttention

# Multimodal processing
from src.processing.multimodal.fusion_strategies import MultimodalFusionStrategy
from src.processing.natural_language.entity_extractor import EnhancedEntityExtractor

# Processing components
from src.processing.natural_language.intent_manager import EnhancedIntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.tokenizer import EnhancedTokenizer

# Speech processing
from src.processing.speech.audio_pipeline import EnhancedAudioPipeline
from src.processing.speech.emotion_detection import EnhancedEmotionDetector
from src.processing.speech.speaker_recognition import EnhancedSpeakerRecognition
from src.processing.speech.speech_to_text import EnhancedWhisperTranscriber
from src.processing.speech.text_to_speech import EnhancedTextToSpeech
from src.processing.vision.face_recognition import FaceRecognition
from src.processing.vision.image_analyzer import ImageAnalyzer
from src.processing.vision.ocr_engine import OCREngine

# Vision processing
from src.processing.vision.vision_processor import VisionProcessor
from src.reasoning.decision_making.decision_tree import DecisionTree
from src.reasoning.inference_engine import InferenceEngine
from src.reasoning.knowledge_graph import KnowledgeGraph

# Reasoning components
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.planning.task_planner import TaskPlanner

# Skills management
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator

# Assistant core components - Commented out to fix circular dependencies
# NOTE: Assistant component configuration moved to application layer initialization
# This prevents circular import issues during module loading
# Components are registered during application startup in main.py or app initialization
# from src.assistant.core import CoreAssistantEngine, EngineConfiguration
# from src.assistant.core import EnhancedComponentManager
# from src.assistant.core import WorkflowOrchestrator
# from src.assistant.core import EnhancedSessionManager
# from src.assistant.core import InteractionHandler
# from src.assistant.core import EnhancedPluginManager











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
    plugin_directories: List[str] = field(
        default_factory=lambda: ["plugins/", "src/plugins/", "data/plugins/"]
    )
    sandbox_enabled: bool = True
    plugin_timeout: float = 30.0


class BaseSettings:
    """
    Base configuration settings for the AI Assistant system.

    This class provides the foundational configuration and dependency injection
    setup for the entire AI assistant system, including:

    - Core system configuration loaded from YAML files
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

        # Initialize YAML configuration loader if available
        if YAML_CONFIG_AVAILABLE and YamlConfigLoader:
            self.yaml_loader = YamlConfigLoader(environment.value)
            self.yaml_config = self.yaml_loader.load()
        else:
            self.yaml_loader = None
            self.yaml_config = {}

        # Core settings from YAML or defaults
        app_config = self.yaml_config.get("app", {})
        self.app_name = app_config.get("name", "AI Assistant")
        self.app_version = app_config.get("version", "1.0.0")
        self.app_description = app_config.get("description", "Advanced AI Assistant with Multimodal Capabilities")

        # System information
        self.system_info = self._get_system_info()

        # Load environment variables
        self._load_environment_variables()

        # Initialize configurations from YAML
        self.database = self._create_database_config()
        self.cache = self._create_cache_config()
        self.security = self._create_security_config()
        self.monitoring = self._create_monitoring_config()
        self.processing = self._create_processing_config()
        self.memory = self._create_memory_config()
        self.learning = self._create_learning_config()
        self.plugins = self._create_plugins_config()

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

        self.logger.info(f"BaseSettings initialized for {environment.value} environment using YAML configuration")

    def _create_database_config(self) -> DatabaseConfig:
        """Create database configuration from YAML."""
        if self.yaml_config:
            storage_config = self.yaml_config.get("integrations", {}).get("storage", {})
            db_config = storage_config.get("database", {})
        else:
            db_config = {}
        
        return DatabaseConfig(
            url=db_config.get("url", "sqlite:///data/assistant.db"),
            pool_size=db_config.get("pool_size", 10),
            max_overflow=db_config.get("max_overflow", 20),
            pool_pre_ping=True,
            pool_recycle=db_config.get("pool_recycle", 300),
            echo=db_config.get("echo", False),
            echo_pool=False,
            migration_dir="migrations",
            backup_enabled=True,
            backup_interval=3600,
        )

    def _create_cache_config(self) -> CacheConfig:
        """Create cache configuration from YAML."""
        cache_config = self.yaml_config.get("integrations", {}).get("cache", {})
        redis_config = cache_config.get("redis", {})
        
        # Build Redis URL from components
        redis_url = f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}/{redis_config.get('db', 0)}"
        
        return CacheConfig(
            redis_url=redis_url,
            default_ttl=cache_config.get("default_ttl", 3600),
            max_connections=redis_config.get("max_connections", 10),
            socket_timeout=redis_config.get("socket_timeout", 5.0),
            socket_connect_timeout=redis_config.get("socket_connect_timeout", 5.0),
            retry_on_timeout=redis_config.get("retry_on_timeout", True),
            health_check_interval=redis_config.get("health_check_interval", 30.0),
            enabled=cache_config.get("enabled", True),
        )

    def _create_security_config(self) -> SecurityConfig:
        """Create security configuration from YAML."""
        security_config = self.yaml_config.get("security", {})
        auth_config = security_config.get("authentication", {})
        api_auth_config = self.yaml_config.get("api", {}).get("rest", {}).get("authentication", {})
        
        return SecurityConfig(
            secret_key=api_auth_config.get("jwt_secret", str(uuid.uuid4())),
            encryption_enabled=security_config.get("encryption", {}).get("enabled", True),
            authentication_required=auth_config.get("enabled", True),
            authorization_enabled=security_config.get("authorization", {}).get("enabled", True),
            session_timeout=auth_config.get("session_timeout", 3600),
            max_login_attempts=auth_config.get("max_login_attempts", 5),
            password_min_length=8,
            jwt_algorithm="HS256",
            jwt_expiration=api_auth_config.get("token_expiry", 3600),
            sanitization_enabled=security_config.get("sanitization", {}).get("enabled", True),
            audit_logging=security_config.get("audit", {}).get("enabled", True),
        )

    def _create_monitoring_config(self) -> MonitoringConfig:
        """Create monitoring configuration from YAML."""
        observability_config = self.yaml_config.get("observability", {})
        metrics_config = observability_config.get("metrics", {})
        tracing_config = observability_config.get("tracing", {})
        health_config = observability_config.get("health_checks", {})
        
        return MonitoringConfig(
            metrics_enabled=metrics_config.get("enabled", True),
            tracing_enabled=tracing_config.get("enabled", True),
            profiling_enabled=observability_config.get("profiling", {}).get("enabled", False),
            health_check_interval=health_config.get("interval", 30.0),
            metrics_port=metrics_config.get("port", 9090),
            metrics_path=metrics_config.get("path", "/metrics"),
            jaeger_endpoint=tracing_config.get("jaeger_endpoint"),
            prometheus_pushgateway=None,
            log_sampling_rate=tracing_config.get("sampling_rate", 1.0),
        )

    def _create_processing_config(self) -> ProcessingConfig:
        """Create processing configuration from YAML."""
        core_config = self.yaml_config.get("core", {}).get("engine", {})
        processing_config = self.yaml_config.get("processing", {})
        
        return ProcessingConfig(
            enable_speech_processing=core_config.get("enable_speech_processing", True),
            enable_vision_processing=core_config.get("enable_vision_processing", True),
            enable_multimodal_fusion=core_config.get("enable_multimodal_fusion", True),
            enable_reasoning=core_config.get("enable_reasoning", True),
            enable_learning=core_config.get("enable_learning", True),
            max_concurrent_requests=core_config.get("max_concurrent_requests", 10),
            default_timeout=core_config.get("default_timeout_seconds", 30.0),
            request_queue_size=100,
            default_quality=core_config.get("default_quality_level", "balanced"),
            adaptive_quality=core_config.get("enable_adaptive_quality", True),
            default_voice="neural",
            speech_model=processing_config.get("speech", {}).get("whisper_model", "base"),
            tts_model="tacotron2",
            vision_model="clip-vit-base",
            ocr_enabled=True,
            face_recognition_enabled=False,
        )

    def _create_memory_config(self) -> MemoryConfig:
        """Create memory configuration from YAML."""
        memory_config = self.yaml_config.get("memory", {})
        working_memory = memory_config.get("working_memory", {})
        context_config = memory_config.get("context", {})
        semantic_memory = memory_config.get("semantic_memory", {})
        
        return MemoryConfig(
            working_memory_size=working_memory.get("capacity", 1000),
            context_window_size=context_config.get("max_context_size", 4096),
            episodic_memory_retention=memory_config.get("episodic_memory", {}).get("retention_days", 30),
            semantic_memory_threshold=semantic_memory.get("similarity_threshold", 0.7),
            consolidation_interval=memory_config.get("episodic_memory", {}).get("consolidation_interval", 3600),
            vector_store_type=memory_config.get("vector_store", {}).get("backend", "faiss"),
            embedding_model=semantic_memory.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            memory_compression=working_memory.get("compression", True),
        )

    def _create_learning_config(self) -> LearningConfig:
        """Create learning configuration from YAML."""
        learning_config = self.yaml_config.get("learning", {})
        continual_learning = learning_config.get("continual_learning", {})
        preference_learning = learning_config.get("preference_learning", {})
        feedback_processing = learning_config.get("feedback_processing", {})
        model_adaptation = learning_config.get("model_adaptation", {})
        
        return LearningConfig(
            continual_learning_enabled=continual_learning.get("enabled", True),
            preference_learning_enabled=preference_learning.get("enabled", True),
            feedback_processing_enabled=feedback_processing.get("enabled", True),
            model_adaptation_enabled=model_adaptation.get("enabled", True),
            learning_rate=continual_learning.get("learning_rate", 0.001),
            adaptation_threshold=0.1,
            feedback_weight=0.5,
            preference_weight=preference_learning.get("preference_decay", 0.3),
            learning_update_interval=300,
            model_save_interval=3600,
        )

    def _create_plugins_config(self) -> PluginConfig:
        """Create plugins configuration from YAML."""
        plugins_config = self.yaml_config.get("core", {}).get("plugins", {})
        
        return PluginConfig(
            enabled=True,
            auto_discovery=plugins_config.get("auto_discovery", True),
            hot_reload=plugins_config.get("hot_reload", False),
            security_validation=plugins_config.get("security_validation", True),
            max_plugins=plugins_config.get("max_plugins", 100),
            plugin_directories=plugins_config.get("directories", ["plugins/", "src/plugins/", "data/plugins/"]),
            sandbox_enabled=plugins_config.get("sandbox_enabled", True),
            plugin_timeout=plugins_config.get("plugin_timeout", 30.0),
        )

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
            "machine": platform.machine(),
        }

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        # The YAML loader handles environment variable interpolation automatically
        # This method now provides additional environment-specific overrides
        
        # Environment-specific overrides
        if self.environment == Environment.PRODUCTION:
            # Production overrides from environment
            if os.getenv("DATABASE_URL"):
                # Update YAML config with environment override
                if "integrations" not in self.yaml_config:
                    self.yaml_config["integrations"] = {}
                if "storage" not in self.yaml_config["integrations"]:
                    self.yaml_config["integrations"]["storage"] = {}
                if "database" not in self.yaml_config["integrations"]["storage"]:
                    self.yaml_config["integrations"]["storage"]["database"] = {}
                self.yaml_config["integrations"]["storage"]["database"]["url"] = os.getenv("DATABASE_URL")
                
        elif self.environment == Environment.DEVELOPMENT:
            # Development overrides
            pass  # Development settings are in config.dev.yaml

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = LogLevel.DEBUG if self.environment == Environment.DEVELOPMENT else LogLevel.INFO

        setup_logging(
            level=log_level.value,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_file=f"data/logs/{self.environment.value}.log",
            max_bytes=10 * 1024 * 1024,  # 10MB
            backup_count=5,
        )

    def _register_core_services(self) -> None:
        """Register core system services."""
        # Configuration loader
        self.container.register(
            ConfigLoader,
            factory=lambda: ConfigLoader(config_dir="configs/", environment=self.environment.value),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        # Event bus
        self.container.register(
            EventBus, EnhancedEventBus, lifecycle=ComponentLifecycle.SINGLETON.value
        )

        # Error handler
        self.container.register(
            ErrorHandler,
            factory=lambda: ErrorHandler(
                max_retries=3, retry_delays=[1, 2, 4], enable_circuit_breaker=True
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        # Health check
        self.container.register(
            HealthCheck,
            factory=lambda: HealthCheck(
                check_interval=self.monitoring.health_check_interval, timeout=5.0
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

    def _register_assistant_components(self) -> None:
        """Register core assistant components."""
        try:
            # Component manager - dynamic import to avoid circular dependencies
            from src.assistant.core import EnhancedComponentManager

            self.container.register(
                EnhancedComponentManager, lifecycle=ComponentLifecycle.SINGLETON.value
            )
        except ImportError:
            self.logger.warning("EnhancedComponentManager not available")

        try:
            # Session manager
            from src.assistant.core import EnhancedSessionManager

            self.container.register(
                EnhancedSessionManager, lifecycle=ComponentLifecycle.SINGLETON.value
            )
        except ImportError:
            self.logger.warning("EnhancedSessionManager not available")

        try:
            # Workflow orchestrator
            from src.assistant.core import WorkflowOrchestrator

            self.container.register(
                WorkflowOrchestrator, lifecycle=ComponentLifecycle.SINGLETON.value
            )
        except ImportError:
            self.logger.warning("WorkflowOrchestrator not available")

        try:
            # Interaction handler
            from src.assistant.core import InteractionHandler

            self.container.register(
                InteractionHandler, lifecycle=ComponentLifecycle.SINGLETON.value
            )
        except ImportError:
            self.logger.warning("InteractionHandler not available")

        # Plugin manager
        if self.plugins.enabled:
            try:
                from src.assistant.core import EnhancedPluginManager

                self.container.register(
                    EnhancedPluginManager, lifecycle=ComponentLifecycle.SINGLETON.value
                )
            except ImportError:
                self.logger.warning("EnhancedPluginManager not available")

        # Core engine (main orchestrator)
        self.container.register(
            CoreAssistantEngine,
            factory=lambda container: CoreAssistantEngine(
                container,
                EngineConfiguration(
                    enable_speech_processing=self.processing.enable_speech_processing,
                    enable_vision_processing=self.processing.enable_vision_processing,
                    enable_multimodal_fusion=self.processing.enable_multimodal_fusion,
                    enable_reasoning=self.processing.enable_reasoning,
                    enable_learning=self.processing.enable_learning,
                    max_concurrent_requests=self.processing.max_concurrent_requests,
                    default_timeout_seconds=self.processing.default_timeout,
                ),
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

    def _register_processing_components(self) -> None:
        """Register processing pipeline components."""
        # Natural language processing
        self.container.register(EnhancedTokenizer, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(EnhancedIntentManager, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(LanguageChain, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(SentimentAnalyzer, lifecycle=ComponentLifecycle.SINGLETON.value)
        self.container.register(
            EnhancedEntityExtractor, lifecycle=ComponentLifecycle.SINGLETON.value
        )

        # Speech processing
        if self.processing.enable_speech_processing:
            self.container.register(
                EnhancedAudioPipeline, lifecycle=ComponentLifecycle.SINGLETON.value
            )
            self.container.register(
                EnhancedWhisperTranscriber, lifecycle=ComponentLifecycle.SINGLETON.value
            )
            self.container.register(
                EnhancedTextToSpeech, lifecycle=ComponentLifecycle.SINGLETON.value
            )
            self.container.register(
                EnhancedEmotionDetector, lifecycle=ComponentLifecycle.SINGLETON.value
            )
            self.container.register(
                EnhancedSpeakerRecognition, lifecycle=ComponentLifecycle.SINGLETON.value
            )

        # Vision processing
        if self.processing.enable_vision_processing:
            self.container.register(VisionProcessor, lifecycle=ComponentLifecycle.SINGLETON.value)
            self.container.register(ImageAnalyzer, lifecycle=ComponentLifecycle.SINGLETON.value)

            if self.processing.ocr_enabled:
                self.container.register(OCREngine, lifecycle=ComponentLifecycle.SINGLETON.value)

            if self.processing.face_recognition_enabled:
                self.container.register(
                    FaceRecognition, lifecycle=ComponentLifecycle.SINGLETON.value
                )

        # Multimodal processing
        if self.processing.enable_multimodal_fusion:
            self.container.register(
                MultimodalFusionStrategy, lifecycle=ComponentLifecycle.SINGLETON.value
            )
            self.container.register(
                CrossModalAttention, lifecycle=ComponentLifecycle.SINGLETON.value
            )

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
                dimension=384,  # Default for MiniLM
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        # Memory types
        self.container.register(
            WorkingMemory,
            factory=lambda: WorkingMemory(max_size=self.memory.working_memory_size, ttl=3600),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        self.container.register(
            EpisodicMemory,
            factory=lambda: EpisodicMemory(retention_days=self.memory.episodic_memory_retention),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        self.container.register(
            SemanticMemory,
            factory=lambda: SemanticMemory(
                similarity_threshold=self.memory.semantic_memory_threshold
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

    def _register_learning_systems(self) -> None:
        """Register learning and adaptation systems."""
        if self.processing.enable_learning:
            self.container.register(
                ContinualLearner,
                factory=lambda: ContinualLearner(
                    learning_rate=self.learning.learning_rate,
                    adaptation_threshold=self.learning.adaptation_threshold,
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value,
            )

            self.container.register(
                PreferenceLearner,
                factory=lambda: PreferenceLearner(
                    preference_weight=self.learning.preference_weight
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value,
            )

            self.container.register(
                FeedbackProcessor,
                factory=lambda: FeedbackProcessor(feedback_weight=self.learning.feedback_weight),
                lifecycle=ComponentLifecycle.SINGLETON.value,
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
                default_ttl=self.cache.default_ttl, enabled=self.cache.enabled
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        # Redis cache
        if self.cache.enabled:
            self.container.register(
                RedisCache,
                factory=lambda: RedisCache(
                    url=self.cache.redis_url,
                    max_connections=self.cache.max_connections,
                    socket_timeout=self.cache.socket_timeout,
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value,
            )

        # Database manager
        self.container.register(
            DatabaseManager,
            factory=lambda: DatabaseManager(
                url=self.database.url,
                pool_size=self.database.pool_size,
                max_overflow=self.database.max_overflow,
                echo=self.database.echo,
            ),
            lifecycle=ComponentLifecycle.SINGLETON.value,
        )

        # Backup manager
        if self.database.backup_enabled:
            self.container.register(
                BackupManager,
                factory=lambda: BackupManager(backup_interval=self.database.backup_interval),
                lifecycle=ComponentLifecycle.SINGLETON.value,
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
                    jwt_expiration=self.security.jwt_expiration,
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value,
            )

        if self.security.authorization_enabled:
            self.container.register(
                AuthorizationManager, lifecycle=ComponentLifecycle.SINGLETON.value
            )

        if self.security.encryption_enabled:
            self.container.register(
                EncryptionManager,
                factory=lambda: EncryptionManager(secret_key=self.security.secret_key),
                lifecycle=ComponentLifecycle.SINGLETON.value,
            )

        if self.security.sanitization_enabled:
            self.container.register(SecuritySanitizer, lifecycle=ComponentLifecycle.SINGLETON.value)

        # Observability
        if self.monitoring.metrics_enabled:
            self.container.register(
                MetricsCollector,
                factory=lambda: MetricsCollector(
                    port=self.monitoring.metrics_port, path=self.monitoring.metrics_path
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value,
            )

        if self.monitoring.tracing_enabled:
            self.container.register(
                TraceManager,
                factory=lambda: TraceManager(
                    jaeger_endpoint=self.monitoring.jaeger_endpoint,
                    sampling_rate=self.monitoring.log_sampling_rate,
                ),
                lifecycle=ComponentLifecycle.SINGLETON.value,
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
            core_engine = self.container.get(CoreAssistantEngine)
            await core_engine.initialize()

            # Initialize component manager
            component_manager = self.container.get(EnhancedComponentManager)
            await component_manager.initialize_all()

            # Initialize plugin manager if enabled
            if self.plugins.enabled:
                try:
                    plugin_manager = self.container.get("EnhancedPluginManager")
                    await plugin_manager.initialize()
                except Exception as e:
                    self.logger.warning(f"Plugin manager initialization failed: {e}")

            self.logger.info("AI assistant system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise

    async def cleanup_system(self) -> None:
        """Cleanup the AI assistant system."""
        try:
            self.logger.info("Starting AI assistant system cleanup...")

            # Shutdown core engine
            core_engine = self.container.get(CoreAssistantEngine)
            await core_engine.shutdown()

            # Shutdown component manager
            component_manager = self.container.get(EnhancedComponentManager)
            await component_manager.shutdown_all()

            # Shutdown plugin manager if enabled
            if self.plugins.enabled:
                try:
                    plugin_manager = self.container.get("EnhancedPluginManager")
                    await plugin_manager.shutdown()
                except Exception as e:
                    self.logger.warning(f"Plugin manager shutdown failed: {e}")

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
        # Return the YAML configuration with some computed values
        config_dict = self.yaml_config.copy()
        
        # Add computed system information
        config_dict["system"] = self.system_info
        
        # Add legacy configuration structure for backward compatibility
        config_dict.update({
            "app": {
                "name": self.app_name,
                "version": self.app_version,
                "description": self.app_description,
                "environment": self.environment.value,
            },
            "database": self.database.__dict__,
            "cache": self.cache.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if k != "secret_key"},
            "monitoring": self.monitoring.__dict__,
            "processing": self.processing.__dict__,
            "memory": self.memory.__dict__,
            "learning": self.learning.__dict__,
            "plugins": self.plugins.__dict__,
        })
        
        return config_dict

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
    # Testing settings will use config.testing.yaml if it exists, otherwise defaults
    return BaseSettings(Environment.TESTING)


def create_staging_settings() -> BaseSettings:
    """Create staging environment settings."""
    return BaseSettings(Environment.STAGING)


def create_production_settings() -> BaseSettings:
    """Create production environment settings."""
    return BaseSettings(Environment.PRODUCTION)


# Global settings factory
def get_settings(environment: Optional[str] = None) -> BaseSettings:
    """
    Get settings for the specified environment.

    Args:
        environment: Environment name (defaults to ENVIRONMENT environment variable)

    Returns:
        BaseSettings instance for the environment
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()

    env_mapping = {
        "development": Environment.DEVELOPMENT,
        "testing": Environment.TESTING,
        "staging": Environment.STAGING,
        "production": Environment.PRODUCTION,
    }

    env_enum = env_mapping.get(environment, Environment.DEVELOPMENT)
    return BaseSettings(env_enum)
