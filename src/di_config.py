"""
Dependency Injection Configuration
Configures the DI container with all AI assistant components
"""

from typing import Any, Dict, Type
import asyncio

from src.core.dependency_injection import Container, LifecycleScope
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.error_handling import ErrorHandler
from src.core.health_check import HealthCheck

# Core components
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.session_memory_integrator import SessionMemoryIntegrator
from src.assistant.core_engine import EnhancedCoreEngine
from src.assistant.plugin_manager import EnhancedPluginManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.component_manager import EnhancedComponentManager
from src.assistant.interaction_handler import InteractionHandler

# Memory components
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory
from src.memory.storage.vector_store import VectorStore
from src.memory.operations.context_manager import ContextManager, MemoryContextManager
from src.memory.operations.retrieval import MemoryRetriever

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor

# Skills and reasoning
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.knowledge_graph import KnowledgeGraph

# Integrations
from src.integrations.llm.model_router import ModelRouter
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Security
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import SecuritySanitizer
from src.core.security.encryption import EncryptionManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


class ComponentConfiguration:
    """Configuration for AI assistant components."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    async def configure_container(self, container: Container) -> None:
        """Configure the dependency injection container with all components."""
        
        # Core services (singletons)
        self._register_core_services(container)
        
        # Processing components
        self._register_processing_components(container)
        
        # Memory system
        self._register_memory_components(container)
        
        # Assistant components
        self._register_assistant_components(container)
        
        # Reasoning and skills
        self._register_reasoning_components(container)
        
        # Integrations
        self._register_integration_components(container)
        
        # Security
        self._register_security_components(container)
        
        # Observability
        self._register_observability_components(container)
        
        self.logger.info("Dependency injection container configured successfully")
    
    def _register_core_services(self, container: Container) -> None:
        """Register core services."""
        
        # Configuration loader (singleton)
        container.register(
            ConfigLoader,
            factory=lambda: ConfigLoader(),
            scope=LifecycleScope.SINGLETON
        )
        
        # Event bus (singleton)
        container.register(
            EventBus,
            factory=lambda: EventBus(),
            scope=LifecycleScope.SINGLETON
        )
        
        # Error handler (singleton)
        container.register(
            ErrorHandler,
            factory=lambda: ErrorHandler(container.get(EventBus)),
            scope=LifecycleScope.SINGLETON
        )
        
        # Health check (singleton)
        container.register(
            HealthCheck,
            factory=lambda: HealthCheck(),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_processing_components(self, container: Container) -> None:
        """Register processing components."""
        
        # NLP components
        container.register(
            IntentManager,
            factory=lambda: IntentManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            LanguageChain,
            factory=lambda: LanguageChain(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            SentimentAnalyzer,
            factory=lambda: SentimentAnalyzer(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            EntityExtractor,
            factory=lambda: EntityExtractor(container),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_memory_components(self, container: Container) -> None:
        """Register memory system components."""
        
        # Vector store (singleton)
        container.register(
            VectorStore,
            factory=lambda: VectorStore(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Memory types
        container.register(
            WorkingMemory,
            factory=lambda: WorkingMemory(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            EpisodicMemory,
            factory=lambda: EpisodicMemory(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            SemanticMemory,
            factory=lambda: SemanticMemory(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Context managers
        container.register(
            ContextManager,
            factory=lambda: ContextManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            MemoryContextManager,
            factory=lambda: MemoryContextManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Memory retriever
        container.register(
            MemoryRetriever,
            factory=lambda: MemoryRetriever(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Memory manager (depends on other memory components)
        container.register(
            MemoryManager,
            factory=lambda: MemoryManager(container),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_assistant_components(self, container: Container) -> None:
        """Register assistant components."""
        
        # Component manager (singleton)
        container.register(
            EnhancedComponentManager,
            factory=lambda: EnhancedComponentManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Session memory integrator
        container.register(
            SessionMemoryIntegrator,
            factory=lambda: SessionMemoryIntegrator(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Session manager
        container.register(
            EnhancedSessionManager,
            factory=lambda: EnhancedSessionManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Workflow orchestrator
        container.register(
            WorkflowOrchestrator,
            factory=lambda: WorkflowOrchestrator(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Interaction handler
        container.register(
            InteractionHandler,
            factory=lambda: InteractionHandler(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Plugin manager
        container.register(
            EnhancedPluginManager,
            factory=lambda: EnhancedPluginManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Core engine (depends on many components)
        container.register(
            EnhancedCoreEngine,
            factory=lambda: EnhancedCoreEngine(container),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_reasoning_components(self, container: Container) -> None:
        """Register reasoning and skills components."""
        
        # Skill components
        container.register(
            SkillValidator,
            factory=lambda: SkillValidator(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            SkillRegistry,
            factory=lambda: SkillRegistry(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            SkillFactory,
            factory=lambda: SkillFactory(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Reasoning components
        container.register(
            LogicEngine,
            factory=lambda: LogicEngine(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            KnowledgeGraph,
            factory=lambda: KnowledgeGraph(container),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_integration_components(self, container: Container) -> None:
        """Register integration components."""
        
        # Model router
        container.register(
            ModelRouter,
            factory=lambda: ModelRouter(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Cache strategy
        container.register(
            CacheStrategy,
            factory=lambda: CacheStrategy(container),
            scope=LifecycleScope.SINGLETON
        )
        
        # Database manager
        container.register(
            DatabaseManager,
            factory=lambda: DatabaseManager(container),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_security_components(self, container: Container) -> None:
        """Register security components."""
        
        container.register(
            SecuritySanitizer,
            factory=lambda: SecuritySanitizer(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            AuthenticationManager,
            factory=lambda: AuthenticationManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            AuthorizationManager,
            factory=lambda: AuthorizationManager(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            EncryptionManager,
            factory=lambda: EncryptionManager(container),
            scope=LifecycleScope.SINGLETON
        )
    
    def _register_observability_components(self, container: Container) -> None:
        """Register observability components."""
        
        container.register(
            MetricsCollector,
            factory=lambda: MetricsCollector(container),
            scope=LifecycleScope.SINGLETON
        )
        
        container.register(
            TraceManager,
            factory=lambda: TraceManager(container),
            scope=LifecycleScope.SINGLETON
        )


async def create_configured_container() -> Container:
    """Create and configure a dependency injection container."""
    container = Container()
    
    configurator = ComponentConfiguration()
    await configurator.configure_container(container)
    
    return container


# Mock implementations for missing components (to allow testing)
class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, container: Container):
        self.container = container
        self.logger = get_logger(self.__class__.__name__)
    
    async def initialize(self):
        """Initialize mock component."""
        pass
    
    async def shutdown(self):
        """Shutdown mock component."""
        pass


def create_mock_factories(container: Container) -> None:
    """Create mock factories for components that don't exist yet."""
    
    # Mock missing processing components
    missing_components = [
        'VisionProcessor', 'ImageAnalyzer', 'MultimodalFusionStrategy',
        'EnhancedAudioPipeline', 'EnhancedWhisperTranscriber', 'EnhancedTextToSpeech',
        'EnhancedEmotionDetector', 'EnhancedSpeakerRecognition',
        'TaskPlanner', 'DecisionTree', 'ContinualLearner', 'PreferenceLearner',
        'FeedbackProcessor', 'ModelAdapter'
    ]
    
    for component_name in missing_components:
        # Create mock class
        mock_class = type(component_name, (MockComponent,), {})
        
        # Register with container
        container.register(
            mock_class,
            factory=lambda cls=mock_class: cls(container),
            scope=LifecycleScope.SINGLETON
        )