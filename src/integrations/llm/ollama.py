"""
Advanced Ollama Integration for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 18:11:08 UTC

This module provides comprehensive Ollama LLM integration for the AI assistant,
including model management, streaming capabilities, performance optimization,
health monitoring, and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
import json
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import logging
import aiohttp
import httpx
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ModelLoaded, ModelUnloaded, ModelSwitched, ModelError, ModelHealthChanged,
    ProcessingStarted, ProcessingCompleted, ProcessingError,
    StreamingStarted, StreamingCompleted, StreamingError,
    ComponentHealthChanged, ComponentInitialized, ComponentStarted,
    SystemStateChanged, ErrorOccurred, PerformanceThresholdExceeded
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Assistant components
from src.assistant.component_manager import EnhancedComponentManager, ComponentMetadata, ComponentPriority
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator

# Integrations
from src.integrations.llm.base_provider import BaseLLMProvider, LLMRequest, LLMResponse
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.model_adaptation import ModelAdapter

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


# Type definitions
T = TypeVar('T')


class OllamaState(Enum):
    """Ollama instance states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ModelState(Enum):
    """Model states in Ollama."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"
    PULLING = "pulling"
    CREATING = "creating"


class ResponseMode(Enum):
    """Response generation modes."""
    STREAMING = "streaming"
    BATCH = "batch"
    HYBRID = "hybrid"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for multiple Ollama instances."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RESPONSE_TIME = "response_time"
    MODEL_AFFINITY = "model_affinity"
    RANDOM = "random"


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str
    tag: str = "latest"
    size: int = 0
    modified_at: Optional[datetime] = None
    digest: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    avg_response_time: float = 0.0
    total_requests: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    
    # Resource usage
    memory_usage: float = 0.0
    vram_usage: float = 0.0
    context_length: int = 2048
    
    # Model capabilities
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_code: bool = False
    
    # Quality metrics
    quality_score: float = 0.0
    user_rating: float = 0.0
    performance_rating: float = 0.0


@dataclass
class OllamaInstance:
    """Represents an Ollama instance."""
    instance_id: str
    host: str
    port: int = 11434
    scheme: str = "http"
    
    # State and health
    state: OllamaState = OllamaState.DISCONNECTED
    health_score: float = 0.0
    last_health_check: Optional[datetime] = None
    
    # Performance metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    active_requests: int = 0
    total_requests: int = 0
    error_count: int = 0
    
    # Resource information
    available_memory: float = 0.0
    total_memory: float = 0.0
    gpu_memory: float = 0.0
    cpu_usage: float = 0.0
    
    # Loaded models
    loaded_models: Set[str] = field(default_factory=set)
    model_cache: Dict[str, OllamaModelInfo] = field(default_factory=dict)
    
    # Configuration
    timeout: float = 30.0
    max_concurrent_requests: int = 10
    keep_alive: str = "5m"
    
    @property
    def base_url(self) -> str:
        """Get the base URL for this instance."""
        return f"{self.scheme}://{self.host}:{self.port}"
    
    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    @property
    def load_factor(self) -> float:
        """Calculate load factor (0-1)."""
        if self.max_concurrent_requests == 0:
            return 1.0
        return self.active_requests / self.max_concurrent_requests


@dataclass
class OllamaRequest(LLMRequest):
    """Ollama-specific request configuration."""
    model: str
    stream: bool = False
    raw: bool = False
    keep_alive: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    seed: Optional[int] = None
    num_predict: int = -1
    num_ctx: int = 2048
    
    # Model-specific options
    options: Dict[str, Any] = field(default_factory=dict)
    format: Optional[str] = None  # json, etc.
    
    # Instance selection
    preferred_instance: Optional[str] = None
    exclude_instances: Set[str] = field(default_factory=set)
    
    # Quality requirements
    min_quality_score: float = 0.0
    max_response_time: Optional[float] = None
    
    # Context management
    context: Optional[List[int]] = None
    images: List[str] = field(default_factory=list)


@dataclass
class OllamaResponse(LLMResponse):
    """Ollama-specific response data."""
    model: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    done: bool = False
    
    # Response content
    response: str = ""
    context: Optional[List[int]] = None
    
    # Generation metrics
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0
    
    # Instance information
    instance_id: Optional[str] = None
    model_info: Optional[OllamaModelInfo] = None
    
    # Quality metrics
    quality_score: float = 0.0
    confidence: float = 0.0


class OllamaError(Exception):
    """Custom exception for Ollama operations."""
    
    def __init__(self, message: str, instance_id: Optional[str] = None, 
                 model: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.instance_id = instance_id
        self.model = model
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class OllamaHealthMonitor:
    """Monitors health of Ollama instances and models."""
    
    def __init__(self, logger):
        self.logger = logger
        self.health_checks: Dict[str, asyncio.Task] = {}
        self.health_callbacks: Dict[str, Callable] = {}
    
    async def start_monitoring(self, instance: OllamaInstance, 
                             check_interval: float = 30.0) -> None:
        """Start health monitoring for an instance."""
        if instance.instance_id in self.health_checks:
            self.health_checks[instance.instance_id].cancel()
        
        self.health_checks[instance.instance_id] = asyncio.create_task(
            self._monitor_instance(instance, check_interval)
        )
    
    async def stop_monitoring(self, instance_id: str) -> None:
        """Stop health monitoring for an instance."""
        if instance_id in self.health_checks:
            self.health_checks[instance_id].cancel()
            del self.health_checks[instance_id]
        
        self.health_callbacks.pop(instance_id, None)
    
    async def _monitor_instance(self, instance: OllamaInstance, interval: float) -> None:
        """Monitor instance health continuously."""
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Perform health check
                health_result = await self._check_instance_health(instance)
                
                # Update instance state
                old_state = instance.state
                instance.health_score = health_result.get('score', 0.0)
                instance.last_health_check = datetime.now(timezone.utc)
                
                # Determine new state
                if health_result.get('status') == 'healthy':
                    instance.state = OllamaState.HEALTHY
                elif health_result.get('status') == 'degraded':
                    instance.state = OllamaState.DEGRADED
                else:
                    instance.state = OllamaState.UNHEALTHY
                
                # Emit state change if needed
                if old_state != instance.state:
                    callback = self.health_callbacks.get(instance.instance_id)
                    if callback:
                        await callback(instance, old_state, instance.state)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error for {instance.instance_id}: {str(e)}")
                instance.state = OllamaState.ERROR
    
    async def _check_instance_health(self, instance: OllamaInstance) -> Dict[str, Any]:
        """Check health of a specific instance."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                # Check basic connectivity
                async with session.get(f"{instance.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate health score
                        health_factors = {
                            'connectivity': 1.0,  # Connected successfully
                            'response_time': max(0.0, 1.0 - (instance.avg_response_time / 10.0)),  # Penalize slow responses
                            'error_rate': max(0.0, 1.0 - (instance.error_count / max(instance.total_requests, 1))),
                            'resource_usage': max(0.0, 1.0 - (instance.cpu_usage / 100.0))
                        }
                        
                        health_score = sum(health_factors.values()) / len(health_factors)
                        
                        # Determine status
                        if health_score >= 0.8:
                            status = 'healthy'
                        elif health_score >= 0.5:
                            status = 'degraded'
                        else:
                            status = 'unhealthy'
                        
                        return {
                            'status': status,
                            'score': health_score,
                            'models': data.get('models', []),
                            'factors': health_factors
                        }
                    else:
                        return {'status': 'unhealthy', 'score': 0.0, 'error': f"HTTP {response.status}"}
        
        except Exception as e:
            return {'status': 'unhealthy', 'score': 0.0, 'error': str(e)}


class OllamaLoadBalancer:
    """Load balancer for multiple Ollama instances."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.round_robin_index = 0
        self.logger = get_logger(__name__)
    
    def select_instance(self, instances: List[OllamaInstance], 
                       request: OllamaRequest) -> Optional[OllamaInstance]:
        """Select the best instance for a request."""
        # Filter healthy instances
        healthy_instances = [
            instance for instance in instances 
            if instance.state in [OllamaState.HEALTHY, OllamaState.DEGRADED]
            and instance.instance_id not in request.exclude_instances
        ]
        
        if not healthy_instances:
            return None
        
        # Check for preferred instance
        if request.preferred_instance:
            for instance in healthy_instances:
                if instance.instance_id == request.preferred_instance:
                    return instance
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_selection(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.MODEL_AFFINITY:
            return self._model_affinity_selection(healthy_instances, request.model)
        else:  # RANDOM
            import random
            return random.choice(healthy_instances)
    
    def _round_robin_selection(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Round-robin instance selection."""
        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return instance
    
    def _least_loaded_selection(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Select instance with least load."""
        return min(instances, key=lambda i: i.load_factor)
    
    def _response_time_selection(self, instances: List[OllamaInstance]) -> OllamaInstance:
        """Select instance with best response time."""
        return min(instances, key=lambda i: i.avg_response_time)
    
    def _model_affinity_selection(self, instances: List[OllamaInstance], 
                                 model: str) -> OllamaInstance:
        """Select instance with model already loaded."""
        # Prefer instances with model already loaded
        model_loaded_instances = [i for i in instances if model in i.loaded_models]
        if model_loaded_instances:
            return self._least_loaded_selection(model_loaded_instances)
        
        # Fallback to least loaded
        return self._least_loaded_selection(instances)


class OllamaModelManager:
    """Manages Ollama models across instances."""
    
    def __init__(self, logger):
        self.logger = logger
        self.model_registry: Dict[str, OllamaModelInfo] = {}
        self.model_aliases: Dict[str, str] = {}
        self.model_cache: Dict[str, Dict[str, Any]] = {}
    
    async def discover_models(self, instance: OllamaInstance) -> List[OllamaModelInfo]:
        """Discover available models on an instance."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{instance.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = []
                        
                        for model_data in data.get('models', []):
                            model_info = OllamaModelInfo(
                                name=model_data['name'],
                                size=model_data.get('size', 0),
                                digest=model_data.get('digest'),
                                modified_at=datetime.fromisoformat(
                                    model_data['modified_at'].replace('Z', '+00:00')
                                ) if model_data.get('modified_at') else None,
                                details=model_data.get('details', {})
                            )
                            
                            # Update registry
                            self.model_registry[model_info.name] = model_info
                            models.append(model_info)
                        
                        return models
                    else:
                        raise OllamaError(f"Failed to discover models: HTTP {response.status}")
        
        except Exception as e:
            self.logger.error(f"Model discovery failed for {instance.instance_id}: {str(e)}")
            raise OllamaError(f"Model discovery failed: {str(e)}", instance.instance_id)
    
    async def pull_model(self, instance: OllamaInstance, model_name: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Pull a model to an instance."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"name": model_name}
                
                async with session.post(
                    f"{instance.base_url}/api/pull",
                    json=payload
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode().strip())
                                    yield data
                                except json.JSONDecodeError:
                                    continue
                    else:
                        raise OllamaError(f"Failed to pull model: HTTP {response.status}")
        
        except Exception as e:
            self.logger.error(f"Model pull failed for {model_name}: {str(e)}")
            raise OllamaError(f"Model pull failed: {str(e)}", instance.instance_id, model_name)
    
    async def load_model(self, instance: OllamaInstance, model_name: str) -> bool:
        """Preload a model on an instance."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_name,
                    "prompt": "",  # Empty prompt to just load the model
                    "stream": False
                }
                
                async with session.post(
                    f"{instance.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        instance.loaded_models.add(model_name)
                        return True
                    else:
                        return False
        
        except Exception as e:
            self.logger.error(f"Model loading failed for {model_name}: {str(e)}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[OllamaModelInfo]:
        """Get information about a model."""
        # Check aliases first
        actual_name = self.model_aliases.get(model_name, model_name)
        return self.model_registry.get(actual_name)
    
    def register_model_alias(self, alias: str, actual_name: str) -> None:
        """Register a model alias."""
        self.model_aliases[alias] = actual_name


class EnhancedOllamaProvider(BaseLLMProvider):
    """
    Advanced Ollama LLM Provider for the AI Assistant.
    
    This provider offers comprehensive Ollama integration including:
    - Multi-instance management with load balancing
    - Model lifecycle management and hot-swapping
    - Streaming and batch response modes
    - Performance optimization and caching
    - Health monitoring and auto-recovery
    - Integration with memory and learning systems
    - Event-driven architecture
    - Security and rate limiting
    - Quality assessment and adaptation
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced Ollama provider.
        
        Args:
            container: Dependency injection container
        """
        super().__init__()
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Assistant components
        self.component_manager = container.get(EnhancedComponentManager)
        self.session_manager = container.get(EnhancedSessionManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        
        # Memory and learning
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.preference_learner = container.get(PreferenceLearner)
            self.model_adapter = container.get(ModelAdapter)
        except Exception:
            self.continual_learner = None
            self.preference_learner = None
            self.model_adapter = None
        
        # Caching and storage
        try:
            self.cache_strategy = container.get(CacheStrategy)
            self.database = container.get(DatabaseManager)
        except Exception:
            self.cache_strategy = None
            self.database = None
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Ollama-specific components
        self.instances: Dict[str, OllamaInstance] = {}
        self.model_manager = OllamaModelManager(self.logger)
        self.health_monitor = OllamaHealthMonitor(self.logger)
        self.load_balancer = OllamaLoadBalancer()
        
        # Configuration
        self._load_configuration()
        
        # State management
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Performance tracking
        self.response_cache: Dict[str, OllamaResponse] = {}
        self.performance_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Setup monitoring
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("ollama_provider", self._health_check_callback)
        
        self.logger.info("EnhancedOllamaProvider initialized successfully")

    def _load_configuration(self) -> None:
        """Load Ollama configuration."""
        self.default_model = self.config.get("ollama.default_model", "llama3.2")
        self.max_concurrent_requests = self.config.get("ollama.max_concurrent_requests", 10)
        self.default_timeout = self.config.get("ollama.timeout", 30.0)
        self.enable_caching = self.config.get("ollama.enable_caching", True)
        self.cache_ttl = self.config.get("ollama.cache_ttl", 3600)
        self.load_balancing_strategy = LoadBalancingStrategy(
            self.config.get("ollama.load_balancing", "least_loaded")
        )
        self.auto_model_loading = self.config.get("ollama.auto_model_loading", True)
        self.model_keep_alive = self.config.get("ollama.model_keep_alive", "5m")
        
        # Load instance configurations
        instances_config = self.config.get("ollama.instances", [
            {"host": "localhost", "port": 11434}
        ])
        
        for i, instance_config in enumerate(instances_config):
            instance_id = instance_config.get("id", f"ollama_{i}")
            instance = OllamaInstance(
                instance_id=instance_id,
                host=instance_config["host"],
                port=instance_config.get("port", 11434),
                scheme=instance_config.get("scheme", "http"),
                timeout=instance_config.get("timeout", self.default_timeout),
                max_concurrent_requests=instance_config.get("max_concurrent_requests", 10),
                keep_alive=instance_config.get("keep_alive", self.model_keep_alive)
            )
            self.instances[instance_id] = instance

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register Ollama-specific metrics
            self.metrics.register_counter("ollama_requests_total")
            self.metrics.register_counter("ollama_requests_successful")
            self.metrics.register_counter("ollama_requests_failed")
            self.metrics.register_histogram("ollama_response_duration_seconds")
            self.metrics.register_gauge("ollama_active_instances")
            self.metrics.register_gauge("ollama_loaded_models")
            self.metrics.register_counter("ollama_model_switches")
            self.metrics.register_histogram("ollama_model_load_time_seconds")
            self.metrics.register_gauge("ollama_instance_health_score")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the Ollama provider."""
        try:
            self.logger.info("Initializing Ollama provider...")
            
            # Initialize instances
            for instance in self.instances.values():
                await self._initialize_instance(instance)
            
            # Start health monitoring
            for instance in self.instances.values():
                await self.health_monitor.start_monitoring(instance)
                self.health_monitor.health_callbacks[instance.instance_id] = self._handle_instance_health_change
            
            # Discover models
            await self._discover_all_models()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Register with component manager
            await self._register_as_component()
            
            self.logger.info("Ollama provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama provider: {str(e)}")
            raise OllamaError(f"Initialization failed: {str(e)}")

    async def _initialize_instance(self, instance: OllamaInstance) -> None:
        """Initialize a single Ollama instance."""
        try:
            instance.state = OllamaState.CONNECTING
            
            # Test connectivity
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5.0)) as session:
                async with session.get(f"{instance.base_url}/api/tags") as response:
                    if response.status == 200:
                        instance.state = OllamaState.CONNECTED
                        self.logger.info(f"Connected to Ollama instance: {instance.instance_id}")
                    else:
                        instance.state = OllamaState.ERROR
                        self.logger.error(f"Failed to connect to {instance.instance_id}: HTTP {response.status}")
        
        except Exception as e:
            instance.state = OllamaState.ERROR
            self.logger.error(f"Failed to initialize instance {instance.instance_id}: {str(e)}")

    async def _discover_all_models(self) -> None:
        """Discover models on all instances."""
        for instance in self.instances.values():
            if instance.state in [OllamaState.CONNECTED, OllamaState.HEALTHY]:
                try:
                    models = await self.model_manager.discover_models(instance)
                    self.logger.info(f"Discovered {len(models)} models on {instance.instance_id}")
                except Exception as e:
                    self.logger.warning(f"Model discovery failed for {instance.instance_id}: {str(e)}")

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Model management task
        self.background_tasks.append(
            asyncio.create_task(self._model_management_loop())
        )
        
        # Performance monitoring task
        self.background_tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        # Cache management task
        if self.enable_caching:
            self.background_tasks.append(
                asyncio.create_task(self._cache_management_loop())
            )
        
        # Model adaptation task
        if self.model_adapter:
            self.background_tasks.append(
                asyncio.create_task(self._model_adaptation_loop())
            )

    async def _register_event_handlers(self) -> None:
        """Register event handlers."""
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)
        
        # Workflow events
        self.event_bus.subscribe("workflow_started", self._handle_workflow_started)

    async def _register_as_component(self) -> None:
        """Register with component manager."""
        try:
            self.component_manager.register_component(
                "ollama_provider",
                EnhancedOllamaProvider,
                ComponentPriority.HIGH,
                [],
                description="Advanced Ollama LLM Provider"
            )
        except Exception as e:
            self.logger.warning(f"Failed to register as component: {str(e)}")

    @handle_exceptions
    async def generate(self, request: OllamaRequest) -> Union[OllamaResponse, AsyncGenerator[OllamaResponse, None]]:
        """
        Generate response using Ollama.
        
        Args:
            request: Ollama request configuration
            
        Returns:
            Response or async generator for streaming
        """
        async with self.request_semaphore:
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            try:
                with self.tracer.trace("ollama_generation") as span:
                    span.set_attributes({
                        "model": request.model,
                        "request_id": request_id,
                        "stream": request.stream,
                        "session_id": getattr(request, 'session_id', None)
                    })
                    
                    # Store active request
                    self.active_requests[request_id] = {
                        "start_time": start_time,
                        "request": request,
                        "status": "processing"
                    }
                    
                    # Check cache first
                    if self.enable_caching and not request.stream:
                        cached_response = await self._check_cache(request)
                        if cached_response:
                            self.logger.debug(f"Cache hit for request {request_id}")
                            return cached_response
                    
                    # Select instance
                    instance = self.load_balancer.select_instance(list(self.instances.values()), request)
                    if not instance:
                        raise OllamaError("No healthy Ollama instances available")
                    
                    # Ensure model is loaded
                    if self.auto_model_loading and request.model not in instance.loaded_models:
                        await self._ensure_model_loaded(instance, request.model)
                    
                    # Emit processing started event
                    await self.event_bus.emit(ProcessingStarted(
                        session_id=getattr(request, 'session_id', ''),
                        request_id=request_id,
                        component="ollama_provider",
                        model=request.model
                    ))
                    
                    # Generate response
                    if request.stream:
                        return self._generate_streaming(instance, request, request_id)
                    else:
                        response = await self._generate_batch(instance, request, request_id)
                        
                        # Cache response
                        if self.enable_caching:
                            await self._cache_response(request, response)
                        
                        # Update performance metrics
                        response_time = time.time() - start_time
                        instance.response_times.append(response_time)
                        instance.total_requests += 1
                        
                        # Update metrics
                        self.metrics.increment("ollama_requests_total")
                        self.metrics.increment("ollama_requests_successful")
                        self.metrics.record("ollama_response_duration_seconds", response_time)
                        
                        # Store performance data
                        self.performance_stats[f"{instance.instance_id}_{request.model}"].append(response_time)
                        
                        # Emit completion event
                        await self.event_bus.emit(ProcessingCompleted(
                            session_id=getattr(request, 'session_id', ''),
                            request_id=request_id,
                            component="ollama_provider",
                            processing_time=response_time,
                            success=True
                        ))
                        
                        return response
                        
            except Exception as e:
                # Handle error
                response_time = time.time() - start_time
                
                # Update error metrics
                self.metrics.increment("ollama_requests_failed")
                if 'instance' in locals():
                    instance.error_count += 1
                
                # Emit error event
                await self.event_bus.emit(ProcessingError(
                    session_id=getattr(request, 'session_id', ''),
                    request_id=request_id,
                    component="ollama_provider",
                    error_type=type(e).__name__,
                    error_message=str(e)
                ))
                
                self.logger.error(f"Ollama generation failed for request {request_id}: {str(e)}")
                raise OllamaError(f"Generation failed: {str(e)}")
                
            finally:
                # Cleanup active request
                self.active_requests.pop(request_id, None)

    async def _generate_batch(self, instance: OllamaInstance, 
                            request: OllamaRequest, request_id: str) -> OllamaResponse:
        """Generate batch response."""
        try:
            instance.active_requests += 1
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=request.timeout or instance.timeout)) as session:
                payload = self._build_request_payload(request)
                
                async with session.post(
                    f"{instance.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        ollama_response = OllamaResponse(
                            text=data.get('response', ''),
                            model=request.model,
                            response=data.get('response', ''),
                            context=data.get('context'),
                            done=data.get('done', True),
                            total_duration=data.get('total_duration', 0),
                            load_duration=data.get('load_duration', 0),
                            prompt_eval_count=data.get('prompt_eval_count', 0),
                            prompt_eval_duration=data.get('prompt_eval_duration', 0),
                            eval_count=data.get('eval_count', 0),
                            eval_duration=data.get('eval_duration', 0),
                            instance_id=instance.instance_id,
                            model_info=self.model_manager.get_model_info(request.model)
                        )
                        
                        # Calculate quality metrics
                        ollama_response.quality_score = self._calculate_quality_score(ollama_response)
                        ollama_response.confidence = self._calculate_confidence(ollama_response)
                        
                        return ollama_response
                    else:
                        error_text = await response.text()
                        raise OllamaError(f"HTTP {response.status}: {error_text}", instance.instance_id, request.model)
        
        finally:
            instance.active_requests -= 1

    async def _generate_streaming(self, instance: OllamaInstance, 
                                request: OllamaRequest, request_id: str) -> AsyncGenerator[OllamaResponse, None]:
        """Generate streaming response."""
        try:
            instance.active_requests += 1
            
            # Emit streaming started event
            await self.event_bus.emit(StreamingStarted(
                session_id=getattr(request, 'session_id', ''),
                request_id=request_id,
                component="ollama_provider"
            ))
            
            async with aiohttp.ClientSession() as session:
                payload = self._build_request_payload(request)
                payload['stream'] = True
                
                async with session.post(
                    f"{instance.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        accumulated_response = ""
                        chunk_count = 0
                        
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode().strip())
                                    
                                    chunk_response = data.get('response', '')
                                    accumulated_response += chunk_response
                                    
                                    ollama_response = OllamaResponse(
                                        text=accumulated_response,
                                        model=request.model,
                                        response=chunk_response,
                                        context=data.get('context'),
                                        done=data.get('done', False),
                                        total_duration=data.get('total_duration', 0),
                                        instance_id=instance.instance_id,
                                        model_info=self.model_manager.get_model_info(request.model)
                                    )
                                    
                                    chunk_count += 1
                                    yield ollama_response
                                    
                                    if data.get('done', False):
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                        
                        # Emit streaming completed event
                        await self.event_bus.emit(StreamingCompleted(
                            session_id=getattr(request, 'session_id', ''),
                            request_id=request_id,
                            component="ollama_provider",
                            total_chunks=chunk_count
                        ))
                    else:
                        error_text = await response.text()
                        await self.event_bus.emit(StreamingError(
                            session_id=getattr(request, 'session_id', ''),
                            request_id=request_id,
                            error_message=f"HTTP {response.status}: {error_text}"
                        ))
                        raise OllamaError(f"HTTP {response.status}: {error_text}", instance.instance_id, request.model)
        
        finally:
            instance.active_requests -= 1

    def _build_request_payload(self, request: OllamaRequest) -> Dict[str, Any]:
        """Build request payload for Ollama API."""
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
            "raw": request.raw,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "repeat_penalty": request.repeat_penalty,
                "num_predict": request.num_predict,
                "num_ctx": request.num_ctx,
                **request.options
            }
        }
        
        # Add optional fields
        if request.keep_alive:
            payload["keep_alive"] = request.keep_alive
        
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        
        if request.format:
            payload["format"] = request.format
        
        if request.context:
            payload["context"] = request.context
        
        if request.images:
            payload["images"] = request.images
        
        return payload

    def _calculate_quality_score(self, response: OllamaResponse) -> float:
        """Calculate quality score for a response."""
        factors = []
        
        # Response length factor
        if response.text:
            length_factor = min(1.0, len(response.text) / 1000.0)  # Normalize to reasonable length
            factors.append(length_factor)
        
        # Generation speed factor
        if response.eval_duration > 0 and response.eval_count > 0:
            tokens_per_second = response.eval_count / (response.eval_duration / 1e9)
            speed_factor = min(1.0, tokens_per_second / 50.0)  # Normalize to 50 tokens/sec
            factors.append(speed_factor)
        
        # Model-specific factors
        model_info = response.model_info
        if model_info:
            factors.append(model_info.quality_score)
        
        return sum(factors) / len(factors) if factors else 0.5

    def _calculate_confidence(self, response: OllamaResponse) -> float:
        """Calculate confidence score for a response."""
        # This would implement more sophisticated confidence calculation
        # For now, a simple heuristic based on response characteristics
        
        factors = []
        
        # Response completeness
        if response.done:
            factors.append(1.0)
        else:
            factors.append(0.5)
        
        # Evaluation count factor (more evaluations generally mean better quality)
        if response.eval_count > 0:
            eval_factor = min(1.0, response.eval_count / 100.0)
            factors.append(eval_factor)
        
        return sum(factors) / len(factors) if factors else 0.5

    async def _ensure_model_loaded(self, instance: OllamaInstance, model: str) -> None:
        """Ensure a model is loaded on an instance."""
        if model not in instance.loaded_models:
            start_time = time.time()
            
            try:
                success = await self.model_manager.load_model(instance, model)
                load_time = time.time() - start_time
                
                if success:
                    # Emit model loaded event
                    await self.event_bus.emit(ModelLoaded(
                        model_name=model,
                        instance_id=instance.instance_id,
                        load_time=load_time
                    ))
                    
                    self.metrics.record("ollama_model_load_time_seconds", load_time)
                    self.logger.info(f"Loaded model {model} on {instance.instance_id} in {load_time:.2f}s")
                else:
                    raise OllamaError(f"Failed to load model {model}", instance.instance_id, model)
                    
            except Exception as e:
                await self.event_bus.emit(ModelError(
                    model_name=model,
                    instance_id=instance.instance_id,
                    error_message=str(e)
                ))
                raise

    async def _check_cache(self, request: OllamaRequest) -> Optional[OllamaResponse]:
        """Check if response is cached."""
        if not self.cache_strategy:
            return None
        
        # Create cache key
        cache_key = self._create_cache_key(request)
        
        try:
            cached_data = await self.cache_strategy.get(cache_key)
            if cached_data:
                response_data = json.loads(cached_data)
                return OllamaResponse(**response_data)
        except Exception as e:
            self.logger.warning(f"Cache check failed: {str(e)}")
        
        return None

    async def _cache_response(self, request: OllamaRequest, response: OllamaResponse) -> None:
        """Cache response."""
        if not self.cache_strategy:
            return
        
        cache_key = self._create_cache_key(request)
        
        try:
            response_data = asdict(response)
            await self.cache_strategy.set(cache_key, json.dumps(response_data, default=str), ttl=self.cache_ttl)
        except Exception as e:
            self.logger.warning(f"Response caching failed: {str(e)}")

    def _create_cache_key(self, request: OllamaRequest) -> str:
        """Create cache key for request."""
        key_data = {
            "model": request.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_tokens": request.max_tokens
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"ollama:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def list_models(self, instance_id: Optional[str] = None) -> List[OllamaModelInfo]:
        """
        List available models.
        
        Args:
            instance_id: Optional specific instance to query
            
        Returns:
            List of model information
        """
        if instance_id and instance_id in self.instances:
            instance = self.instances[instance_id]
            return await self.model_manager.discover_models(instance)
        else:
            # Return models from all instances
            all_models = []
            for instance in self.instances.values():
                if instance.state in [OllamaState.HEALTHY, OllamaState.CONNECTED]:
                    try:
                        models = await self.model_manager.discover_models(instance)
                        all_models.extend(models)
                    except Exception as e:
                        self.logger.warning(f"Failed to list models from {instance.instance_id}: {str(e)}")
            
            return all_models

    async def switch_model(self, old_model: str, new_model: str, 
                          instance_id: Optional[str] = None) -> bool:
        """
        Switch from one model to another.
        
        Args:
            old_model: Current model name
            new_model: Target model name
            instance_id: Optional specific instance
            
        Returns:
            Success status
        """
        try:
            instances_to_update = []
            
            if instance_id and instance_id in self.instances:
                instances_to_update = [self.instances[instance_id]]
            else:
                instances_to_update = [
                    instance for instance in self.instances.values()
                    if old_model in instance.loaded_models
                ]
            
            success_count = 0
            
            for instance in instances_to_update:
                try:
                    # Load new model
                    await self._ensure_model_loaded(instance, new_model)
                    
                    # Remove old model from loaded set (unloading is automatic in Ollama)
                    instance.loaded_models.discard(old_model)
                    
                    success_count += 1
                    
                    # Emit model switch event
                    await self.event_bus.emit(ModelSwitched(
                        old_model=old_model,
                        new_model=new_model,
                        instance_id=instance.instance_id
                    ))
                    
                except Exception as e:
                    self.logger.error(f"Model switch failed on {instance.instance_id}: {str(e)}")
            
            # Update metrics
            if success_count > 0:
                self.metrics.increment("ollama_model_switches")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Model switch operation failed: {str(e)}")
            return False

    async def pull_model(self, model_name: str, instance_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Pull a model to Ollama instance(s).
        
        Args:
            model_name: Model name to pull
            instance_id: Optional specific instance
            
        Yields:
            Pull progress updates
        """
        instances_to_pull = []
        
        if instance_id and instance_id in self.instances:
            instances_to_pull = [self.instances[instance_id]]
        else:
            # Pull to all healthy instances
            instances_to_pull = [
                instance for instance in self.instances.values()
                if instance.state in [OllamaState.HEALTHY, OllamaState.CONNECTED]
            ]
        
        for instance in instances_to_pull:
            try:
                async for progress in self.model_manager.pull_model(instance, model_name):
                    progress['instance_id'] = instance.instance_id
                    yield progress
                    
                    # If pull completed successfully, update model registry
                    if progress.get('status') == 'success':
                        await self.model_manager.discover_models(instance)
                        
            except Exception as e:
                self.logger.error(f"Model pull failed on {instance.instance_id}: {str(e)}")
                yield {
                    'instance_id': instance.instance_id,
                    'status': 'error',
                    'error': str(e)
                }

    def get_instance_status(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of Ollama instances.
        
        Args:
            instance_id: Optional specific instance
            
        Returns:
            Instance status information
        """
        if instance_id and instance_id in self.instances:
            instance = self.instances[instance_id]
            return self._format_instance_status(instance)
        else:
            return {
                instance_id: self._format_instance_status(instance)
                for instance_id, instance in self.instances.items()
            }

    def _format_instance_status(self, instance: OllamaInstance) -> Dict[str, Any]:
        """Format instance status for response."""
        return {
            'instance_id': instance.instance_id,
            'host': instance.host,
            'port': instance.port,
            'state': instance.state.value,
            'health_score': instance.health_score,
            'last_health_check': instance.last_health_check.isoformat() if instance.last_health_check else None,
            'avg_response_time': instance.avg_response_time,
            'active_requests': instance.active_requests,
            'total_requests': instance.total_requests,
            'error_count': instance.error_count,
            'load_factor': instance.load_factor,
            'loaded_models': list(instance.loaded_models),
            'available_memory': instance.available_memory,
            'total_memory': instance.total_memory,
            'cpu_usage': instance.cpu_usage
        }

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_requests = sum(instance.total_requests for instance in self.instances.values())
        total_errors = sum(instance.error_count for instance in self.instances.values())
        
        active_instances = len([
            instance for instance in self.instances.values()
            if instance.state == OllamaState.HEALTHY
        ])
        
        # Calculate overall response time
        all_response_times = []
        for instance in self.instances.values():
            all_response_times.extend(instance.response_times)
        
        avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0.0
        
        return {
            'total_instances': len(self.instances),
            'active_instances': active_instances,
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_requests, 1),
            'avg_response_time': avg_response_time,
            'active_requests': len(self.active_requests),
            'cache_enabled': self.enable_caching,
            'load_balancing_strategy': self.load_balancing_strategy.value,
            'total_models': len(self.model_manager.model_registry),
            'instances': {
                instance_id: self._format_instance_status(instance)
                for instance_id, instance in self.instances.items()
            }
        }

    # Background task methods
    async def _model_management_loop(self) -> None:
        """Background task for model management."""
        while True:
            try:
                # Periodically discover new models
                await self._discover_all_models()
                
                # Update model metrics
                for instance in self.instances.values():
                    self.metrics.set(
                        "ollama_loaded_models",
                        len(instance.loaded_models),
                        tags={'instance_id': instance.instance_id}
                    )
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Model management loop error: {str(e)}")
                await asyncio.sleep(300)

    async def _performance_monitoring_loop(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                # Update instance metrics
                healthy_instances = 0
                
                for instance in self.instances.values():
                    # Update health score metric
                    self.metrics.set(
                        "ollama_instance_health_score",
                        instance.health_score,
                        tags={'instance_id': instance.instance_id}
                    )
                    
                    if instance.state == OllamaState.HEALTHY:
                        healthy_instances += 1
                
                # Update global metrics
                self.metrics.set("ollama_active_instances", healthy_instances)
                
                # Check performance thresholds
                for instance in self.instances.values():
                    if instance.avg_response_time > 10.0:  # 10 second threshold
                        await self.event_bus.emit(PerformanceThresholdExceeded(
                            component="ollama_provider",
                            metric="response_time",
                            value=instance.avg_response_time,
                            threshold=10.0,
                            instance_id=instance.instance_id
                        ))
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring loop error: {str(e)}")
                await asyncio.sleep(30)

    async def _cache_management_loop(self) -> None:
        """Background task for cache management."""
        while True:
            try:
                # Clean up expired cache entries
                if self.cache_strategy and hasattr(self.cache_strategy, 'cleanup_expired'):
                    await self.cache_strategy.cleanup_expired()
                
                # Clean up in-memory response cache
                current_time = time.time()
                expired_keys = []
                
                for key, response in self.response_cache.items():
                    if hasattr(response, 'created_at'):
                        age = current_time - response.created_at.timestamp()
                        if age > self.cache_ttl:
                            expired_keys.append(key)
                
                for key in expired_keys:
                    self.response_cache.pop(key, None)
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Cache management loop error: {str(e)}")
                await asyncio.sleep(600)

    async def _model_adaptation_loop(self) -> None:
        """Background task for model adaptation."""
        if not self.model_adapter:
            return
        
        while True:
            try:
                # Analyze performance data and adapt models
                for instance_model, response_times in self.performance_stats.items():
                    if len(response_times) >= 10:  # Enough data for analysis
                        avg_time = sum(response_times) / len(response_times)
                        
                        # If performance is degrading, consider model adaptation
                        if avg_time > 5.0:  # 5 second threshold
                            instance_id, model = instance_model.split('_', 1)
                            
                            adaptation_data = {
                                'instance_id': instance_id,
                                'model': model,
                                'avg_response_time': avg_time,
                                'sample_count': len(response_times)
                            }
                            
                            await self.model_adapter.analyze_performance(adaptation_data)
                
                await asyncio.sleep(1800)  # Run every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Model adaptation loop error: {str(e)}")
                await asyncio.sleep(1800)

    # Event handlers
    async def _handle_instance_health_change(self, instance: OllamaInstance, 
                                           old_state: OllamaState, new_state: OllamaState) -> None:
        """Handle instance health state changes."""
        await self.event_bus.emit(ComponentHealthChanged(
            component=f"ollama_instance_{instance.instance_id}",
            healthy=new_state in [OllamaState.HEALTHY, OllamaState.CONNECTED],
            details={
                'old_state': old_state.value,
                'new_state': new_state.value,
                'health_score': instance.health_score,
                'instance_id': instance.instance_id
            }
        ))
        
        self.logger.info(f"Instance {instance.instance_id} state changed: {old_state.value} -> {new_state.value}")

    async def _handle_session_started(self, event) -> None:
        """Handle session started events."""
        # Could preload commonly used models for the session
        pass

    async def _handle_session_ended(self, event) -> None:
        """Handle session ended events."""
        # Could implement session-specific cleanup
        pass

    async def _handle_system_shutdown(self, event) -> None:
        """Handle system shutdown."""
        await self.cleanup()

    async def _handle_workflow_started(self, event) -> None:
        """Handle workflow started events."""
        # Could optimize model loading based on workflow requirements
        pass

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the Ollama provider."""
        try:
            healthy_instances = len([
                instance for instance in self.instances.values()
                if instance.state == OllamaState.HEALTHY
            ])
            
            total_instances = len(self.instances)
            
            return {
                "status": "healthy" if healthy_instances > 0 else "unhealthy",
                "healthy_instances": healthy_instances,
                "total_instances": total_instances,
                "active_requests": len(self.active_requests
