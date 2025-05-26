"""
Advanced Workflow Orchestration Engine
Author: Drmusab
Last Modified: 2025-05-26 15:06:22 UTC

This module provides comprehensive workflow orchestration for the AI assistant,
managing complex multi-step processes, parallel execution, conditional branching,
and dynamic workflow adaptation based on context and user preferences.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    WorkflowStarted, WorkflowCompleted, WorkflowFailed, WorkflowPaused, WorkflowResumed,
    WorkflowStepStarted, WorkflowStepCompleted, WorkflowStepFailed, WorkflowStepSkipped,
    WorkflowBranchingOccurred, WorkflowMerged, WorkflowAdapted, WorkflowCancelled,
    ProcessingStarted, ProcessingCompleted, SkillExecutionStarted, SkillExecutionCompleted,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Assistant components
from src.assistant.session_manager import SessionManager
from src.assistant.component_manager import ComponentManager

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.reasoning.planning.task_planner import TaskPlanner
from src.reasoning.decision_making.decision_tree import DecisionTree
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.knowledge_graph import KnowledgeGraph

# Skills management
from src.skills.skill_factory import SkillFactory
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator

# Memory systems
from src.memory.memory_manager import MemoryManager
from src.memory.context_manager import ContextManager
from src.memory.working_memory import WorkingMemory

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class WorkflowState(Enum):
    """Workflow execution states."""
    CREATED = "created"
    PLANNING = "planning"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class StepState(Enum):
    """Individual step execution states."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"
    BLOCKED = "blocked"


class ExecutionMode(Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ADAPTIVE = "adaptive"
    STREAMING = "streaming"
    BATCH = "batch"


class WorkflowPriority(Enum):
    """Workflow execution priorities."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4


class StepType(Enum):
    """Types of workflow steps."""
    SKILL_EXECUTION = "skill_execution"
    DATA_PROCESSING = "data_processing"
    DECISION_POINT = "decision_point"
    PARALLEL_GATEWAY = "parallel_gateway"
    MERGE_GATEWAY = "merge_gateway"
    CONDITION_CHECK = "condition_check"
    USER_INTERACTION = "user_interaction"
    MEMORY_OPERATION = "memory_operation"
    EXTERNAL_API_CALL = "external_api_call"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"


@dataclass
class WorkflowCondition:
    """Condition for workflow execution paths."""
    condition_id: str
    expression: str
    condition_type: str = "boolean"  # boolean, numeric, string, custom
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against the provided context."""
        try:
            # This would implement condition evaluation logic
            # For now, a simplified version
            if self.condition_type == "boolean":
                return bool(eval(self.expression, {"__builtins__": {}}, context))
            elif self.condition_type == "numeric":
                return float(eval(self.expression, {"__builtins__": {}}, context)) > 0
            else:
                return True
        except Exception:
            return False


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""
    step_id: str
    step_type: StepType
    name: str
    description: Optional[str] = None
    
    # Execution configuration
    skill_name: Optional[str] = None
    component_name: Optional[str] = None
    function_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Flow control
    dependencies: List[str] = field(default_factory=list)
    conditions: List[WorkflowCondition] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Error handling
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_error: bool = False
    fallback_steps: List[str] = field(default_factory=list)
    
    # Performance
    timeout_seconds: float = 30.0
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    
    # State tracking
    state: StepState = StepState.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    workflow_id: str
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Steps and flow
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    start_steps: List[str] = field(default_factory=list)
    end_steps: List[str] = field(default_factory=list)
    
    # Execution configuration
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout_seconds: float = 300.0
    max_concurrent_steps: int = 5
    
    # Error handling
    error_handling_strategy: str = "stop_on_error"  # stop_on_error, continue_on_error, retry
    global_retry_count: int = 0
    max_global_retries: int = 3
    
    # Context and variables
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    category: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Runtime workflow execution instance."""
    execution_id: str
    workflow_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # State management
    state: WorkflowState = WorkflowState.CREATED
    current_steps: Set[str] = field(default_factory=set)
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    
    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    # Context and variables
    context: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    step_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: WorkflowPriority = WorkflowPriority.NORMAL


class WorkflowError(Exception):
    """Custom exception for workflow operations."""
    
    def __init__(self, message: str, workflow_id: Optional[str] = None, 
                 execution_id: Optional[str] = None, step_id: Optional[str] = None):
        super().__init__(message)
        self.workflow_id = workflow_id
        self.execution_id = execution_id
        self.step_id = step_id
        self.timestamp = datetime.now(timezone.utc)


class StepExecutor(ABC):
    """Abstract base class for step executors."""
    
    @abstractmethod
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a workflow step."""
        pass
    
    @abstractmethod
    def can_execute(self, step: WorkflowStep) -> bool:
        """Check if this executor can handle the step."""
        pass


class SkillStepExecutor(StepExecutor):
    """Executor for skill-based workflow steps."""
    
    def __init__(self, skill_registry: SkillRegistry, skill_factory: SkillFactory):
        self.skill_registry = skill_registry
        self.skill_factory = skill_factory
        self.logger = get_logger(__name__)
    
    def can_execute(self, step: WorkflowStep) -> bool:
        """Check if this executor can handle skill execution steps."""
        return step.step_type == StepType.SKILL_EXECUTION and step.skill_name is not None
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a skill-based step."""
        if not step.skill_name:
            raise WorkflowError(f"No skill name specified for step {step.step_id}")
        
        # Get skill instance
        skill = await self.skill_factory.create_skill(step.skill_name)
        
        # Prepare parameters
        params = {**step.parameters, **context}
        
        # Execute skill
        result = await skill.execute(**params)
        
        return result


class ComponentStepExecutor(StepExecutor):
    """Executor for component-based workflow steps."""
    
    def __init__(self, component_manager: ComponentManager):
        self.component_manager = component_manager
        self.logger = get_logger(__name__)
    
    def can_execute(self, step: WorkflowStep) -> bool:
        """Check if this executor can handle component steps."""
        return (step.step_type == StepType.DATA_PROCESSING and 
                step.component_name is not None)
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a component-based step."""
        if not step.component_name:
            raise WorkflowError(f"No component name specified for step {step.step_id}")
        
        # Get component instance
        component = await self.component_manager.get_component(step.component_name)
        
        # Get function to call
        function_name = step.function_name or "process"
        if not hasattr(component, function_name):
            raise WorkflowError(f"Component {step.component_name} has no function {function_name}")
        
        func = getattr(component, function_name)
        
        # Prepare parameters
        params = {**step.parameters, **context}
        
        # Execute function
        if asyncio.iscoroutinefunction(func):
            result = await func(**params)
        else:
            result = func(**params)
        
        return result


class DecisionStepExecutor(StepExecutor):
    """Executor for decision point steps."""
    
    def __init__(self, decision_tree: DecisionTree):
        self.decision_tree = decision_tree
        self.logger = get_logger(__name__)
    
    def can_execute(self, step: WorkflowStep) -> bool:
        """Check if this executor can handle decision steps."""
        return step.step_type == StepType.DECISION_POINT
    
    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a decision step."""
        # Evaluate conditions
        decision_result = {
            "conditions_met": [],
            "next_steps": []
        }
        
        for condition in step.conditions:
            if condition.evaluate(context):
                decision_result["conditions_met"].append(condition.condition_id)
        
        # Determine next steps based on decision tree logic
        if hasattr(self.decision_tree, 'decide'):
            next_steps = await self.decision_tree.decide(context, step.conditions)
            decision_result["next_steps"] = next_steps
        else:
            # Fallback logic
            decision_result["next_steps"] = step.next_steps
        
        return decision_result


class WorkflowBuilder:
    """Builder for creating workflow definitions."""
    
    def __init__(self):
        self.workflow = WorkflowDefinition(
            workflow_id=str(uuid.uuid4()),
            name="New Workflow"
        )
    
    def set_basic_info(self, workflow_id: str, name: str, description: str = None, 
                      version: str = "1.0.0") -> 'WorkflowBuilder':
        """Set basic workflow information."""
        self.workflow.workflow_id = workflow_id
        self.workflow.name = name
        self.workflow.description = description
        self.workflow.version = version
        return self
    
    def add_step(self, step: WorkflowStep) -> 'WorkflowBuilder':
        """Add a step to the workflow."""
        self.workflow.steps[step.step_id] = step
        return self
    
    def add_skill_step(self, step_id: str, skill_name: str, name: str = None, 
                      parameters: Dict[str, Any] = None, **kwargs) -> 'WorkflowBuilder':
        """Add a skill execution step."""
        step = WorkflowStep(
            step_id=step_id,
            step_type=StepType.SKILL_EXECUTION,
            name=name or f"Execute {skill_name}",
            skill_name=skill_name,
            parameters=parameters or {},
            **kwargs
        )
        return self.add_step(step)
    
    def add_decision_step(self, step_id: str, name: str, conditions: List[WorkflowCondition],
                         **kwargs) -> 'WorkflowBuilder':
        """Add a decision point step."""
        step = WorkflowStep(
            step_id=step_id,
            step_type=StepType.DECISION_POINT,
            name=name,
            conditions=conditions,
            **kwargs
        )
        return self.add_step(step)
    
    def add_component_step(self, step_id: str, component_name: str, function_name: str,
                          name: str = None, parameters: Dict[str, Any] = None,
                          **kwargs) -> 'WorkflowBuilder':
        """Add a component execution step."""
        step = WorkflowStep(
            step_id=step_id,
            step_type=StepType.DATA_PROCESSING,
            name=name or f"Execute {component_name}.{function_name}",
            component_name=component_name,
            function_name=function_name,
            parameters=parameters or {},
            **kwargs
        )
        return self.add_step(step)
    
    def set_flow(self, from_step: str, to_step: str, condition: WorkflowCondition = None) -> 'WorkflowBuilder':
        """Set flow between steps."""
        if from_step in self.workflow.steps:
            self.workflow.steps[from_step].next_steps.append(to_step)
            if condition:
                self.workflow.steps[from_step].conditions.append(condition)
        
        if to_step in self.workflow.steps:
            self.workflow.steps[to_step].dependencies.append(from_step)
        
        return self
    
    def set_start_steps(self, *step_ids: str) -> 'WorkflowBuilder':
        """Set the starting steps of the workflow."""
        self.workflow.start_steps = list(step_ids)
        return self
    
    def set_end_steps(self, *step_ids: str) -> 'WorkflowBuilder':
        """Set the ending steps of the workflow."""
        self.workflow.end_steps = list(step_ids)
        return self
    
    def set_execution_mode(self, mode: ExecutionMode) -> 'WorkflowBuilder':
        """Set the workflow execution mode."""
        self.workflow.execution_mode = mode
        return self
    
    def build(self) -> WorkflowDefinition:
        """Build and return the workflow definition."""
        self._validate_workflow()
        return self.workflow
    
    def _validate_workflow(self) -> None:
        """Validate the workflow definition."""
        if not self.workflow.steps:
            raise WorkflowError("Workflow must have at least one step")
        
        if not self.workflow.start_steps:
            raise WorkflowError("Workflow must have at least one start step")
        
        # Validate step references
        all_step_ids = set(self.workflow.steps.keys())
        
        for step in self.workflow.steps.values():
            for next_step in step.next_steps:
                if next_step not in all_step_ids:
                    raise WorkflowError(f"Step {step.step_id} references non-existent step {next_step}")
            
            for dependency in step.dependencies:
                if dependency not in all_step_ids:
                    raise WorkflowError(f"Step {step.step_id} depends on non-existent step {dependency}")


class WorkflowOrchestrator:
    """
    Advanced Workflow Orchestration Engine for the AI Assistant.
    
    This orchestrator manages complex multi-step workflows including:
    - Dynamic workflow creation and execution
    - Parallel and sequential step execution
    - Conditional branching and merging
    - Error handling and recovery
    - Performance monitoring and optimization
    - Context-aware workflow adaptation
    - Integration with skills, memory, and reasoning systems
    - Real-time workflow modification
    - Workflow templates and reusable patterns
    """
    
    def __init__(self, container: Container):
        """
        Initialize the workflow orchestrator.
        
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
        self.session_manager = container.get(SessionManager)
        self.component_manager = container.get(ComponentManager)
        
        # Processing and reasoning
        self.intent_manager = container.get(IntentManager)
        self.task_planner = container.get(TaskPlanner)
        self.decision_tree = container.get(DecisionTree)
        self.logic_engine = container.get(LogicEngine)
        self.knowledge_graph = container.get(KnowledgeGraph)
        
        # Skills management
        self.skill_registry = container.get(SkillRegistry)
        self.skill_factory = container.get(SkillFactory)
        self.skill_validator = container.get(SkillValidator)
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        
        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # State management
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.workflow_templates: Dict[str, WorkflowDefinition] = {}
        
        # Execution infrastructure
        self.step_executors: List[StepExecutor] = []
        self.execution_semaphore = asyncio.Semaphore(10)  # Max concurrent workflows
        self.step_semaphore = asyncio.Semaphore(20)  # Max concurrent steps
        
        # Performance tracking
        self.execution_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.step_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.max_execution_time = self.config.get("workflows.max_execution_time", 300.0)
        self.max_step_retries = self.config.get("workflows.max_step_retries", 3)
        self.enable_adaptive_workflows = self.config.get("workflows.enable_adaptive", True)
        self.enable_workflow_learning = self.config.get("workflows.enable_learning", True)
        
        # Initialize components
        self._setup_step_executors()
        self._setup_monitoring()
        self._load_workflow_templates()
        
        # Register health check
        self.health_check.register_component("workflow_orchestrator", self._health_check_callback)
        
        self.logger.info("WorkflowOrchestrator initialized successfully")

    def _setup_step_executors(self) -> None:
        """Setup step executors for different step types."""
        try:
            # Skill execution
            self.step_executors.append(SkillStepExecutor(self.skill_registry, self.skill_factory))
            
            # Component execution
            self.step_executors.append(ComponentStepExecutor(self.component_manager))
            
            # Decision points
            self.step_executors.append(DecisionStepExecutor(self.decision_tree))
            
            self.logger.info(f"Initialized {len(self.step_executors)} step executors")
            
        except Exception as e:
            self.logger.error(f"Failed to setup step executors: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register workflow metrics
            self.metrics.register_counter("workflow_executions_total")
            self.metrics.register_counter("workflow_executions_successful")
            self.metrics.register_counter("workflow_executions_failed")
            self.metrics.register_histogram("workflow_execution_duration_seconds")
            self.metrics.register_gauge("active_workflows")
            self.metrics.register_counter("workflow_steps_executed")
            self.metrics.register_histogram("workflow_step_duration_seconds")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _load_workflow_templates(self) -> None:
        """Load predefined workflow templates."""
        try:
            # Load built-in workflow templates
            templates_path = Path("configs/workflows/templates")
            if templates_path.exists():
                for template_file in templates_path.glob("*.json"):
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                        template = self._deserialize_workflow(template_data)
                        self.workflow_templates[template.workflow_id] = template
            
            # Create some basic templates programmatically
            self._create_builtin_templates()
            
            self.logger.info(f"Loaded {len(self.workflow_templates)} workflow templates")
            
        except Exception as e:
            self.logger.warning(f"Failed to load workflow templates: {str(e)}")

    def _create_builtin_templates(self) -> None:
        """Create built-in workflow templates."""
        # Simple question-answering workflow
        qa_workflow = (WorkflowBuilder()
                      .set_basic_info("builtin_qa", "Question Answering", 
                                    "Simple question answering workflow")
                      .add_component_step("intent_detection", "intent_manager", "detect_intent",
                                        "Detect User Intent")
                      .add_skill_step("knowledge_retrieval", "knowledge_search", 
                                    "Retrieve Knowledge")
                      .add_skill_step("response_generation", "text_generation",
                                    "Generate Response")
                      .set_flow("intent_detection", "knowledge_retrieval")
                      .set_flow("knowledge_retrieval", "response_generation")
                      .set_start_steps("intent_detection")
                      .set_end_steps("response_generation")
                      .build())
        
        self.workflow_templates["builtin_qa"] = qa_workflow
        
        # Multimodal processing workflow
        multimodal_workflow = (WorkflowBuilder()
                             .set_basic_info("builtin_multimodal", "Multimodal Processing",
                                           "Process multimodal input")
                             .add_component_step("speech_processing", "speech_processor", "process",
                                               "Process Speech Input")
                             .add_component_step("vision_processing", "vision_processor", "process",
                                               "Process Vision Input")
                             .add_component_step("text_processing", "text_processor", "process",
                                               "Process Text Input")
                             .add_skill_step("multimodal_fusion", "fusion_skill", "Fuse Modalities")
                             .add_skill_step("response_generation", "response_generator",
                                           "Generate Response")
                             .set_flow("speech_processing", "multimodal_fusion")
                             .set_flow("vision_processing", "multimodal_fusion")
                             .set_flow("text_processing", "multimodal_fusion")
                             .set_flow("multimodal_fusion", "response_generation")
                             .set_start_steps("speech_processing", "vision_processing", "text_processing")
                             .set_end_steps("response_generation")
                             .set_execution_mode(ExecutionMode.PARALLEL)
                             .build())
        
        self.workflow_templates["builtin_multimodal"] = multimodal_workflow

    async def initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        try:
            # Initialize step executors
            for executor in self.step_executors:
                if hasattr(executor, 'initialize'):
                    await executor.initialize()
            
            # Start background tasks
            asyncio.create_task(self._execution_monitor_loop())
            asyncio.create_task(self._performance_optimization_loop())
            
            if self.enable_workflow_learning:
                asyncio.create_task(self._learning_update_loop())
            
            self.logger.info("WorkflowOrchestrator initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WorkflowOrchestrator: {str(e)}")
            raise WorkflowError(f"Initialization failed: {str(e)}")

    @handle_exceptions
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Register a workflow definition.
        
        Args:
            workflow: Workflow definition to register
        """
        # Validate workflow
        self._validate_workflow_definition(workflow)
        
        # Store workflow
        self.workflow_definitions[workflow.workflow_id] = workflow
        
        # Emit registration event
        asyncio.create_task(self.event_bus.emit(WorkflowStarted(
            workflow_id=workflow.workflow_id,
            execution_id="",  # Not executed yet
            session_id="",
            workflow_name=workflow.name
        )))
        
        self.logger.info(f"Registered workflow: {workflow.workflow_id} ({workflow.name})")

    def _validate_workflow_definition(self, workflow: WorkflowDefinition) -> None:
        """Validate a workflow definition."""
        if not workflow.workflow_id:
            raise WorkflowError("Workflow must have an ID")
        
        if not workflow.name:
            raise WorkflowError("Workflow must have a name")
        
        if not workflow.steps:
            raise WorkflowError("Workflow must have at least one step")
        
        if not workflow.start_steps:
            raise WorkflowError("Workflow must have at least one start step")
        
        # Validate step references and dependencies
        step_ids = set(workflow.steps.keys())
        
        for step in workflow.steps.values():
            # Check next step references
            for next_step in step.next_steps:
                if next_step not in step_ids:
                    raise WorkflowError(f"Step {step.step_id} references unknown step {next_step}")
            
            # Check dependencies
            for dependency in step.dependencies:
                if dependency not in step_ids:
                    raise WorkflowError(f"Step {step.step_id} depends on unknown step {dependency}")
            
            # Check for circular dependencies
            if self._has_circular_dependency(workflow, step.step_id):
                raise WorkflowError(f"Circular dependency detected involving step {step.step_id}")

    def _has_circular_dependency(self, workflow: WorkflowDefinition, step_id: str, 
                                visited: Set[str] = None, path: Set[str] = None) -> bool:
        """Check for circular dependencies in workflow."""
        if visited is None:
            visited = set()
        if path is None:
            path = set()
        
        if step_id in path:
            return True
        
        if step_id in visited:
            return False
        
        visited.add(step_id)
        path.add(step_id)
        
        step = workflow.steps.get(step_id)
        if step:
            for next_step in step.next_steps:
                if self._has_circular_dependency(workflow, next_step, visited, path):
                    return True
        
        path.remove(step_id)
        return False

    @handle_exceptions
    async def execute_workflow(
        self,
        workflow_id: str,
        session_id: str,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None,
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            session_id: Session ID for the execution
            input_data: Input data for the workflow
            user_id: Optional user ID
            priority: Execution priority
            context: Optional execution context
            
        Returns:
            Execution ID
        """
        # Get workflow definition
        if workflow_id not in self.workflow_definitions:
            # Try to find in templates
            if workflow_id in self.workflow_templates:
                workflow_def = self.workflow_templates[workflow_id]
            else:
                raise WorkflowError(f"Workflow {workflow_id} not found")
        else:
            workflow_def = self.workflow_definitions[workflow_id]
        
        # Create execution instance
        execution = WorkflowExecution(
            execution_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            session_id=session_id,
            user_id=user_id,
            input_data=input_data,
            context=context or {},
            priority=priority
        )
        
        # Store execution
        self.active_executions[execution.execution_id] = execution
        
        # Start execution in background
        asyncio.create_task(self._execute_workflow_async(workflow_def, execution))
        
        self.logger.info(f"Started workflow execution: {execution.execution_id}")
        return execution.execution_id

    async def _execute_workflow_async(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute a workflow asynchronously."""
        async with self.execution_semaphore:
            start_time = time.time()
            execution.start_time = datetime.now(timezone.utc)
            execution.state = WorkflowState.RUNNING
            
            try:
                with self.tracer.trace("workflow_execution") as span:
                    span.set_attributes({
                        "workflow_id": workflow_def.workflow_id,
                        "execution_id": execution.execution_id,
                        "session_id": execution.session_id,
                        "user_id": execution.user_id or "anonymous"
                    })
                    
                    # Emit workflow started event
                    await self.event_bus.emit(WorkflowStarted(
                        workflow_id=workflow_def.workflow_id,
                        execution_id=execution.execution_id,
                        session_id=execution.session_id,
                        workflow_name=workflow_def.name
                    ))
                    
                    # Initialize execution context
                    await self._initialize_execution_context(workflow_def, execution)
                    
                    # Execute workflow based on execution mode
                    if workflow_def.execution_mode == ExecutionMode.SEQUENTIAL:
                        await self._execute_sequential(workflow_def, execution)
                    elif workflow_def.execution_mode == ExecutionMode.PARALLEL:
                        await self._execute_parallel(workflow_def, execution)
                    elif workflow_def.execution_mode == ExecutionMode.ADAPTIVE:
                        await self._execute_adaptive(workflow_def, execution)
                    else:
                        await self._execute_sequential(workflow_def, execution)  # Default
                    
                    # Finalize execution
                    execution.state = WorkflowState.COMPLETED
                    execution.end_time = datetime.now(timezone.utc)
                    execution.execution_time = time.time() - start_time
                    
                    # Update metrics
                    self.metrics.increment("workflow_executions_total")
                    self.metrics.increment("workflow_executions_successful")
                    self.metrics.record("workflow_execution_duration_seconds", execution.execution_time)
                    
                    # Emit completion event
                    await self.event_bus.emit(WorkflowCompleted(
                        workflow_id=workflow_def.workflow_id,
                        execution_id=execution.execution_id,
                        session_id=execution.session_id,
                        execution_time=execution.execution_time,
                        steps_completed=len(execution.completed_steps)
                    ))
                    
                    # Store execution in memory for learning
                    if self.enable_workflow_learning:
                        await self._store_execution_for_learning(workflow_def, execution)
                    
                    self.logger.info(
                        f"Workflow execution completed: {execution.execution_id} "
                        f"in {execution.execution_time:.2f}s"
                    )
                    
            except Exception as e:
                # Handle execution failure
                execution.state = WorkflowState.FAILED
                execution.end_time = datetime.now(timezone.utc)
                execution.execution_time = time.time() - start_time
                execution.errors.append(str(e))
                
                self.metrics.increment("workflow_executions_failed")
                
                await self.event_bus.emit(WorkflowFailed(
                    workflow_id=workflow_def.workflow_id,
                    execution_id=execution.execution_id,
                    session_id=execution.session_id,
                    error_message=str(e),
                    execution_time=execution.execution_time
                ))
                
                self.logger.error(f"Workflow execution failed: {execution.execution_id}: {str(e)}")
                
            finally:
                # Move to execution history
                self.execution_history.append(execution)
                self.active_executions.pop(execution.execution_id, None)

    async def _initialize_execution_context(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Initialize execution context with workflow variables and session data."""
        # Set workflow variables
        execution.variables.update(workflow_def.variables)
        
        # Add input data to context
        execution.context.update(execution.input_data)
        
        # Get session context
        session_context = await self.context_manager.get_session_context(execution.session_id)
        execution.context.update(session_context)
        
        # Get user preferences if available
        if execution.user_id and hasattr(self.preference_learner, 'get_user_preferences'):
            user_prefs = await self.preference_learner.get_user_preferences(execution.user_id)
            execution.context['user_preferences'] = user_prefs

    async def _execute_sequential(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute workflow steps sequentially."""
        # Start with initial steps
        ready_steps = set(workflow_def.start_steps)
        
        while ready_steps and execution.state == WorkflowState.RUNNING:
            # Execute ready steps one by one
            for step_id in list(ready_steps):
                if step_id in execution.completed_steps:
                    ready_steps.remove(step_id)
                    continue
                
                step = workflow_def.steps[step_id]
                
                # Check if dependencies are satisfied
                if not self._are_dependencies_satisfied(step, execution):
                    continue
                
                # Execute step
                try:
                    await self._execute_step(step, execution)
                    execution.completed_steps.add(step_id)
                    ready_steps.remove(step_id)
                    
                    # Add next steps to ready queue
                    for next_step_id in step.next_steps:
                        if next_step_id not in execution.completed_steps:
                            ready_steps.add(next_step_id)
                            
                except Exception as e:
                    if not step.continue_on_error:
                        raise
                    execution.failed_steps.add(step_id)
                    execution.errors.append(f"Step {step_id} failed: {str(e)}")

    async def _execute_parallel(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute workflow steps in parallel where possible."""
        # Start with initial steps
        ready_steps = set(workflow_def.start_steps)
        running_tasks = {}
        
        while (ready_steps or running_tasks) and execution.state == WorkflowState.RUNNING:
            # Start new tasks for ready steps
            for step_id in list(ready_steps):
                if step_id in execution.completed_steps:
                    ready_steps.remove(step_id)
                    continue
                
                step = workflow_def.steps[step_id]
                
                # Check if dependencies are satisfied
                if not self._are_dependencies_satisfied(step, execution):
                    continue
                
                # Start step execution
                task = asyncio.create_task(self._execute_step(step, execution))
                running_tasks[step_id] = task
                ready_steps.remove(step_id)
            
            # Wait for at least one task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    step_id = None
                    for sid, t in running_tasks.items():
                        if t == task:
                            step_id = sid
                            break
                    
                    if step_id:
                        del running_tasks[step_id]
                        
                        try:
                            await task  # Get result or raise exception
                            execution.completed_steps.add(step_id)
                            
                            # Add next steps to ready queue
                            step = workflow_def.steps[step_id]
                            for next_step_id in step.next_steps:
                                if next_step_id not in execution.completed_steps:
                                    ready_steps.add(next_step_id)
                                    
                        except Exception as e:
                            step = workflow_def.steps[step_id]
                            if not step.continue_on_error:
                                # Cancel all running tasks
                                for pending_task in pending:
                                    pending_task.cancel()
                                raise
                            execution.failed_steps.add(step_id)
                            execution.errors.append(f"Step {step_id} failed: {str(e)}")

    async def _execute_adaptive(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute workflow with adaptive step selection."""
        # Use task planner to determine optimal execution order
        if hasattr(self.task_planner, 'plan_workflow_execution'):
            execution_plan = await self.task_planner.plan_workflow_execution(
                workflow_def, execution.context
            )
            
            # Execute according to adaptive plan
            for step_batch in execution_plan:
                # Execute batch in parallel
                tasks = []
                for step_id in step_batch:
                    if step_id in workflow_def.steps:
                        step = workflow_def.steps[step_id]
                        task = asyncio.create_task(self._execute_step(step, execution))
                        tasks.append((step_id, task))
                
                # Wait for batch completion
                for step_id, task in tasks:
                    try:
                        await task
                        execution.completed_steps.add(step_id)
                    except Exception as e:
                        step = workflow_def.steps[step_id]
                        if not step.continue_on_error:
                            raise
                        execution.failed_steps.add(step_id)
                        execution.errors.append(f"Step {step_id} failed: {str(e)}")
        else:
            # Fallback to sequential execution
            await self._execute_sequential(workflow_def, execution)

    def _are_dependencies_satisfied(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> bool:
        """Check if step dependencies are satisfied."""
        for dependency in step.dependencies:
            if dependency not in execution.completed_steps:
                return False
        
        # Check conditions
        for condition in step.conditions:
            if not condition.evaluate(execution.context):
                return False
        
        return True

    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Any:
        """Execute a single workflow step."""
        async with self.step_semaphore:
            step_start_time = time.time()
            step.start_time = datetime.now(timezone.utc)
            step.state = StepState.RUNNING
            execution.current_steps.add(step.step_id)
            
            try:
                # Emit step started event
                await self.event_bus.emit(WorkflowStepStarted(
                    workflow_id=execution.workflow_id,
                    execution_id=execution.execution_id,
                    step_id=step.step_id,
                    step_name=step.name,
                    step_type=step.step_type.value
                ))
                
                # Find appropriate executor
                executor = None
                for exec_candidate in self.step_executors:
                    if exec_candidate.can_execute(step):
                        executor = exec_candidate
                        break
                
                if not executor:
                    raise WorkflowError(f"No executor found for step {step.step_id} of type {step.step_type}")
                
                # Execute step with timeout
                result = await asyncio.wait_for(
                    executor.execute(step, execution.context),
                    timeout=step.timeout_seconds
                )
                
                # Store result
                step.result = result
                step.state = StepState.COMPLETED
                execution.intermediate_results[step.step_id] = result
                
                # Update context with result
                execution.context[f"step_{step.step_id}_result"] = result
                
                # Update execution time
                step.execution_time = time.time() - step_start_time
                step.end_time = datetime.now(timezone.utc)
                
                # Update metrics
                self.metrics.increment("workflow_steps_executed")
                self.metrics.record("workflow_step_duration_seconds", step.execution_time)
                
                # Track performance for learning
                self.step_performance[f"{step.step_type.value}_{step.step_id}"].append(step.execution_time)
                
                # Emit step completed event
                await self.event_bus.emit(WorkflowStepCompleted(
                    workflow_id=execution.workflow_id,
                    execution_id=execution.execution_id,
                    step_id=step.step_id,
                    step_name=step.name,
                    execution_time=step.execution_time,
                    success=True
                ))
                
                self.logger.debug(f"Step {step.step_id} completed in {step.execution_time:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                step.state = StepState.FAILED
                step.error = WorkflowError(f"Step {step.step_id} timed out after {step.timeout_seconds}s")
                
                await self.event_bus.emit(WorkflowStepFailed(
                    workflow_id=execution.workflow_id,
                    execution_id=execution.execution_id,
                    step_id=step.step_id,
                    error_message="Step execution timed out",
                    error_type="TimeoutError"
                ))
                
                raise step.error
                
            except Exception as e:
                step.state = StepState.FAILED
                step.error = e
                step.execution_time = time.time() - step_start_time
                
                # Handle retries
                if step.retry_count < step.max_retries:
                    step.retry_count += 1
                    step.state = StepState.RETRY
                    
                    self.logger.warning(
                        f"Step {step.step_id} failed, retrying ({step.retry_count}/{step.max_retries}): {str(e)}"
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(step.retry_delay * step.retry_count)
                    
                    # Retry the step
                    return await self._execute_step(step, execution)
                
                await self.event_bus.emit(WorkflowStepFailed(
                    workflow_id=execution.workflow_id,
                    execution_id=execution.execution_id,
                    step_id=step.step_id,
                    error_message=str(e),
                    error_type=type(e).__name__
                ))
                
                self.logger.error(f"Step {step.step_id} failed after {step.retry_count} retries: {str(e)}")
                raise
                
            finally:
                execution.current_steps.discard(step.step_id)

    async def _store_execution_for_learning(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Store execution data for learning and optimization."""
        try:
            learning_data = {
                'workflow_id': workflow_def.workflow_id,
                'execution_id': execution.execution_id,
                'session_id': execution.session_id,
                'user_id': execution.user_id,
                'execution_time': execution.execution_time,
                'success': execution.state == WorkflowState.COMPLETED,
                'step_count': len(workflow_def.steps),
                'completed_steps': len(execution.completed_steps),
                'failed_steps': len(execution.failed_steps),
                'context': execution.context,
                'performance_metrics': execution.performance_metrics,
                'timestamp': execution.created_at
            }
            
            # Store in episodic memory
            await self.memory_manager.store_episodic_memory(
                event_type="workflow_execution",
                data=learning_data,
                session_id=execution.session_id
            )
            
            # Update continual learning
            await self.continual_learner.learn_from_workflow_execution(learning_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to store execution for learning: {str(e)}")

    @handle_exceptions
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow execution.
        
        Args:
            execution_id: Execution ID to query
            
        Returns:
            Execution status information
        """
        # Check active executions
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            return self._format_execution_status(execution)
        
        # Check execution history
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return self._format_execution_status(execution)
        
        raise WorkflowError(f"Execution {execution_id} not found")

    def _format_execution_status(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Format execution status for response."""
        return {
            'execution_id': execution.execution_id,
            'workflow_id': execution.workflow_id,
            'session_id': execution.session_id,
            'user_id': execution.user_id,
            'state': execution.state.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'execution_time': execution.execution_time,
            'current_steps': list(execution.current_steps),
            'completed_steps': list(execution.completed_steps),
            'failed_steps': list(execution.failed_steps),
            'progress': len(execution.completed_steps) / max(len(execution.completed_steps) + len(execution.current_steps) + len(execution.failed_steps), 1),
            'errors': execution.errors,
            'warnings': execution.warnings,
            'performance_metrics': execution.performance_metrics
        }

    @handle_exceptions
    async def cancel_execution(self, execution_id: str) -> None:
        """
        Cancel a running workflow execution.
        
        Args:
            execution_id: Execution ID to cancel
        """
        if execution_id not in self.active_executions:
            raise WorkflowError(f"Execution {execution_id} not found or not active")
        
        execution = self.active_executions[execution_id]
        execution.state = WorkflowState.CANCELLED
        
        # Emit cancellation event
        await self.event_bus.emit(WorkflowCancelled(
            workflow_id=execution.workflow_id,
            execution_id=execution.execution_id,
            session_id=execution.session_id
        ))
        
        self.logger.info(f"Cancelled workflow execution: {execution_id}")

    @handle_exceptions
    async def pause_execution(self, execution_id: str) -> None:
        """
        Pause a running workflow execution.
        
        Args:
            execution_id: Execution ID to pause
        """
        if execution_id not in self.active_executions:
            raise WorkflowError(f"Execution {execution_id} not found or not active")
        
        execution = self.active_executions[execution_id]
        if execution.state == WorkflowState.RUNNING:
            execution.state = WorkflowState.PAUSED
            
            await self.event_bus.emit(WorkflowPaused(
                workflow_id=execution.workflow_id,
                execution_id=execution.execution_id,
                session_id=execution.session_id
            ))
            
            self.logger.info(f"Paused workflow execution: {execution_id}")

    @handle_exceptions
    async def resume_execution(self, execution_id: str) -> None:
        """
        Resume a paused workflow execution.
        
        Args:
            execution_id: Execution ID to resume
        """
        if execution_id not in self.active_executions:
            raise WorkflowError(f"Execution {execution_id} not found or not active")
        
        execution = self.active_executions[execution_id]
        if execution.state == WorkflowState.PAUSED:
            execution.state = WorkflowState.RUNNING
            
            await self.event_bus.emit(WorkflowResumed(
                workflow_id=execution.workflow_id,
                execution_id=execution.execution_id,
                session_id=execution.session_id
            ))
            
            self.logger.info(f"Resumed workflow execution: {execution_id}")

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows."""
        workflows = []
        
        # Add registered workflows
        for workflow in self.workflow_definitions.values():
            workflows.append({
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'version': workflow.version,
                'description': workflow.description,
                'step_count': len(workflow.steps),
                'execution_mode': workflow.execution_mode.value,
                'created_at': workflow.created_at.isoformat(),
                'type': 'registered'
            })
        
        # Add templates
        for workflow in self.workflow_templates.values():
            workflows.append({
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'version': workflow.version,
                'description': workflow.description,
                'step_count': len(workflow.steps),
                'execution_mode': workflow.execution_mode.value,
                'created_at': workflow.created_at.isoformat(),
                'type': 'template'
            })
        
        return workflows

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get list of active workflow executions."""
        return [
            self._format_execution_status(execution)
            for execution in self.active_executions.values()
        ]

    async def create_workflow_from_intent(
        self,
        intent: str,
        context: Dict[str, Any],
        session_id: str
    ) -> str:
        """
        Create and execute a workflow based on detected intent.
        
        Args:
            intent: Detected user intent
            context: Context information
            session_id: Session ID
            
        Returns:
            Execution ID
        """
        try:
            # Use task planner to create workflow
            workflow_plan = await self.task_planner.create_workflow_for_intent(intent, context)
            
            # Convert plan to workflow definition
            workflow = self._plan_to_workflow(workflow_plan, intent)
            
            # Register and execute workflow
            self.register_workflow(workflow)
            
            execution_id = await self.execute_workflow(
                workflow.workflow_id,
                session_id,
                context
            )
            
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow from intent {intent}: {str(e)}")
            raise WorkflowError(f"Failed to create workflow from intent: {str(e)}")

    def _plan_to_workflow(self, plan: Dict[str, Any], intent: str) -> WorkflowDefinition:
        """Convert a task plan to a workflow definition."""
        builder = WorkflowBuilder()
        builder.set_basic_info(
            workflow_id=f"intent_{intent}_{int(time.time())}",
            name=f"Workflow for {intent}",
            description=f"Auto-generated workflow for intent: {intent}"
        )
        
        # Add steps from plan
        for i, task in enumerate(plan.get('tasks', [])):
            step_id = f"step_{i}"
            
            if task.get('type') == 'skill':
                builder.add_skill_step(
                    step_id=step_id,
                    skill_name=task.get('skill_name'),
                    name=task.get('name', f"Step {i}"),
                    parameters=task.get('parameters', {})
                )
            elif task.get('type') == 'component':
                builder.add_component_step(
                    step_id=step_id,
                    component_name=task.get('component_name'),
                    function_name=task.get('function_name', 'process'),
                    name=task.get('name', f"Step {i}"),
                    parameters=task.get('parameters', {})
                )
            
            # Set dependencies
            if i > 0:
                builder.set_flow(f"step_{i-1}", step_id)
        
        # Set start and end steps
        if plan.get('tasks'):
            builder.set_start_steps("step_0")
            builder.set_end_steps(f"step_{len(plan['tasks'])-1}")
        
        return builder.build()

    def _serialize_workflow(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Serialize workflow definition to dictionary."""
        return asdict(workflow)

    def _deserialize_workflow(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Deserialize workflow definition from dictionary."""
        # Convert step data back to WorkflowStep objects
        steps = {}
        for step_id, step_data in data.get('steps', {}).items():
            # Convert enums back
            step_data['step_type'] = StepType(step_data['step_type'])
            step_data['state'] = StepState(step_data['state'])
            step_data['priority'] = WorkflowPriority(step_data['priority'])
            
            # Convert conditions
            conditions = []
            for cond_data in step_data.get('conditions', []):
                conditions.append(WorkflowCondition(**cond_data))
            step_data['conditions'] = conditions
            
            # Convert dates
            if step_data.get('start_time'):
                step_data['start_time'] = datetime.fromisoformat(step_data['start_time'])
            if step_data.get('end_time'):
                step_data['end_time'] = datetime.fromisoformat(step_data['end_time'])
            
            steps[step_id] = WorkflowStep(**step_data)
        
        data['steps'] = steps
        
        # Convert enums and dates at workflow level
        data['execution_mode'] = ExecutionMode(data['execution_mode'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return WorkflowDefinition(**data)

    async def _execution_monitor_loop(self) -> None:
        """Background task to monitor workflow executions."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check for timed out executions
                for execution_id, execution in list(self.active_executions.items()):
                    if execution.start_time:
                        runtime = (current_time - execution.start_time).total_seconds()
                        
                        # Check for global timeout
                        workflow_def = self.workflow_definitions.get(execution.workflow_id)
                        if workflow_def and runtime > workflow_def.timeout_seconds:
                            execution.state = WorkflowState.
