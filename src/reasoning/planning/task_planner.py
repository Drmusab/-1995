"""
Advanced Task Planning Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 15:45:33 UTC

This module provides intelligent task planning and decomposition for the AI assistant,
creating optimized execution plans, managing task dependencies, and adapting plans
based on context, user preferences, and available resources.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, Tuple, AsyncGenerator
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
import copy
from concurrent.futures import ThreadPoolExecutor
import heapq
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    TaskPlanningStarted, TaskPlanningCompleted, TaskPlanningFailed,
    TaskDecomposed, TaskPrioritized, TaskScheduled, TaskRescheduled,
    PlanOptimized, PlanAdapted, GoalDecomposed, StrategySelected,
    ResourceAllocationChanged, ConstraintViolationDetected,
    ErrorOccurred, SystemStateChanged, ComponentHealthChanged
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.entity_extractor import EntityExtractor

# Reasoning components
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.reasoning.decision_making.decision_tree import DecisionTree

# Skills and components
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator
from src.assistant.component_manager import ComponentManager

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, SemanticMemory

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


class TaskPriority(Enum):
    """Task priority levels for planning."""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5


class TaskStatus(Enum):
    """Task execution status."""
    PLANNED = "planned"
    READY = "ready"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


class PlanningStrategy(Enum):
    """Planning strategies for different scenarios."""
    GREEDY = "greedy"                    # Quick, locally optimal
    DEPTH_FIRST = "depth_first"          # Complete exploration
    BREADTH_FIRST = "breadth_first"      # Level-by-level
    A_STAR = "a_star"                    # Heuristic search
    HIERARCHICAL = "hierarchical"        # Multi-level planning
    ADAPTIVE = "adaptive"                # Context-aware
    COLLABORATIVE = "collaborative"      # Multi-agent
    ITERATIVE = "iterative"              # Refinement-based


class ResourceType(Enum):
    """Types of resources for task execution."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    GPU = "gpu"
    SKILL = "skill"
    COMPONENT = "component"
    API_QUOTA = "api_quota"
    TIME = "time"
    USER_ATTENTION = "user_attention"


class ConstraintType(Enum):
    """Types of planning constraints."""
    TEMPORAL = "temporal"               # Time-based constraints
    RESOURCE = "resource"               # Resource limitations
    DEPENDENCY = "dependency"           # Task dependencies
    PRECEDENCE = "precedence"           # Ordering constraints
    BUDGET = "budget"                   # Cost constraints
    QUALITY = "quality"                 # Quality requirements
    SECURITY = "security"               # Security restrictions
    USER_PREFERENCE = "user_preference" # User preferences


@dataclass
class TaskConstraint:
    """Represents a constraint on task execution."""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hard_constraint: bool = True  # True for hard, False for soft
    weight: float = 1.0  # Weight for soft constraints
    violation_penalty: float = 100.0  # Penalty for violating this constraint
    
    def check_constraint(self, task: 'Task', context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if constraint is satisfied and return satisfaction level."""
        try:
            if self.constraint_type == ConstraintType.TEMPORAL:
                return self._check_temporal_constraint(task, context)
            elif self.constraint_type == ConstraintType.RESOURCE:
                return self._check_resource_constraint(task, context)
            elif self.constraint_type == ConstraintType.DEPENDENCY:
                return self._check_dependency_constraint(task, context)
            else:
                return True, 1.0  # Default to satisfied
        except Exception:
            return False, 0.0
    
    def _check_temporal_constraint(self, task: 'Task', context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check temporal constraints."""
        deadline = self.parameters.get('deadline')
        if deadline and task.estimated_duration:
            current_time = datetime.now(timezone.utc)
            time_available = (deadline - current_time).total_seconds()
            if time_available >= task.estimated_duration:
                return True, min(1.0, time_available / (task.estimated_duration * 2))
            else:
                return False, max(0.0, time_available / task.estimated_duration)
        return True, 1.0
    
    def _check_resource_constraint(self, task: 'Task', context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check resource constraints."""
        required_resources = task.required_resources
        available_resources = context.get('available_resources', {})
        
        satisfaction = 1.0
        for resource_type, required_amount in required_resources.items():
            available_amount = available_resources.get(resource_type, 0)
            if available_amount < required_amount:
                if self.hard_constraint:
                    return False, 0.0
                satisfaction = min(satisfaction, available_amount / required_amount)
        
        return satisfaction >= 0.5, satisfaction
    
    def _check_dependency_constraint(self, task: 'Task', context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check dependency constraints."""
        completed_tasks = context.get('completed_tasks', set())
        for dependency in task.dependencies:
            if dependency not in completed_tasks:
                return False, 0.0
        return True, 1.0


@dataclass
class Resource:
    """Represents an available resource."""
    resource_id: str
    resource_type: ResourceType
    capacity: float
    current_usage: float = 0.0
    reserved: float = 0.0
    cost_per_unit: float = 0.0
    availability_schedule: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_capacity(self) -> float:
        """Get currently available capacity."""
        return max(0.0, self.capacity - self.current_usage - self.reserved)
    
    def can_allocate(self, amount: float) -> bool:
        """Check if resource can allocate specified amount."""
        return self.available_capacity >= amount
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource if possible."""
        if self.can_allocate(amount):
            self.current_usage += amount
            return True
        return False
    
    def deallocate(self, amount: float) -> None:
        """Deallocate resource."""
        self.current_usage = max(0.0, self.current_usage - amount)


@dataclass
class Task:
    """Represents a planning task."""
    task_id: str
    name: str
    description: Optional[str] = None
    
    # Task type and execution
    task_type: str = "generic"  # skill, component, decision, etc.
    execution_target: Optional[str] = None  # skill/component name
    execution_method: Optional[str] = None  # method/function name
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Planning attributes
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 30.0  # seconds
    estimated_cost: float = 0.0
    complexity_score: float = 1.0
    
    # Dependencies and relationships
    dependencies: Set[str] = field(default_factory=set)
    soft_dependencies: Set[str] = field(default_factory=set)
    sub_tasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None
    
    # Resource requirements
    required_resources: Dict[ResourceType, float] = field(default_factory=dict)
    preferred_resources: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Constraints
    constraints: List[TaskConstraint] = field(default_factory=list)
    deadline: Optional[datetime] = None
    earliest_start: Optional[datetime] = None
    
    # Quality and preferences
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    status: TaskStatus = TaskStatus.PLANNED
    scheduled_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    actual_duration: Optional[float] = None
    success_probability: float = 0.8
    
    # Results and feedback
    result: Optional[Any] = None
    error: Optional[Exception] = None
    feedback_score: Optional[float] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Represents a complete execution plan."""
    plan_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Plan structure
    tasks: Dict[str, Task] = field(default_factory=dict)
    execution_order: List[List[str]] = field(default_factory=list)  # Batches of parallel tasks
    critical_path: List[str] = field(default_factory=list)
    
    # Plan metadata
    planning_strategy: PlanningStrategy = PlanningStrategy.ADAPTIVE
    estimated_total_duration: float = 0.0
    estimated_total_cost: float = 0.0
    success_probability: float = 0.0
    
    # Constraints and requirements
    global_constraints: List[TaskConstraint] = field(default_factory=list)
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Context and goals
    original_intent: Optional[str] = None
    user_goals: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Quality and optimization
    quality_score: float = 0.0
    optimization_score: float = 0.0
    adaptability_score: float = 0.0
    
    # Execution tracking
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1


class PlanningError(Exception):
    """Custom exception for planning operations."""
    
    def __init__(self, message: str, plan_id: Optional[str] = None, 
                 task_id: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.plan_id = plan_id
        self.task_id = task_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class PlanningHeuristic(ABC):
    """Abstract base class for planning heuristics."""
    
    @abstractmethod
    def estimate_cost(self, task: Task, context: Dict[str, Any]) -> float:
        """Estimate the cost of executing a task."""
        pass
    
    @abstractmethod
    def estimate_benefit(self, task: Task, context: Dict[str, Any]) -> float:
        """Estimate the benefit of completing a task."""
        pass
    
    @abstractmethod
    def estimate_success_probability(self, task: Task, context: Dict[str, Any]) -> float:
        """Estimate the probability of task success."""
        pass


class DefaultPlanningHeuristic(PlanningHeuristic):
    """Default implementation of planning heuristics."""
    
    def estimate_cost(self, task: Task, context: Dict[str, Any]) -> float:
        """Estimate task cost based on duration and resources."""
        base_cost = task.estimated_duration
        
        # Add resource costs
        for resource_type, amount in task.required_resources.items():
            resource_cost = amount * context.get(f"{resource_type.value}_cost", 1.0)
            base_cost += resource_cost
        
        # Complexity multiplier
        complexity_factor = 1 + (task.complexity_score - 1) * 0.5
        
        return base_cost * complexity_factor
    
    def estimate_benefit(self, task: Task, context: Dict[str, Any]) -> float:
        """Estimate task benefit based on priority and user value."""
        base_benefit = task.priority.value * 10
        
        # User preference bonus
        user_prefs = context.get('user_preferences', {})
        preference_bonus = user_prefs.get(task.task_type, 0) * 5
        
        # Goal alignment bonus
        goal_alignment = context.get('goal_alignment', {}).get(task.task_id, 0.5)
        alignment_bonus = goal_alignment * 20
        
        return base_benefit + preference_bonus + alignment_bonus
    
    def estimate_success_probability(self, task: Task, context: Dict[str, Any]) -> float:
        """Estimate success probability based on historical data and resources."""
        base_probability = task.success_probability
        
        # Historical performance
        history = context.get('task_history', {})
        if task.task_type in history:
            historical_success = history[task.task_type].get('success_rate', 0.8)
            base_probability = (base_probability + historical_success) / 2
        
        # Resource availability factor
        resource_availability = 1.0
        available_resources = context.get('available_resources', {})
        for resource_type, required in task.required_resources.items():
            available = available_resources.get(resource_type, 0)
            if available < required:
                resource_availability *= available / required
        
        return min(1.0, base_probability * resource_availability)


class TaskDecomposer:
    """Decomposes complex tasks into manageable sub-tasks."""
    
    def __init__(self, skill_registry: SkillRegistry, component_manager: ComponentManager,
                 knowledge_graph: KnowledgeGraph):
        self.skill_registry = skill_registry
        self.component_manager = component_manager
        self.knowledge_graph = knowledge_graph
        self.logger = get_logger(__name__)
    
    async def decompose_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a complex task into sub-tasks."""
        try:
            if task.complexity_score <= 2.0:
                return [task]  # Simple task, no decomposition needed
            
            # Try different decomposition strategies
            sub_tasks = []
            
            if task.task_type == "skill":
                sub_tasks = await self._decompose_skill_task(task, context)
            elif task.task_type == "component":
                sub_tasks = await self._decompose_component_task(task, context)
            elif task.task_type == "workflow":
                sub_tasks = await self._decompose_workflow_task(task, context)
            else:
                sub_tasks = await self._decompose_generic_task(task, context)
            
            # Set up parent-child relationships
            for sub_task in sub_tasks:
                sub_task.parent_task = task.task_id
                task.sub_tasks.append(sub_task.task_id)
            
            return sub_tasks if sub_tasks else [task]
            
        except Exception as e:
            self.logger.warning(f"Failed to decompose task {task.task_id}: {str(e)}")
            return [task]
    
    async def _decompose_skill_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a skill-based task."""
        skill_name = task.execution_target
        if not skill_name:
            return []
        
        # Get skill information
        skill_info = await self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            return []
        
        # Check if skill has sub-skills or dependencies
        sub_skills = skill_info.get('sub_skills', [])
        if sub_skills:
            sub_tasks = []
            for i, sub_skill in enumerate(sub_skills):
                sub_task = Task(
                    task_id=f"{task.task_id}_sub_{i}",
                    name=f"{task.name} - {sub_skill.get('name', f'Sub-skill {i}')}",
                    task_type="skill",
                    execution_target=sub_skill.get('skill_name'),
                    estimated_duration=sub_skill.get('estimated_duration', task.estimated_duration / len(sub_skills)),
                    priority=task.priority,
                    required_resources=sub_skill.get('resources', {}),
                    parameters=sub_skill.get('parameters', {})
                )
                sub_tasks.append(sub_task)
            
            return sub_tasks
        
        return []
    
    async def _decompose_component_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a component-based task."""
        component_name = task.execution_target
        if not component_name:
            return []
        
        # Get component information
        try:
            component = await self.component_manager.get_component(component_name)
            component_info = getattr(component, '_component_info', {})
            
            # Check for sub-operations
            sub_operations = component_info.get('sub_operations', [])
            if sub_operations:
                sub_tasks = []
                for i, operation in enumerate(sub_operations):
                    sub_task = Task(
                        task_id=f"{task.task_id}_op_{i}",
                        name=f"{task.name} - {operation.get('name', f'Operation {i}')}",
                        task_type="component",
                        execution_target=component_name,
                        execution_method=operation.get('method'),
                        estimated_duration=operation.get('duration', task.estimated_duration / len(sub_operations)),
                        priority=task.priority,
                        parameters=operation.get('parameters', {})
                    )
                    sub_tasks.append(sub_task)
                
                return sub_tasks
        
        except Exception as e:
            self.logger.warning(f"Failed to get component info for {component_name}: {str(e)}")
        
        return []
    
    async def _decompose_workflow_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a workflow task."""
        # This would integrate with workflow definitions
        workflow_steps = task.parameters.get('workflow_steps', [])
        if workflow_steps:
            sub_tasks = []
            for i, step in enumerate(workflow_steps):
                sub_task = Task(
                    task_id=f"{task.task_id}_step_{i}",
                    name=f"{task.name} - {step.get('name', f'Step {i}')}",
                    task_type=step.get('type', 'generic'),
                    execution_target=step.get('target'),
                    execution_method=step.get('method'),
                    estimated_duration=step.get('duration', 30.0),
                    priority=task.priority,
                    parameters=step.get('parameters', {})
                )
                
                # Set dependencies between steps
                if i > 0:
                    sub_task.dependencies.add(f"{task.task_id}_step_{i-1}")
                
                sub_tasks.append(sub_task)
            
            return sub_tasks
        
        return []
    
    async def _decompose_generic_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """Decompose a generic task using knowledge graph."""
        try:
            # Use knowledge graph to find task decomposition patterns
            decomposition_query = {
                'task_type': task.task_type,
                'complexity': task.complexity_score,
                'context': context
            }
            
            decomposition_patterns = await self.knowledge_graph.query_decomposition_patterns(decomposition_query)
            
            if decomposition_patterns:
                pattern = decomposition_patterns[0]  # Use best pattern
                sub_tasks = []
                
                for i, sub_pattern in enumerate(pattern.get('sub_tasks', [])):
                    sub_task = Task(
                        task_id=f"{task.task_id}_generic_{i}",
                        name=f"{task.name} - {sub_pattern.get('name', f'Sub-task {i}')}",
                        task_type=sub_pattern.get('type', 'generic'),
                        estimated_duration=sub_pattern.get('duration', task.estimated_duration / len(pattern['sub_tasks'])),
                        priority=task.priority,
                        complexity_score=sub_pattern.get('complexity', 1.0)
                    )
                    sub_tasks.append(sub_task)
                
                return sub_tasks
        
        except Exception as e:
            self.logger.warning(f"Knowledge graph decomposition failed: {str(e)}")
        
        return []


class ResourceManager:
    """Manages resource allocation and tracking."""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.allocations: Dict[str, Dict[str, float]] = defaultdict(dict)  # task_id -> resource_id -> amount
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    def register_resource(self, resource: Resource) -> None:
        """Register a resource for management."""
        with self.lock:
            self.resources[resource.resource_id] = resource
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get total available resources by type."""
        with self.lock:
            available = defaultdict(float)
            for resource in self.resources.values():
                available[resource.resource_type] += resource.available_capacity
            return dict(available)
    
    def can_allocate_resources(self, task: Task) -> bool:
        """Check if resources can be allocated for a task."""
        with self.lock:
            for resource_type, required_amount in task.required_resources.items():
                available_amount = sum(
                    r.available_capacity for r in self.resources.values()
                    if r.resource_type == resource_type
                )
                if available_amount < required_amount:
                    return False
            return True
    
    def allocate_resources(self, task: Task) -> bool:
        """Allocate resources for a task."""
        with self.lock:
            allocated_resources = {}
            
            try:
                # Try to allocate all required resources
                for resource_type, required_amount in task.required_resources.items():
                    remaining = required_amount
                    
                    # Find suitable resources
                    suitable_resources = [
                        r for r in self.resources.values()
                        if r.resource_type == resource_type and r.available_capacity > 0
                    ]
                    
                    # Sort by available capacity (largest first)
                    suitable_resources.sort(key=lambda r: r.available_capacity, reverse=True)
                    
                    for resource in suitable_resources:
                        if remaining <= 0:
                            break
                        
                        allocation_amount = min(remaining, resource.available_capacity)
                        if resource.allocate(allocation_amount):
                            allocated_resources[resource.resource_id] = allocation_amount
                            remaining -= allocation_amount
                    
                    if remaining > 0:
                        # Not enough resources, rollback
                        for resource_id, amount in allocated_resources.items():
                            self.resources[resource_id].deallocate(amount)
                        return False
                
                # Store allocation record
                self.allocations[task.task_id] = allocated_resources
                return True
                
            except Exception as e:
                # Rollback on error
                for resource_id, amount in allocated_resources.items():
                    self.resources[resource_id].deallocate(amount)
                self.logger.error(f"Resource allocation failed for task {task.task_id}: {str(e)}")
                return False
    
    def deallocate_resources(self, task: Task) -> None:
        """Deallocate resources for a completed task."""
        with self.lock:
            if task.task_id in self.allocations:
                for resource_id, amount in self.allocations[task.task_id].items():
                    if resource_id in self.resources:
                        self.resources[resource_id].deallocate(amount)
                del self.allocations[task.task_id]


class TaskPlanner:
    """
    Advanced Task Planning Engine for the AI Assistant.
    
    This planner provides intelligent task decomposition, scheduling, and optimization:
    - Goal-oriented task planning and decomposition
    - Multi-strategy planning algorithms (A*, hierarchical, adaptive)
    - Resource-aware scheduling and optimization
    - Constraint satisfaction and conflict resolution
    - Context-aware plan adaptation
    - Integration with skills, memory, and reasoning systems
    - Real-time plan modification and re-planning
    - Learning from execution feedback
    - Multi-user and collaborative planning
    """
    
    def __init__(self, container: Container):
        """
        Initialize the task planner.
        
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
        
        # Processing components
        self.intent_manager = container.get(IntentManager)
        self.language_chain = container.get(LanguageChain)
        self.entity_extractor = container.get(EntityExtractor)
        
        # Reasoning components
        self.logic_engine = container.get(LogicEngine)
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.decision_tree = container.get(DecisionTree)
        
        # Skills and components
        self.skill_registry = container.get(SkillRegistry)
        self.skill_validator = container.get(SkillValidator)
        self.component_manager = container.get(ComponentManager)
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.semantic_memory = container.get(SemanticMemory)
        
        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Planning components
        self.task_decomposer = TaskDecomposer(
            self.skill_registry, self.component_manager, self.knowledge_graph
        )
        self.resource_manager = ResourceManager()
        self.heuristic = DefaultPlanningHeuristic()
        
        # State management
        self.active_plans: Dict[str, ExecutionPlan] = {}
        self.plan_history: deque = deque(maxlen=1000)
        self.planning_cache: Dict[str, ExecutionPlan] = {}
        self.planning_lock = asyncio.Lock()
        
        # Configuration
        self.default_strategy = PlanningStrategy(
            self.config.get("planning.default_strategy", "adaptive")
        )
        self.max_planning_time = self.config.get("planning.max_time_seconds", 30.0)
        self.enable_learning = self.config.get("planning.enable_learning", True)
        self.enable_caching = self.config.get("planning.enable_caching", True)
        self.max_task_depth = self.config.get("planning.max_task_depth", 5)
        
        # Performance tracking
        self.planning_stats: Dict[str, Any] = defaultdict(int)
        self.execution_feedback: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize resources
        self._setup_default_resources()
        self._setup_monitoring()
        
        # Register health check
        self.health_check.register_component("task_planner", self._health_check_callback)
        
        self.logger.info("TaskPlanner initialized successfully")

    def _setup_default_resources(self) -> None:
        """Setup default system resources."""
        try:
            # CPU resources
            cpu_cores = self.config.get("system.cpu_cores", 4)
            self.resource_manager.register_resource(Resource(
                resource_id="system_cpu",
                resource_type=ResourceType.CPU,
                capacity=cpu_cores * 100.0,  # 100% per core
                cost_per_unit=0.01
            ))
            
            # Memory resources
            memory_gb = self.config.get("system.memory_gb", 8)
            self.resource_manager.register_resource(Resource(
                resource_id="system_memory",
                resource_type=ResourceType.MEMORY,
                capacity=memory_gb * 1024.0,  # MB
                cost_per_unit=0.001
            ))
            
            # Network resources
            self.resource_manager.register_resource(Resource(
                resource_id="system_network",
                resource_type=ResourceType.NETWORK,
                capacity=1000.0,  # Arbitrary units
                cost_per_unit=0.005
            ))
            
            # Time resource (user attention)
            self.resource_manager.register_resource(Resource(
                resource_id="user_attention",
                resource_type=ResourceType.USER_ATTENTION,
                capacity=100.0,  # Percentage
                cost_per_unit=0.1
            ))
            
        except Exception as e:
            self.logger.warning(f"Failed to setup default resources: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register planning metrics
            self.metrics.register_counter("plans_created_total")
            self.metrics.register_counter("plans_executed_total")
            self.metrics.register_counter("plans_failed_total")
            self.metrics.register_histogram("planning_duration_seconds")
            self.metrics.register_histogram("plan_execution_duration_seconds")
            self.metrics.register_gauge("active_plans")
            self.metrics.register_counter("tasks_planned_total")
            self.metrics.register_counter("tasks_executed_total")
            self.metrics.register_histogram("task_execution_duration_seconds")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the task planner."""
        try:
            # Initialize knowledge graph with planning patterns
            await self._load_planning_patterns()
            
            # Start background tasks
            asyncio.create_task(self._plan_optimization_loop())
            asyncio.create_task(self._resource_monitoring_loop())
            
            if self.enable_learning:
                asyncio.create_task(self._learning_update_loop())
            
            self.logger.info("TaskPlanner initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TaskPlanner: {str(e)}")
            raise PlanningError(f"Initialization failed: {str(e)}")

    async def _load_planning_patterns(self) -> None:
        """Load planning patterns and heuristics from knowledge base."""
        try:
            # Load common task patterns
            patterns = {
                'question_answering': {
                    'tasks': [
                        {'type': 'component', 'target': 'intent_manager', 'method': 'detect_intent'},
                        {'type': 'skill', 'target': 'knowledge_search'},
                        {'type': 'skill', 'target': 'text_generation'}
                    ],
                    'dependencies': [('0', '1'), ('1', '2')]
                },
                'multimodal_processing': {
                    'tasks': [
                        {'type': 'component', 'target': 'speech_processor', 'method': 'process'},
                        {'type': 'component', 'target': 'vision_processor', 'method': 'process'},
                        {'type': 'skill', 'target': 'multimodal_fusion'},
                        {'type': 'skill', 'target': 'response_generation'}
                    ],
                    'dependencies': [('0', '2'), ('1', '2'), ('2', '3')]
                }
            }
            
            # Store patterns in knowledge graph
            for pattern_name, pattern_data in patterns.items():
                await self.knowledge_graph.store_planning_pattern(pattern_name, pattern_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to load planning patterns: {str(e)}")

    @handle_exceptions
    async def create_plan_for_intent(
        self,
        intent: str,
        context: Dict[str, Any],
        session_id: str,
        user_id: Optional[str] = None,
        strategy: Optional[PlanningStrategy] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for a detected intent.
        
        Args:
            intent: Detected user intent
            context: Context information
            session_id: Session identifier
            user_id: Optional user identifier
            strategy: Planning strategy to use
            
        Returns:
            Execution plan
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        async with self.planning_lock:
            try:
                with self.tracer.trace("plan_creation") as span:
                    span.set_attributes({
                        "intent": intent,
                        "session_id": session_id,
                        "user_id": user_id or "anonymous",
                        "strategy": strategy.value
                    })
                    
                    # Emit planning started event
                    await self.event_bus.emit(TaskPlanningStarted(
                        session_id=session_id,
                        intent=intent,
                        strategy=strategy.value
                    ))
                    
                    # Create plan
                    plan = ExecutionPlan(
                        plan_id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        original_intent=intent,
                        context=context.copy(),
                        planning_strategy=strategy
                    )
                    
                    # Extract goals from intent and context
                    plan.user_goals = await self._extract_goals(intent, context)
                    
                    # Generate initial tasks based on intent
                    initial_tasks = await self._generate_tasks_for_intent(intent, context)
                    
                    # Decompose complex tasks
                    all_tasks = []
                    for task in initial_tasks:
                        decomposed = await self.task_decomposer.decompose_task(task, context)
                        all_tasks.extend(decomposed)
                    
                    # Add tasks to plan
                    for task in all_tasks:
                        plan.tasks[task.task_id] = task
                    
                    # Apply constraints
                    await self._apply_constraints(plan, context)
                    
                    # Optimize plan based on strategy
                    await self._optimize_plan(plan, strategy)
                    
                    # Calculate plan metrics
                    await self._calculate_plan_metrics(plan)
                    
                    # Store plan
                    self.active_plans[plan.plan_id] = plan
                    
                    planning_time = time.time() - start_time
                    
                    # Update metrics
                    self.metrics.increment("plans_created_total")
                    self.metrics.record("planning_duration_seconds", planning_time)
                    self.metrics.set("active_plans", len(self.active_plans))
                    
                    # Emit planning completed event
                    await self.event_bus.emit(TaskPlanningCompleted(
                        session_id=session_id,
                        plan_id=plan.plan_id,
                        task_count=len(plan.tasks),
                        planning_time=planning_time
                    ))
                    
                    self.logger.info(
                        f"Created plan {plan.plan_id} for intent '{intent}' "
                        f"with {len(plan.tasks)} tasks in {planning_time:.2f}s"
                    )
                    
                    return plan
                    
            except Exception as e:
                planning_time = time.time() - start_time
                
                await self.event_bus.emit(TaskPlanningFailed(
                    session_id=session_id,
                    intent=intent,
                    error_message=str(e),
                    planning_time=planning_time
                ))
                
                self.logger.error(f"Failed to create plan for intent '{intent}': {str(e)}")
                raise PlanningError(f"Planning failed: {str(e)}")

    async def _extract_goals(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """Extract user goals from intent and context."""
        goals = []
        
        # Primary goal from intent
        goals.append(intent)
        
        # Extract additional goals from entities
        entities = context.get('entities', [])
        for entity in entities:
            if entity.get('type') == 'goal':
                goals.append(entity.get('value'))
        
        # Extract goals from user preferences
        user_prefs = context.get('user_preferences', {})
        user_goals = user_prefs.get('goals', [])
        goals.extend(user_goals)
        
        return [goal for goal in goals if goal]

    async def _generate_tasks_for_intent(
        self,
        intent: str,
        context: Dict[str, Any]
    ) -> List[Task]:
        """Generate initial tasks based on intent."""
        tasks = []
        
        try:
            # Get planning pattern from knowledge graph
            pattern = await self.knowledge_graph.get_planning_pattern(intent)
            
            if pattern:
                # Create tasks from pattern
                for i, task_def in enumerate(pattern.get('tasks', [])):
                    task = Task(
                        task_id=f"task_{i}",
                        name=task_def.get('name', f"Task {i}"),
                        task_type=task_def.get('type', 'generic'),
                        execution_target=task_def.get('target'),
                        execution_method=task_def.get('method'),
                        estimated_duration=task_def.get('duration', 30.0),
                        parameters=task_def.get('parameters', {}),
                        required_resources=self._parse_resource_requirements(task_def.get('resources', {}))
                    )
                    tasks.append(task)
                
                # Set dependencies
                for dep in pattern.get('dependencies', []):
                    if len(dep) == 2:
                        source_idx, target_idx = dep
                        if source_idx < len(tasks) and target_idx < len(tasks):
                            tasks[target_idx].dependencies.add(tasks[source_idx].task_id)
            else:
                # Fallback: create generic task for intent
                task = Task(
                    task_id="generic_task",
                    name=f"Handle {intent}",
                    task_type="generic",
                    estimated_duration=60.0,
                    parameters={'intent': intent, 'context': context}
                )
                tasks.append(task)
            
        except Exception as e:
            self.logger.warning(f"Failed to generate tasks for intent '{intent}': {str(e)}")
            
            # Fallback task
            task = Task(
                task_id="fallback_task",
                name=f"Process {intent}",
                task_type="generic",
                estimated_duration=30.0
            )
            tasks.append(task)
        
        return tasks

    def _parse_resource_requirements(self, resources_dict: Dict[str, Any]) -> Dict[ResourceType, float]:
        """Parse resource requirements from dictionary."""
        requirements = {}
        
        for resource_name, amount in resources_dict.items():
            try:
                resource_type = ResourceType(resource_name.lower())
                requirements[resource_type] = float(amount)
            except (ValueError, TypeError):
                continue
        
        return requirements

    async def _apply_constraints(self, plan: ExecutionPlan, context: Dict[str, Any]) -> None:
        """Apply constraints to the execution plan."""
        # Time constraints
        max_execution_time = context.get('max_execution_time')
        if max_execution_time:
            constraint = TaskConstraint(
                constraint_id="global_time_limit",
                constraint_type=ConstraintType.TEMPORAL,
                description=f"Complete within {max_execution_time} seconds",
                parameters={'max_time': max_execution_time},
                hard_constraint=True
            )
            plan.global_constraints.append(constraint)
        
        # Resource constraints
        resource_limits = context.get('resource_limits', {})
        for resource_type, limit in resource_limits.items():
            constraint = TaskConstraint(
                constraint_id=f"resource_limit_{resource_type}",
                constraint_type=ConstraintType.RESOURCE,
                description=f"Use at most {limit} units of {resource_type}",
                parameters={'resource_type': resource_type, 'limit': limit},
                hard_constraint=True
            )
            plan.global_constraints.append(constraint)
        
        # User preference constraints
        user_prefs = context.get('user_preferences', {})
        if 'quality_preference' in user_prefs:
            quality_pref = user_prefs['quality_preference']
            constraint = TaskConstraint(
                constraint_id="quality_preference",
                constraint_type=ConstraintType.QUALITY,
                description=f"Maintain {quality_pref} quality level",
                parameters={'quality_level': quality_pref},
                hard_constraint=False,
                weight=2.0
            )
            plan.global_constraints.append(constraint)

    async def _optimize_plan(self, plan: ExecutionPlan, strategy: PlanningStrategy) -> None:
        """Optimize the execution plan using the specified strategy."""
        if strategy == PlanningStrategy.GREEDY:
            await self._optimize_greedy(plan)
        elif strategy == PlanningStrategy.A_STAR:
            await self._optimize_a_star(plan)
        elif strategy == PlanningStrategy.HIERARCHICAL:
            await self._optimize_hierarchical(plan)
        elif strategy == PlanningStrategy.ADAPTIVE:
            await self._optimize_adaptive(plan)
        else:
            await self._optimize_greedy(plan)  # Default fallback

    async def _optimize_greedy(self, plan: ExecutionPlan) -> None:
        """Optimize plan using greedy approach."""
        # Sort tasks by priority and benefit/cost ratio
        tasks = list(plan.tasks.values())
        
        def task_score(task: Task) -> float:
            benefit = self.heuristic.estimate_benefit(task, plan.context)
            cost = self.heuristic.estimate_cost(task, plan.context)
            priority_bonus = task.priority.value * 10
            return (benefit + priority_bonus) / max(cost, 1.0)
        
        # Create execution order based on dependencies and scores
        plan.execution_order = await self._create_execution_order(tasks, task_score)

    async def _optimize_a_star(self, plan: ExecutionPlan) -> None:
        """Optimize plan using A* search algorithm."""
        # Implement A* search for optimal task ordering
        tasks = list(plan.tasks.values())
        
        # Create search space
        initial_state = frozenset()  # No tasks completed
        goal_state = frozenset(task.task_id for task in tasks)  # All tasks completed
        
        # A* search implementation would go here
        # For now, fall back to greedy
        await self._optimize_greedy(plan)

    async def _optimize_hierarchical(self, plan: ExecutionPlan) -> None:
        """Optimize plan using hierarchical approach."""
        # Group tasks by hierarchy level and optimize within levels
        task_levels = defaultdict(list)
        
        for task in plan.tasks.values():
            level = len(task.dependencies)  # Simple level calculation
            task_levels[level].append(task)
        
        execution_order = []
        for level in sorted(task_levels.keys()):
            level_tasks = task_levels[level]
            
            # Optimize within level
            level_tasks.sort(key=lambda t: (
                t.priority.value,
                -self.heuristic.estimate_cost(t, plan.context)
            ), reverse=True)
            
            execution_order.append([t.task_id for t in level_tasks])
        
        plan.execution_order = execution_order

    async def _optimize_adaptive(self, plan: ExecutionPlan) -> None:
        """Optimize plan using adaptive approach based on context."""
        # Choose optimization strategy based on plan characteristics
        task_count = len(plan.tasks)
        complexity = sum(task.complexity_score for task in plan.tasks.values())
        
        if task_count <= 3 and complexity <= 5:
            await self._optimize_greedy(plan)
        elif task_count <= 10:
            await self._optimize_hierarchical(plan)
        else:
            await self._optimize_a_star(plan)

    async def _create_execution_order(
        self,
        tasks: List[Task],
        score_func: Callable[[Task], float]
    ) -> List[List[str]]:
        """Create execution order respecting dependencies."""
        execution_order = []
        completed = set()
        task_dict = {task.task_id: task for task in tasks}
        
        while len(completed) < len(tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in tasks:
                if (task.task_id not in completed and 
                    task.dependencies.issubset(completed)):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Deadlock detection - should not happen with valid dependencies
                remaining_tasks = [t.task_id for t in tasks if t.task_id not in completed]
                self.logger.warning(f"Potential deadlock with tasks: {remaining_tasks}")
                break
            
            # Sort by score and add to execution order
            ready_tasks.sort(key=score_func, reverse=True)
            batch = [task.task_id for task in ready_tasks]
            execution_order.append(batch)
            
            # Mark as completed
            completed.update(batch)
        
        return execution_order

    async def _calculate_plan_metrics(self, plan: ExecutionPlan) -> None:
        """Calculate quality and performance metrics for the plan."""
        tasks = plan.tasks.values()
        
        # Estimated total duration (critical path)
        critical_path_duration = 0.0
        for batch in plan.execution_order:
            batch_duration = max(
                plan.tasks[task_id].estimated_duration
                for task_id in batch
            ) if batch else 0.0
            critical_path_duration += batch_duration
        
        plan.estimated_total_duration = critical_path_duration
        
        # Estimated total cost
        plan.estimated_total_cost = sum(
            self.heuristic.estimate_cost(task, plan.context)
            for task in tasks
        )
        
        # Success probability (product of individual probabilities)
        success_prob = 1.0
        for task in tasks:
            task_prob = self.heuristic.estimate_success_probability(task, plan.context)
            success_prob *= task_prob
        
        plan.success_probability = success_prob
        
        # Quality score based on constraint satisfaction
        quality_score = 1.0
        for constraint in plan.global_constraints:
            for task in tasks:
                satisfied, satisfaction_level = constraint.check_constraint(task, plan.context)
                if not satisfied and constraint.hard_constraint:
                    quality_score = 0.0
                    break
                quality_score *= satisfaction_level
            if quality_score == 0.0:
                break
        
        plan.quality_score = quality_score
        
        # Optimization score (benefit/cost ratio)
        total_benefit = sum(
            self.heuristic.estimate_benefit(task, plan.context)
            for task in tasks
        )
        plan.optimization_score = total_benefit / max(plan.estimated_total_cost, 1.0)

    @handle_exceptions
    async def create_workflow_for_intent(
        self,
        intent: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a workflow definition for a detected intent.
        
        Args:
            intent: Detected user intent
            context: Context information
            
        Returns:
            Workflow definition dictionary
        """
        try:
            # Create execution plan
            plan = await self.create_plan_for_intent(
                intent, context, context.get('session_id', 'default')
            )
            
            # Convert plan to workflow format
            workflow_def = {
                'workflow_id': f"intent_{intent}_{int(time.time())}",
                'name': f"Workflow for {intent}",
                'description': f"Auto-generated workflow for intent: {intent}",
                'tasks': [],
                'dependencies': []
            }
            
            # Convert tasks
            for task in plan.tasks.values():
                task_def = {
                    'task_id': task.task_id,
                    'name': task.name,
                    'type': task.task_type,
                    'duration': task.estimated_duration,
                    'parameters': task.parameters
                }
                
                if task.execution_target:
                    if task.task_type == 'skill':
                        task_def['skill_name'] = task.execution_target
                    elif task.task_type == 'component':
                        task_def['component_name'] = task.execution_target
                        task_def['function_name'] = task.execution_method or 'process'
                
                workflow_def['tasks'].append(task_def)
            
            # Convert dependencies
            for task in plan.tasks.values():
                for dep in task.dependencies:
                    workflow_def['dependencies'].append((dep, task.task_id))
            
            return workflow_def
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow for intent '{intent}': {str(e)}")
            raise PlanningError(f"Workflow creation failed: {str(e)}")

    @handle_exceptions
    async def plan_workflow_execution(
        self,
        workflow_def: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[List[str]]:
        """
        Plan the execution order for a workflow definition.
        
        Args:
            workflow_def: Workflow definition
            context: Execution context
            
        Returns:
            List of execution batches (task ID lists)
        """
        try:
            # Convert workflow to tasks
            tasks = []
            for task_def in workflow_def.get('steps', []):
                task = Task(
                    task_id=task_def.get('step_id', task_def.get('task_id')),
                    name=task_def.get('name', 'Workflow Step'),
                    task_type=task_def.get('step_type', task_def.get('type', 'generic')),
                    execution_target=task_def.get('skill_name') or task_def.get('component_name'),
                    execution_method=task_def.get('function_name'),
                    estimated_duration=task_def.get('timeout_seconds', 30.0),
                    parameters=task_def.get('parameters', {})
                )
                
                # Set dependencies
                for dep in task_def.get('dependencies', []):
                    task.dependencies.add(dep)
                
                tasks.append(task)
            
            # Create execution order
            execution_order = await self._create_execution_order(
                tasks,
                lambda t: t.priority.value
            )
            
            return execution_order
            
        except Exception as e:
            self.logger.error(f"Failed to plan workflow execution: {str(e)}")
            return []

    @handle_exceptions
    async def adapt_plan(
        self,
        plan_id: str,
        execution_feedback: Dict[str, Any],
        context_changes: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Adapt an existing plan based on execution feedback and context changes.
        
        Args:
            plan_id: Plan identifier
            execution_feedback: Feedback from task execution
            context_changes: Changes in execution context
            
        Returns:
            Updated execution plan
        """
        if plan_id not in self.active_plans:
            raise PlanningError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        
        try:
            # Update context
            plan.context.update(context_changes)
            
            # Process execution feedback
            completed_tasks = execution_feedback.get('completed_tasks', set())
            failed_tasks = execution_feedback.get('failed_tasks', set())
            
            plan.completed_tasks.update(completed_tasks)
            plan.failed_tasks.update(failed_tasks)
            
            # Re-plan remaining tasks
            remaining_tasks = [
                task for task_id, task in plan.tasks.items()
                if task_id not in completed_tasks and task_id not in failed_tasks
            ]
            
            if remaining_tasks:
                # Re-optimize remaining execution
                await self._optimize_plan(plan, plan.planning_strategy)
                
                # Recalculate metrics
                await self._calculate_plan_metrics(plan)
                
                # Update version
                plan.version += 1
                plan.last_modified = datetime.now(timezone.utc)
                
                # Emit adaptation event
                await self.event_bus.emit(PlanAdapted(
                    plan_id=plan_id,
                    session_id=plan.session_id,
                    changes=list(context_changes.keys()),
                    remaining_tasks=len(remaining_tasks)
                ))
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to adapt plan {plan_id}: {str(e)}")
            raise PlanningError(f"Plan adaptation failed: {str(e)}")

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get the status of an execution plan."""
        if plan_id not in self.active_plans:
            raise PlanningError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        
        total_tasks = len(plan.tasks)
        completed_tasks = len(plan.completed_tasks)
        failed_tasks = len(plan.failed_tasks)
        remaining_tasks = total_tasks - completed_tasks - failed_tasks
        
        return {
            'plan_id': plan_id,
            'session_id': plan.session_id,
            'user_id': plan.user_id,
            'original_intent': plan.original_intent,
            'strategy': plan.planning_strategy.value,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'remaining_tasks': remaining_tasks,
            'progress': completed_tasks / max(total_tasks, 1),
            'estimated_duration': plan.estimated_total_duration,
            'estimated_cost': plan.estimated_total_cost,
            'success_probability': plan.success_probability,
            'quality_score': plan.quality_score,
            'created_at': plan.created_at.isoformat(),
            'last_modified': plan.last_modified.isoformat(),
            'version': plan.version
        }

    def list_active_plans(self) -> List[Dict[str, Any]]:
        """List all active execution plans."""
        return [
            self.get_plan_status(plan_id)
            for plan_id in self.active_plans.keys()
        ]

    async def _plan_optimization_loop(self) -> None:
        """Background task for continuous plan optimization."""
        while True:
            try:
                # Optimize active plans
                for plan_id, plan in list(self.active_plans.items()):
                    try:
                        # Check if plan needs optimization
                        if self._needs_optimization(plan):
                            await self._optimize_plan(plan, plan.planning_strategy)
                            await self._calculate_plan_metrics(plan)
                            
                            await self.event_bus.emit(PlanOptimized(
                                plan_id=plan_id,
                                session_id=plan.session_id,
                                optimization_score=plan.optimization_score
                            ))
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to optimize plan {plan_id}: {str(e)}")
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                self.logger.error(f"Error in plan optimization loop: {str(e)}")
                await asyncio.sleep(60)

    def _needs_optimization(self, plan: ExecutionPlan) -> bool:
        """Check if a plan needs optimization."""
        # Optimize if quality score is low
        if plan.quality_score < 0.7:
            return True
        
        # Optimize if success probability is low
        if plan.success_probability < 0.6:
            return True
        
        # Optimize if plan is old and has new context
        plan_age = (datetime.now(timezone.utc) - plan.last_modified).total_seconds()
        if plan_age > 300:  # 5 minutes
            return True
        
        return False

    async def _resource_monitoring_loop(self) -> None:
        """Background task for resource monitoring."""
        while True:
            try:
                # Update resource availability
                available_resources = self.resource_manager.get_available_resources()
                
                # Check for resource constraints
                for plan in self.active_plans.values():
                    for task in plan.tasks.values():
                        if task.status == TaskStatus.READY:
                            if not self.resource_manager.can_allocate_resources(task):
                                task.status = TaskStatus.BLOCKED
                                
                                await self.event_bus.emit(ConstraintViolationDetected(
                                    plan_id=plan.plan_id,
                                    task_id=task.task_id,
                                    constraint_type="resource",
                                    description="Insufficient resources available"
                                ))
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")
                await asyncio.sleep(30)

    async def _learning_update_loop(self) -> None:
        """Background task for learning updates."""
        while True:
            try:
                # Collect execution feedback
                if self.execution_feedback:
                    # Update planning
