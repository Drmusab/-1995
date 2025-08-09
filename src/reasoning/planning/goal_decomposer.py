"""
Advanced Goal Decomposition Engine for AI Assistant
Author: Drmusab
Last Modified: 2025-06-13 08:00:00 UTC

This module provides intelligent goal decomposition and hierarchical planning for the AI assistant,
breaking down complex user goals into manageable sub-goals and action sequences, while maintaining
coherence with user intent and system capabilities.
"""

import copy
import hashlib
import heapq
import inspect
import json
import logging
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import asyncio
import numpy as np

from src.assistant.core import ComponentManager

# Assistant components
from src.assistant.core import CoreAssistantEngine
from src.assistant.core import InteractionHandler
from src.assistant.core import SessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ErrorOccurred,
    GoalAbandoned,
    GoalAchieved,
    GoalAdapted,
    GoalConflictDetected,
    GoalConflictResolved,
    GoalDecomposed,
    GoalDecompositionCompleted,
    GoalDecompositionFailed,
    GoalDecompositionStarted,
    GoalDependencyDetected,
    GoalHierarchyUpdated,
    GoalPrioritized,
    StrategySelected,
    SubGoalGenerated,
    SystemStateChanged,
)
from src.core.health_check import HealthCheck

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.processing.natural_language.entity_extractor import EntityExtractor

# Processing components
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.reasoning.decision_making.decision_tree import DecisionTree
from src.reasoning.knowledge_graph import KnowledgeGraph

# Reasoning components
from src.reasoning.logic_engine import LogicEngine
from src.reasoning.planning.task_planner import ExecutionPlan, Task, TaskPlanner, TaskPriority
from src.skills.skill_factory import SkillFactory

# Skills and components
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator


class GoalType(Enum):
    """Types of goals in the system."""

    PRIMARY = "primary"  # Main user goal
    SECONDARY = "secondary"  # Supporting goal
    INSTRUMENTAL = "instrumental"  # Goal to achieve another goal
    MAINTENANCE = "maintenance"  # System maintenance goal
    CONSTRAINT = "constraint"  # Constraint-based goal
    OPTIMIZATION = "optimization"  # Performance optimization goal
    EXPLORATION = "exploration"  # Learning/discovery goal
    CREATIVE = "creative"  # Creative/generative goal
    ANALYTICAL = "analytical"  # Analysis/reasoning goal
    INTERACTIVE = "interactive"  # User interaction goal


class GoalStatus(Enum):
    """Goal achievement status."""

    IDENTIFIED = "identified"
    DECOMPOSED = "decomposed"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    PARTIALLY_ACHIEVED = "partially_achieved"
    ACHIEVED = "achieved"
    FAILED = "failed"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    REFINED = "refined"


class DecompositionStrategy(Enum):
    """Strategies for goal decomposition."""

    TEMPORAL = "temporal"  # Time-based decomposition
    FUNCTIONAL = "functional"  # Function-based decomposition
    HIERARCHICAL = "hierarchical"  # Hierarchy-based decomposition
    RESOURCE_BASED = "resource_based"  # Resource-driven decomposition
    CONSTRAINT_BASED = "constraint_based"  # Constraint-driven decomposition
    USER_PREFERENCE = "user_preference"  # User preference driven
    SKILL_BASED = "skill_based"  # Available skills driven
    CONTEXT_ADAPTIVE = "context_adaptive"  # Context-aware decomposition
    COLLABORATIVE = "collaborative"  # Multi-agent decomposition
    ITERATIVE = "iterative"  # Iterative refinement


class GoalPriority(Enum):
    """Goal priority levels."""

    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    CRITICAL = 5
    URGENT = 6


class ConflictType(Enum):
    """Types of goal conflicts."""

    RESOURCE_CONFLICT = "resource_conflict"  # Competing for same resources
    TEMPORAL_CONFLICT = "temporal_conflict"  # Time-based conflicts
    LOGICAL_CONFLICT = "logical_conflict"  # Logically contradictory
    VALUE_CONFLICT = "value_conflict"  # Conflicting values/preferences
    DEPENDENCY_CONFLICT = "dependency_conflict"  # Circular dependencies
    CONSTRAINT_CONFLICT = "constraint_conflict"  # Constraint violations


@dataclass
class GoalMetrics:
    """Metrics for goal achievement and quality."""

    achievability_score: float = 0.0
    complexity_score: float = 0.0
    resource_intensity: float = 0.0
    time_sensitivity: float = 0.0
    user_value: float = 0.0
    system_impact: float = 0.0
    risk_score: float = 0.0
    confidence_score: float = 0.0
    success_probability: float = 0.0
    estimated_effort: float = 0.0


@dataclass
class GoalConstraint:
    """Constraint on goal achievement."""

    constraint_id: str
    constraint_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    hard_constraint: bool = True
    weight: float = 1.0
    violation_penalty: float = 100.0

    def check_satisfaction(self, goal: "Goal", context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if constraint is satisfied for the goal."""
        try:
            if self.constraint_type == "temporal":
                return self._check_temporal_constraint(goal, context)
            elif self.constraint_type == "resource":
                return self._check_resource_constraint(goal, context)
            elif self.constraint_type == "dependency":
                return self._check_dependency_constraint(goal, context)
            elif self.constraint_type == "quality":
                return self._check_quality_constraint(goal, context)
            else:
                return True, 1.0
        except Exception:
            return False, 0.0

    def _check_temporal_constraint(
        self, goal: "Goal", context: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Check temporal constraints."""
        deadline = self.parameters.get("deadline")
        if deadline and goal.estimated_completion_time:
            current_time = datetime.now(timezone.utc)
            time_available = (deadline - current_time).total_seconds()
            if time_available >= goal.estimated_completion_time:
                return True, min(1.0, time_available / (goal.estimated_completion_time * 2))
            else:
                return False, max(0.0, time_available / goal.estimated_completion_time)
        return True, 1.0

    def _check_resource_constraint(
        self, goal: "Goal", context: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Check resource constraints."""
        required_resources = goal.resource_requirements
        available_resources = context.get("available_resources", {})

        satisfaction = 1.0
        for resource_type, required_amount in required_resources.items():
            available_amount = available_resources.get(resource_type, 0)
            if available_amount < required_amount:
                if self.hard_constraint:
                    return False, 0.0
                satisfaction = min(satisfaction, available_amount / required_amount)

        return satisfaction >= 0.5, satisfaction

    def _check_dependency_constraint(
        self, goal: "Goal", context: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Check dependency constraints."""
        achieved_goals = context.get("achieved_goals", set())
        for dependency in goal.dependencies:
            if dependency not in achieved_goals:
                return False, 0.0
        return True, 1.0

    def _check_quality_constraint(
        self, goal: "Goal", context: Dict[str, Any]
    ) -> Tuple[bool, float]:
        """Check quality constraints."""
        min_quality = self.parameters.get("min_quality", 0.7)
        expected_quality = goal.quality_requirements.get("overall_quality", 0.8)
        if expected_quality >= min_quality:
            return True, expected_quality / min_quality
        else:
            return self.hard_constraint is False, expected_quality / min_quality


@dataclass
class Goal:
    """Represents a goal in the hierarchical goal structure."""

    goal_id: str
    description: str
    goal_type: GoalType = GoalType.PRIMARY
    priority: GoalPriority = GoalPriority.NORMAL

    # Hierarchical structure
    parent_goal: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    level: int = 0  # Depth in hierarchy

    # Goal properties
    success_criteria: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    user_intent: Optional[str] = None
    domain: Optional[str] = None

    # Temporal aspects
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    estimated_completion_time: float = 300.0  # seconds
    actual_start_time: Optional[datetime] = None
    actual_completion_time: Optional[datetime] = None

    # Dependencies and relationships
    dependencies: Set[str] = field(default_factory=set)
    conflicts_with: Set[str] = field(default_factory=set)
    enables: Set[str] = field(default_factory=set)
    requires: Set[str] = field(default_factory=set)

    # Resources and constraints
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    constraints: List[GoalConstraint] = field(default_factory=list)
    quality_requirements: Dict[str, float] = field(default_factory=dict)

    # Execution
    status: GoalStatus = GoalStatus.IDENTIFIED
    progress: float = 0.0
    assigned_tasks: List[str] = field(default_factory=list)
    execution_strategy: Optional[str] = None

    # Metrics and evaluation
    metrics: GoalMetrics = field(default_factory=GoalMetrics)
    achievement_evidence: List[Dict[str, Any]] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)

    # Learning and adaptation
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_leaf_goal(self) -> bool:
        """Check if this is a leaf goal (no sub-goals)."""
        return len(self.sub_goals) == 0

    def is_root_goal(self) -> bool:
        """Check if this is a root goal (no parent)."""
        return self.parent_goal is None

    def can_be_achieved(self, context: Dict[str, Any]) -> bool:
        """Check if goal can be achieved given current context."""
        # Check constraints
        for constraint in self.constraints:
            satisfied, _ = constraint.check_satisfaction(self, context)
            if not satisfied and constraint.hard_constraint:
                return False

        # Check dependencies
        achieved_goals = context.get("achieved_goals", set())
        if not self.dependencies.issubset(achieved_goals):
            return False

        return True


@dataclass
class GoalHierarchy:
    """Represents a hierarchical structure of goals."""

    hierarchy_id: str
    session_id: str
    user_id: Optional[str] = None

    # Goal structure
    goals: Dict[str, Goal] = field(default_factory=dict)
    root_goals: Set[str] = field(default_factory=set)
    goal_levels: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))

    # Execution tracking
    active_goals: Set[str] = field(default_factory=set)
    achieved_goals: Set[str] = field(default_factory=set)
    failed_goals: Set[str] = field(default_factory=set)

    # Conflict management
    conflicts: List[Tuple[str, str, ConflictType]] = field(default_factory=list)
    conflict_resolutions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metrics
    overall_progress: float = 0.0
    total_estimated_time: float = 0.0
    actual_time_spent: float = 0.0
    success_rate: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def add_goal(self, goal: Goal) -> None:
        """Add a goal to the hierarchy."""
        self.goals[goal.goal_id] = goal

        if goal.is_root_goal():
            self.root_goals.add(goal.goal_id)

        self.goal_levels[goal.level].add(goal.goal_id)
        self.last_updated = datetime.now(timezone.utc)

    def get_goals_at_level(self, level: int) -> List[Goal]:
        """Get all goals at a specific hierarchy level."""
        goal_ids = self.goal_levels.get(level, set())
        return [self.goals[goal_id] for goal_id in goal_ids if goal_id in self.goals]

    def get_leaf_goals(self) -> List[Goal]:
        """Get all leaf goals (actionable goals)."""
        return [goal for goal in self.goals.values() if goal.is_leaf_goal()]

    def get_ready_goals(self) -> List[Goal]:
        """Get goals that are ready to be executed."""
        ready = []
        for goal in self.goals.values():
            if goal.status == GoalStatus.PLANNED and goal.dependencies.issubset(
                self.achieved_goals
            ):
                ready.append(goal)
        return ready

    def calculate_progress(self) -> float:
        """Calculate overall progress of the goal hierarchy."""
        if not self.goals:
            return 0.0

        total_weight = 0.0
        weighted_progress = 0.0

        for goal in self.goals.values():
            weight = goal.priority.value + 1  # Convert to weight
            total_weight += weight
            weighted_progress += goal.progress * weight

        self.overall_progress = weighted_progress / max(total_weight, 1.0)
        return self.overall_progress


class GoalDecompositionError(Exception):
    """Custom exception for goal decomposition operations."""

    def __init__(
        self,
        message: str,
        goal_id: Optional[str] = None,
        hierarchy_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.goal_id = goal_id
        self.hierarchy_id = hierarchy_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class GoalTemplate:
    """Template for common goal patterns."""

    def __init__(self, template_id: str, name: str, description: str):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.goal_patterns: List[Dict[str, Any]] = []
        self.success_criteria_templates: List[str] = []
        self.decomposition_rules: List[Dict[str, Any]] = []
        self.context_requirements: Dict[str, Any] = {}
        self.resource_estimates: Dict[str, float] = {}

    def generate_goals(self, context: Dict[str, Any]) -> List[Goal]:
        """Generate goals from this template."""
        goals = []

        for pattern in self.goal_patterns:
            goal = Goal(
                goal_id=f"{self.template_id}_{len(goals)}",
                description=pattern.get("description", "").format(**context),
                goal_type=GoalType(pattern.get("type", "primary")),
                priority=GoalPriority(pattern.get("priority", 2)),
                success_criteria=[sc.format(**context) for sc in self.success_criteria_templates],
                context=context.copy(),
                resource_requirements=self.resource_estimates.copy(),
            )
            goals.append(goal)

        return goals


class DecompositionHeuristic(ABC):
    """Abstract base class for goal decomposition heuristics."""

    @abstractmethod
    async def should_decompose(self, goal: Goal, context: Dict[str, Any]) -> bool:
        """Determine if a goal should be decomposed."""
        pass

    @abstractmethod
    async def decompose_goal(self, goal: Goal, context: Dict[str, Any]) -> List[Goal]:
        """Decompose a goal into sub-goals."""
        pass

    @abstractmethod
    def estimate_decomposition_quality(
        self, original_goal: Goal, sub_goals: List[Goal], context: Dict[str, Any]
    ) -> float:
        """Estimate the quality of a decomposition."""
        pass


class ComplexityBasedHeuristic(DecompositionHeuristic):
    """Decomposition heuristic based on goal complexity."""

    def __init__(self, complexity_threshold: float = 7.0):
        self.complexity_threshold = complexity_threshold
        self.logger = get_logger(__name__)

    async def should_decompose(self, goal: Goal, context: Dict[str, Any]) -> bool:
        """Decompose if goal complexity exceeds threshold."""
        return goal.metrics.complexity_score > self.complexity_threshold

    async def decompose_goal(self, goal: Goal, context: Dict[str, Any]) -> List[Goal]:
        """Decompose based on complexity factors."""
        sub_goals = []

        # Decompose by major components
        if goal.metrics.complexity_score > 10:
            # High complexity: break into phases
            phases = ["preparation", "execution", "validation"]
            for i, phase in enumerate(phases):
                sub_goal = Goal(
                    goal_id=f"{goal.goal_id}_phase_{i}",
                    description=f"{goal.description} - {phase.title()} Phase",
                    goal_type=GoalType.INSTRUMENTAL,
                    priority=goal.priority,
                    parent_goal=goal.goal_id,
                    level=goal.level + 1,
                    estimated_completion_time=goal.estimated_completion_time / len(phases),
                    context=goal.context.copy(),
                )

                # Set dependencies between phases
                if i > 0:
                    sub_goal.dependencies.add(f"{goal.goal_id}_phase_{i-1}")

                sub_goals.append(sub_goal)
        else:
            # Medium complexity: break by function
            functions = self._identify_functions(goal, context)
            for i, function in enumerate(functions):
                sub_goal = Goal(
                    goal_id=f"{goal.goal_id}_func_{i}",
                    description=f"{goal.description} - {function}",
                    goal_type=GoalType.INSTRUMENTAL,
                    priority=goal.priority,
                    parent_goal=goal.goal_id,
                    level=goal.level + 1,
                    estimated_completion_time=goal.estimated_completion_time / len(functions),
                    context=goal.context.copy(),
                )
                sub_goals.append(sub_goal)

        return sub_goals

    def _identify_functions(self, goal: Goal, context: Dict[str, Any]) -> List[str]:
        """Identify functional components of a goal."""
        # This would use NLP and domain knowledge to identify functions
        # For now, return generic functions
        return ["analyze", "process", "generate", "validate"]

    def estimate_decomposition_quality(
        self, original_goal: Goal, sub_goals: List[Goal], context: Dict[str, Any]
    ) -> float:
        """Estimate quality based on coverage and coherence."""
        if not sub_goals:
            return 0.0

        # Coverage: do sub-goals cover the original goal?
        total_sub_complexity = sum(sg.metrics.complexity_score for sg in sub_goals)
        coverage = min(1.0, total_sub_complexity / original_goal.metrics.complexity_score)

        # Coherence: are sub-goals logically related?
        coherence = 1.0  # Simplified - would analyze semantic similarity

        # Efficiency: is decomposition efficient?
        total_sub_time = sum(sg.estimated_completion_time for sg in sub_goals)
        efficiency = original_goal.estimated_completion_time / max(total_sub_time, 1.0)
        efficiency = min(1.0, efficiency)

        return (coverage + coherence + efficiency) / 3.0


class KnowledgeBasedHeuristic(DecompositionHeuristic):
    """Decomposition heuristic using knowledge graph patterns."""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.logger = get_logger(__name__)

    async def should_decompose(self, goal: Goal, context: Dict[str, Any]) -> bool:
        """Decompose if knowledge graph has decomposition patterns."""
        try:
            patterns = await self.knowledge_graph.query_goal_patterns(
                goal.description, goal.goal_type.value, goal.domain
            )
            return len(patterns) > 0 and any(p.get("decomposable", False) for p in patterns)
        except Exception:
            return False

    async def decompose_goal(self, goal: Goal, context: Dict[str, Any]) -> List[Goal]:
        """Decompose using knowledge graph patterns."""
        sub_goals = []

        try:
            patterns = await self.knowledge_graph.query_goal_patterns(
                goal.description, goal.goal_type.value, goal.domain
            )

            if patterns:
                best_pattern = patterns[0]  # Use best matching pattern
                sub_patterns = best_pattern.get("sub_goals", [])

                for i, sub_pattern in enumerate(sub_patterns):
                    sub_goal = Goal(
                        goal_id=f"{goal.goal_id}_kb_{i}",
                        description=sub_pattern.get("description", f"Sub-goal {i}"),
                        goal_type=GoalType(sub_pattern.get("type", "instrumental")),
                        priority=goal.priority,
                        parent_goal=goal.goal_id,
                        level=goal.level + 1,
                        success_criteria=sub_pattern.get("success_criteria", []),
                        estimated_completion_time=sub_pattern.get(
                            "estimated_time", goal.estimated_completion_time / len(sub_patterns)
                        ),
                        context=goal.context.copy(),
                    )

                    # Set dependencies from pattern
                    for dep in sub_pattern.get("dependencies", []):
                        if dep < len(sub_patterns):
                            sub_goal.dependencies.add(f"{goal.goal_id}_kb_{dep}")

                    sub_goals.append(sub_goal)

        except Exception as e:
            self.logger.warning(f"Knowledge-based decomposition failed: {str(e)}")

        return sub_goals

    def estimate_decomposition_quality(
        self, original_goal: Goal, sub_goals: List[Goal], context: Dict[str, Any]
    ) -> float:
        """Estimate quality based on knowledge graph confidence."""
        # Would query knowledge graph for quality metrics
        return 0.8  # Simplified


class ConflictResolver:
    """Resolves conflicts between goals."""

    def __init__(self, preference_learner: PreferenceLearner):
        self.preference_learner = preference_learner
        self.logger = get_logger(__name__)

    async def detect_conflicts(
        self, hierarchy: GoalHierarchy
    ) -> List[Tuple[str, str, ConflictType]]:
        """Detect conflicts between goals in hierarchy."""
        conflicts = []
        goals = list(hierarchy.goals.values())

        # Check pairwise conflicts
        for i, goal1 in enumerate(goals):
            for goal2 in goals[i + 1 :]:
                conflict_type = await self._check_goal_conflict(goal1, goal2, hierarchy)
                if conflict_type:
                    conflicts.append((goal1.goal_id, goal2.goal_id, conflict_type))

        hierarchy.conflicts = conflicts
        return conflicts

    async def _check_goal_conflict(
        self, goal1: Goal, goal2: Goal, hierarchy: GoalHierarchy
    ) -> Optional[ConflictType]:
        """Check if two goals conflict."""
        # Resource conflict
        for resource, amount1 in goal1.resource_requirements.items():
            amount2 = goal2.resource_requirements.get(resource, 0)
            if amount1 > 0 and amount2 > 0:
                # Both need same resource - potential conflict
                total_available = 100.0  # Simplified
                if amount1 + amount2 > total_available:
                    return ConflictType.RESOURCE_CONFLICT

        # Temporal conflict
        if goal1.deadline and goal2.deadline:
            time_overlap = self._check_time_overlap(goal1, goal2)
            if time_overlap and self._compete_for_resources(goal1, goal2):
                return ConflictType.TEMPORAL_CONFLICT

        # Logical conflict
        if self._are_logically_contradictory(goal1, goal2):
            return ConflictType.LOGICAL_CONFLICT

        return None

    def _check_time_overlap(self, goal1: Goal, goal2: Goal) -> bool:
        """Check if goals have overlapping time requirements."""
        # Simplified time overlap check
        return True  # Would implement actual temporal logic

    def _compete_for_resources(self, goal1: Goal, goal2: Goal) -> bool:
        """Check if goals compete for same resources."""
        common_resources = set(goal1.resource_requirements.keys()) & set(
            goal2.resource_requirements.keys()
        )
        return len(common_resources) > 0

    def _are_logically_contradictory(self, goal1: Goal, goal2: Goal) -> bool:
        """Check if goals are logically contradictory."""
        # Would implement semantic analysis
        return False

    async def resolve_conflicts(
        self, hierarchy: GoalHierarchy, user_id: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve conflicts in goal hierarchy."""
        resolutions = {}

        for goal1_id, goal2_id, conflict_type in hierarchy.conflicts:
            resolution = await self._resolve_single_conflict(
                hierarchy.goals[goal1_id],
                hierarchy.goals[goal2_id],
                conflict_type,
                hierarchy,
                user_id,
            )

            conflict_key = f"{goal1_id}_{goal2_id}_{conflict_type.value}"
            resolutions[conflict_key] = resolution

        hierarchy.conflict_resolutions = resolutions
        return resolutions

    async def _resolve_single_conflict(
        self,
        goal1: Goal,
        goal2: Goal,
        conflict_type: ConflictType,
        hierarchy: GoalHierarchy,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Resolve a single conflict between two goals."""
        resolution = {
            "strategy": "unknown",
            "actions": [],
            "priority_adjustment": False,
            "schedule_adjustment": False,
            "resource_reallocation": False,
        }

        # Get user preferences if available
        user_prefs = {}
        if user_id and self.preference_learner:
            try:
                user_prefs = await self.preference_learner.get_user_preferences(user_id)
            except Exception:
                pass

        if conflict_type == ConflictType.RESOURCE_CONFLICT:
            resolution = await self._resolve_resource_conflict(goal1, goal2, user_prefs)
        elif conflict_type == ConflictType.TEMPORAL_CONFLICT:
            resolution = await self._resolve_temporal_conflict(goal1, goal2, user_prefs)
        elif conflict_type == ConflictType.LOGICAL_CONFLICT:
            resolution = await self._resolve_logical_conflict(goal1, goal2, user_prefs)

        return resolution

    async def _resolve_resource_conflict(
        self, goal1: Goal, goal2: Goal, user_prefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve resource conflict between goals."""
        # Priority-based resolution
        if goal1.priority.value > goal2.priority.value:
            return {
                "strategy": "priority_allocation",
                "actions": [f"Allocate resources to {goal1.goal_id} first"],
                "resource_reallocation": True,
            }
        elif goal2.priority.value > goal1.priority.value:
            return {
                "strategy": "priority_allocation",
                "actions": [f"Allocate resources to {goal2.goal_id} first"],
                "resource_reallocation": True,
            }
        else:
            # Equal priority - use time-sharing
            return {
                "strategy": "time_sharing",
                "actions": ["Split resource usage by time slots"],
                "schedule_adjustment": True,
            }

    async def _resolve_temporal_conflict(
        self, goal1: Goal, goal2: Goal, user_prefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve temporal conflict between goals."""
        return {
            "strategy": "sequential_execution",
            "actions": ["Execute goals sequentially based on deadline"],
            "schedule_adjustment": True,
        }

    async def _resolve_logical_conflict(
        self, goal1: Goal, goal2: Goal, user_prefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve logical conflict between goals."""
        return {
            "strategy": "goal_refinement",
            "actions": ["Refine goals to eliminate contradiction"],
            "priority_adjustment": True,
        }


class GoalDecomposer:
    """
    Advanced Goal Decomposition Engine for the AI Assistant.

    This decomposer provides intelligent goal hierarchy management:
    - Automatic goal decomposition using multiple strategies
    - Hierarchical goal structure management
    - Goal conflict detection and resolution
    - Context-aware goal adaptation
    - Integration with task planning and execution
    - Learning from goal achievement patterns
    - User preference-aware goal prioritization
    - Multi-level goal optimization
    - Goal template management and reuse
    """

    def __init__(self, container: Container):
        """
        Initialize the goal decomposer.

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
        self.sentiment_analyzer = container.get(SentimentAnalyzer)

        # Reasoning components
        self.logic_engine = container.get(LogicEngine)
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.decision_tree = container.get(DecisionTree)
        self.task_planner = container.get(TaskPlanner)

        # Assistant components
        self.core_engine = container.get(CoreAssistantEngine)
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(SessionManager)
        self.interaction_handler = container.get(InteractionHandler)

        # Skills management
        self.skill_registry = container.get(SkillRegistry)
        self.skill_validator = container.get(SkillValidator)
        self.skill_factory = container.get(SkillFactory)

        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.semantic_memory = container.get(SemanticMemory)
        self.episodic_memory = container.get(EpisodicMemory)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Decomposition components
        self.decomposition_heuristics: List[DecompositionHeuristic] = []
        self.conflict_resolver = ConflictResolver(self.preference_learner)
        self.goal_templates: Dict[str, GoalTemplate] = {}

        # State management
        self.active_hierarchies: Dict[str, GoalHierarchy] = {}
        self.hierarchy_history: deque = deque(maxlen=1000)
        self.decomposition_cache: Dict[str, List[Goal]] = {}

        # Configuration
        self.max_decomposition_depth = self.config.get("goals.max_depth", 5)
        self.min_goal_complexity = self.config.get("goals.min_complexity", 3.0)
        self.enable_caching = self.config.get("goals.enable_caching", True)
        self.enable_learning = self.config.get("goals.enable_learning", True)
        self.conflict_resolution_enabled = self.config.get("goals.resolve_conflicts", True)

        # Performance tracking
        self.decomposition_stats: Dict[str, Any] = defaultdict(int)
        self.goal_success_rates: Dict[str, List[float]] = defaultdict(list)

        # Initialize components
        self._setup_heuristics()
        self._load_goal_templates()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("goal_decomposer", self._health_check_callback)

        self.logger.info("GoalDecomposer initialized successfully")

    def _setup_heuristics(self) -> None:
        """Setup decomposition heuristics."""
        try:
            # Complexity-based heuristic
            complexity_threshold = self.config.get("goals.complexity_threshold", 7.0)
            self.decomposition_heuristics.append(ComplexityBasedHeuristic(complexity_threshold))

            # Knowledge-based heuristic
            self.decomposition_heuristics.append(KnowledgeBasedHeuristic(self.knowledge_graph))

            self.logger.info(
                f"Initialized {len(self.decomposition_heuristics)} decomposition heuristics"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup heuristics: {str(e)}")

    def _load_goal_templates(self) -> None:
        """Load predefined goal templates."""
        try:
            # Question answering template
            qa_template = GoalTemplate(
                "question_answering", "Question Answering", "Answer user questions effectively"
            )
            qa_template.goal_patterns = [
                {
                    "description": "Understand the question: {question}",
                    "type": "analytical",
                    "priority": 3,
                },
                {
                    "description": "Retrieve relevant information",
                    "type": "instrumental",
                    "priority": 3,
                },
                {"description": "Generate comprehensive answer", "type": "primary", "priority": 4},
            ]
            qa_template.success_criteria_templates = [
                "User question is fully understood",
                "Relevant information is retrieved",
                "Answer is accurate and helpful",
            ]
            self.goal_templates["question_answering"] = qa_template

            # Task completion template
            task_template = GoalTemplate(
                "task_completion", "Task Completion", "Complete user-requested tasks"
            )
            task_template.goal_patterns = [
                {
                    "description": "Analyze task requirements: {task}",
                    "type": "analytical",
                    "priority": 3,
                },
                {"description": "Plan task execution", "type": "instrumental", "priority": 3},
                {"description": "Execute task steps", "type": "primary", "priority": 4},
                {"description": "Validate task completion", "type": "maintenance", "priority": 2},
            ]
            self.goal_templates["task_completion"] = task_template

            # Creative assistance template
            creative_template = GoalTemplate(
                "creative_assistance", "Creative Assistance", "Assist with creative tasks"
            )
            creative_template.goal_patterns = [
                {
                    "description": "Understand creative intent: {intent}",
                    "type": "creative",
                    "priority": 3,
                },
                {"description": "Generate creative ideas", "type": "creative", "priority": 4},
                {"description": "Refine and present results", "type": "creative", "priority": 3},
            ]
            self.goal_templates["creative_assistance"] = creative_template

            self.logger.info(f"Loaded {len(self.goal_templates)} goal templates")

        except Exception as e:
            self.logger.warning(f"Failed to load goal templates: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register goal metrics
            self.metrics.register_counter("goals_created_total")
            self.metrics.register_counter("goals_decomposed_total")
            self.metrics.register_counter("goals_achieved_total")
            self.metrics.register_counter("goals_failed_total")
            self.metrics.register_histogram("goal_decomposition_duration_seconds")
            self.metrics.register_histogram("goal_achievement_time_seconds")
            self.metrics.register_gauge("active_goal_hierarchies")
            self.metrics.register_counter("goal_conflicts_detected")
            self.metrics.register_counter("goal_conflicts_resolved")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the goal decomposer."""
        try:
            # Initialize knowledge base with goal patterns
            await self._load_goal_knowledge()

            # Start background tasks
            asyncio.create_task(self._goal_monitoring_loop())
            asyncio.create_task(self._hierarchy_optimization_loop())

            if self.enable_learning:
                asyncio.create_task(self._learning_update_loop())

            self.logger.info("GoalDecomposer initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize GoalDecomposer: {str(e)}")
            raise GoalDecompositionError(f"Initialization failed: {str(e)}")

    async def _load_goal_knowledge(self) -> None:
        """Load goal decomposition knowledge into the knowledge graph."""
        try:
            # Load common goal decomposition patterns
            patterns = {
                "question_answering_decomposition": {
                    "goal_type": "analytical",
                    "sub_goals": [
                        {"description": "Parse question", "type": "analytical"},
                        {"description": "Identify information needs", "type": "analytical"},
                        {"description": "Retrieve information", "type": "instrumental"},
                        {"description": "Synthesize answer", "type": "primary"},
                    ],
                    "dependencies": [(0, 1), (1, 2), (2, 3)],
                },
                "task_execution_decomposition": {
                    "goal_type": "primary",
                    "sub_goals": [
                        {"description": "Understand task", "type": "analytical"},
                        {"description": "Plan execution", "type": "instrumental"},
                        {"description": "Execute steps", "type": "primary"},
                        {"description": "Verify completion", "type": "maintenance"},
                    ],
                    "dependencies": [(0, 1), (1, 2), (2, 3)],
                },
                "creative_process_decomposition": {
                    "goal_type": "creative",
                    "sub_goals": [
                        {"description": "Brainstorm ideas", "type": "creative"},
                        {"description": "Select best concepts", "type": "analytical"},
                        {"description": "Develop chosen concept", "type": "creative"},
                        {"description": "Refine output", "type": "optimization"},
                    ],
                    "dependencies": [(0, 1), (1, 2), (2, 3)],
                },
            }

            # Store patterns in knowledge graph
            for pattern_name, pattern_data in patterns.items():
                await self.knowledge_graph.store_goal_pattern(pattern_name, pattern_data)

        except Exception as e:
            self.logger.warning(f"Failed to load goal knowledge: {str(e)}")

    @handle_exceptions
    async def create_goal_hierarchy_from_intent(
        self, intent: str, context: Dict[str, Any], session_id: str, user_id: Optional[str] = None
    ) -> GoalHierarchy:
        """
        Create a goal hierarchy from user intent.

        Args:
            intent: User intent
            context: Context information
            session_id: Session identifier
            user_id: Optional user identifier

        Returns:
            Goal hierarchy
        """
        start_time = time.time()

        try:
            with self.tracer.trace("goal_hierarchy_creation") as span:
                span.set_attributes(
                    {"intent": intent, "session_id": session_id, "user_id": user_id or "anonymous"}
                )

                # Emit decomposition started event
                await self.event_bus.emit(
                    GoalDecompositionStarted(session_id=session_id, intent=intent, user_id=user_id)
                )

                # Create hierarchy
                hierarchy = GoalHierarchy(
                    hierarchy_id=str(uuid.uuid4()), session_id=session_id, user_id=user_id
                )

                # Create root goal from intent
                root_goal = await self._create_root_goal_from_intent(intent, context)
                hierarchy.add_goal(root_goal)

                # Decompose the root goal
                await self._decompose_goal_hierarchy(hierarchy, root_goal.goal_id, context)

                # Detect and resolve conflicts
                if self.conflict_resolution_enabled:
                    conflicts = await self.conflict_resolver.detect_conflicts(hierarchy)
                    if conflicts:
                        await self.conflict_resolver.resolve_conflicts(hierarchy, user_id)

                        await self.event_bus.emit(
                            GoalConflictDetected(
                                hierarchy_id=hierarchy.hierarchy_id,
                                session_id=session_id,
                                conflict_count=len(conflicts),
                            )
                        )

                # Calculate hierarchy metrics
                hierarchy.calculate_progress()

                # Store hierarchy
                self.active_hierarchies[hierarchy.hierarchy_id] = hierarchy

                decomposition_time = time.time() - start_time

                # Update metrics
                self.metrics.increment("goals_created_total")
                self.metrics.record("goal_decomposition_duration_seconds", decomposition_time)
                self.metrics.set("active_goal_hierarchies", len(self.active_hierarchies))

                # Emit completion event
                await self.event_bus.emit(
                    GoalDecompositionCompleted(
                        hierarchy_id=hierarchy.hierarchy_id,
                        session_id=session_id,
                        goal_count=len(hierarchy.goals),
                        decomposition_time=decomposition_time,
                    )
                )

                self.logger.info(
                    f"Created goal hierarchy {hierarchy.hierarchy_id} for intent '{intent}' "
                    f"with {len(hierarchy.goals)} goals in {decomposition_time:.2f}s"
                )

                return hierarchy

        except Exception as e:
            decomposition_time = time.time() - start_time

            await self.event_bus.emit(
                GoalDecompositionFailed(
                    session_id=session_id,
                    intent=intent,
                    error_message=str(e),
                    decomposition_time=decomposition_time,
                )
            )

            self.logger.error(f"Failed to create goal hierarchy for intent '{intent}': {str(e)}")
            raise GoalDecompositionError(f"Goal hierarchy creation failed: {str(e)}")

    async def _create_root_goal_from_intent(self, intent: str, context: Dict[str, Any]) -> Goal:
        """Create a root goal from user intent."""
        # Determine goal type based on intent
        goal_type = await self._classify_goal_type(intent, context)

        # Extract entities for success criteria
        entities = context.get("entities", [])
        success_criteria = [f"Successfully address intent: {intent}"]

        for entity in entities:
            if entity.get("type") in ["objective", "goal", "target"]:
                success_criteria.append(f"Achieve: {entity.get('value')}")

        # Estimate complexity and resources
        complexity = await self._estimate_goal_complexity(intent, context)
        resources = await self._estimate_resource_requirements(intent, context)

        # Create root goal
        root_goal = Goal(
            goal_id=f"root_{int(time.time())}",
            description=f"Primary goal: {intent}",
            goal_type=goal_type,
            priority=GoalPriority.HIGH,
            success_criteria=success_criteria,
            context=context.copy(),
            user_intent=intent,
            domain=context.get("domain", "general"),
            estimated_completion_time=complexity * 30.0,  # Rough estimate
            resource_requirements=resources,
            level=0,
        )

        # Set metrics
        root_goal.metrics.complexity_score = complexity
        root_goal.metrics.user_value = 10.0  # High value for primary goal
        root_goal.metrics.confidence_score = 0.8
        root_goal.metrics.success_probability = 0.7

        return root_goal

    async def _classify_goal_type(self, intent: str, context: Dict[str, Any]) -> GoalType:
        """Classify the type of goal based on intent."""
        try:
            # Use intent analysis to classify
            intent_analysis = await self.intent_manager.analyze_intent(intent)

            if "question" in intent.lower() or "what" in intent.lower() or "how" in intent.lower():
                return GoalType.ANALYTICAL
            elif (
                "create" in intent.lower()
                or "generate" in intent.lower()
                or "design" in intent.lower()
            ):
                return GoalType.CREATIVE
            elif "optimize" in intent.lower() or "improve" in intent.lower():
                return GoalType.OPTIMIZATION
            elif "explore" in intent.lower() or "learn" in intent.lower():
                return GoalType.EXPLORATION
            else:
                return GoalType.PRIMARY

        except Exception:
            return GoalType.PRIMARY

    async def _estimate_goal_complexity(self, intent: str, context: Dict[str, Any]) -> float:
        """Estimate goal complexity based on intent and context."""
        base_complexity = 5.0

        # Length factor
        word_count = len(intent.split())
        complexity = base_complexity + word_count * 0.2

        # Context factors
        if context.get("multimodal", False):
            complexity += 2.0

        if context.get("real_time_required", False):
            complexity += 1.5

        if context.get("external_apis_needed", False):
            complexity += 1.0

        # Domain-specific complexity
        domain = context.get("domain", "general")
        domain_complexity = {
            "scientific": 8.0,
            "medical": 9.0,
            "legal": 7.0,
            "technical": 6.0,
            "creative": 5.0,
            "general": 4.0,
        }

        complexity += domain_complexity.get(domain, 4.0)

        return min(complexity, 20.0)  # Cap at 20

    async def _estimate_resource_requirements(
        self, intent: str, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Estimate resource requirements for goal achievement."""
        resources = {"cpu": 10.0, "memory": 100.0, "network": 5.0, "time": 60.0}

        # Adjust based on intent type
        if "analyze" in intent.lower() or "compute" in intent.lower():
            resources["cpu"] *= 2.0
            resources["memory"] *= 1.5

        if "search" in intent.lower() or "find" in intent.lower():
            resources["network"] *= 3.0

        if context.get("large_dataset", False):
            resources["memory"] *= 3.0
            resources["time"] *= 2.0

        return resources

    async def _decompose_goal_hierarchy(
        self, hierarchy: GoalHierarchy, goal_id: str, context: Dict[str, Any], depth: int = 0
    ) -> None:
        """Recursively decompose goals in hierarchy."""
        if depth >= self.max_decomposition_depth:
            return

        goal = hierarchy.goals[goal_id]

        # Check if goal should be decomposed
        should_decompose = False
        for heuristic in self.decomposition_heuristics:
            if await heuristic.should_decompose(goal, context):
                should_decompose = True
                break

        if not should_decompose:
            return

        # Decompose using best heuristic
        best_sub_goals = []
        best_quality = 0.0

        for heuristic in self.decomposition_heuristics:
            try:
                sub_goals = await heuristic.decompose_goal(goal, context)
                if sub_goals:
                    quality = heuristic.estimate_decomposition_quality(goal, sub_goals, context)
                    if quality > best_quality:
                        best_quality = quality
                        best_sub_goals = sub_goals
            except Exception as e:
                self.logger.warning(f"Heuristic decomposition failed: {str(e)}")

        # Add sub-goals to hierarchy
        for sub_goal in best_sub_goals:
            hierarchy.add_goal(sub_goal)
            goal.sub_goals.append(sub_goal.goal_id)

            # Emit sub-goal generated event
            await self.event_bus.emit(
                SubGoalGenerated(
                    hierarchy_id=hierarchy.hierarchy_id,
                    parent_goal_id=goal_id,
                    sub_goal_id=sub_goal.goal_id,
                    decomposition_quality=best_quality,
                )
            )

            # Recursively decompose sub-goals
            await self._decompose_goal_hierarchy(hierarchy, sub_goal.goal_id, context, depth + 1)

        # Update metrics
        self.metrics.increment("goals_decomposed_total")

    @handle_exceptions
    async def plan_goal_execution(
        self, hierarchy_id: str, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for a goal hierarchy.

        Args:
            hierarchy_id: Goal hierarchy identifier
            context: Optional execution context

        Returns:
            Execution plan
        """
        if hierarchy_id not in self.active_hierarchies:
            raise GoalDecompositionError(f"Hierarchy {hierarchy_id} not found")

        hierarchy = self.active_hierarchies[hierarchy_id]
        context = context or {}

        try:
            # Get ready goals (leaf goals with satisfied dependencies)
            ready_goals = hierarchy.get_ready_goals()
            leaf_goals = hierarchy.get_leaf_goals()

            # Convert goals to tasks
            tasks = []
            for goal in leaf_goals:
                task = await self._convert_goal_to_task(goal, context)
                tasks.append(task)

            # Create execution plan using task planner
            plan = await self.task_planner.create_plan_for_goals(
                tasks, hierarchy.session_id, hierarchy.user_id
            )

            # Link plan to hierarchy
            plan.context["goal_hierarchy_id"] = hierarchy_id

            return plan

        except Exception as e:
            self.logger.error(
                f"Failed to plan goal execution for hierarchy {hierarchy_id}: {str(e)}"
            )
            raise GoalDecompositionError(f"Goal execution planning failed: {str(e)}")

    async def _convert_goal_to_task(self, goal: Goal, context: Dict[str, Any]) -> Task:
        """Convert a goal to a task for execution planning."""
        # Determine task type and execution target
        task_type = "generic"
        execution_target = None
        execution_method = None

        if goal.goal_type == GoalType.ANALYTICAL:
            # Look for analysis skills
            analysis_skills = await self.skill_registry.find_skills_by_category("analysis")
            if analysis_skills:
                task_type = "skill"
                execution_target = analysis_skills[0].get("name")
        elif goal.goal_type == GoalType.CREATIVE:
            # Look for creative skills
            creative_skills = await self.skill_registry.find_skills_by_category("creative")
            if creative_skills:
                task_type = "skill"
                execution_target = creative_skills[0].get("name")

        # Create task
        task = Task(
            task_id=f"task_{goal.goal_id}",
            name=f"Execute: {goal.description}",
            description=goal.description,
            task_type=task_type,
            execution_target=execution_target,
            execution_method=execution_method,
            priority=TaskPriority(goal.priority.value),
            estimated_duration=goal.estimated_completion_time,
            required_resources=goal.resource_requirements,
            success_probability=goal.metrics.success_probability,
            parameters={"goal_id": goal.goal_id, "goal_context": goal.context},
        )

        # Set dependencies based on goal dependencies
        for dep_goal_id in goal.dependencies:
            task.dependencies.add(f"task_{dep_goal_id}")

        return task

    @handle_exceptions
    async def update_goal_progress(
        self,
        hierarchy_id: str,
        goal_id: str,
        progress: float,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update progress for a specific goal.

        Args:
            hierarchy_id: Goal hierarchy identifier
            goal_id: Goal identifier
            progress: Progress value (0.0 to 1.0)
            evidence: Optional evidence of progress
        """
        if hierarchy_id not in self.active_hierarchies:
            raise GoalDecompositionError(f"Hierarchy {hierarchy_id} not found")

        hierarchy = self.active_hierarchies[hierarchy_id]

        if goal_id not in hierarchy.goals:
            raise GoalDecompositionError(f"Goal {goal_id} not found in hierarchy")

        goal = hierarchy.goals[goal_id]
        old_progress = goal.progress
        goal.progress = max(0.0, min(1.0, progress))

        # Add evidence if provided
        if evidence:
            goal.achievement_evidence.append(
                {
                    "timestamp": datetime.now(timezone.utc),
                    "progress": progress,
                    "evidence": evidence,
                }
            )

        # Update goal status based on progress
        if goal.progress >= 1.0:
            goal.status = GoalStatus.ACHIEVED
            goal.actual_completion_time = datetime.now(timezone.utc)
            hierarchy.achieved_goals.add(goal_id)

            # Emit goal achieved event
            await self.event_bus.emit(
                GoalAchieved(
                    hierarchy_id=hierarchy_id,
                    goal_id=goal_id,
                    achievement_time=(
                        goal.actual_completion_time - goal.created_at
                    ).total_seconds(),
                    evidence_count=len(goal.achievement_evidence),
                )
            )

            self.metrics.increment("goals_achieved_total")

        elif goal.progress > 0.0:
            goal.status = GoalStatus.PARTIALLY_ACHIEVED

        # Propagate progress to parent goals
        await self._propagate_progress_update(hierarchy, goal_id)

        # Recalculate hierarchy progress
        hierarchy.calculate_progress()
        hierarchy.last_updated = datetime.now(timezone.utc)

        self.logger.info(
            f"Updated goal {goal_id} progress from {old_progress:.2f} to {goal.progress:.2f}"
        )

    async def _propagate_progress_update(self, hierarchy: GoalHierarchy, goal_id: str) -> None:
        """Propagate progress updates to parent goals."""
        goal = hierarchy.goals[goal_id]

        if goal.parent_goal:
            parent = hierarchy.goals[goal.parent_goal]

            # Calculate parent progress based on sub-goal progress
            if parent.sub_goals:
                total_progress = 0.0
                for sub_goal_id in parent.sub_goals:
                    if sub_goal_id in hierarchy.goals:
                        total_progress += hierarchy.goals[sub_goal_id].progress

                parent.progress = total_progress / len(parent.sub_goals)

                # Recursively update grandparent
                await self._propagate_progress_update(hierarchy, parent.goal_id)

    @handle_exceptions
    async def adapt_goal_hierarchy(
        self,
        hierarchy_id: str,
        context_changes: Dict[str, Any],
        feedback: Optional[Dict[str, Any]] = None,
    ) -> GoalHierarchy:
        """
        Adapt goal hierarchy based on context changes and feedback.

        Args:
            hierarchy_id: Goal hierarchy identifier
            context_changes: Changes in context
            feedback: Optional user feedback

        Returns:
            Updated goal hierarchy
        """
        if hierarchy_id not in self.active_hierarchies:
            raise GoalDecompositionError(f"Hierarchy {hierarchy_id} not found")

        hierarchy = self.active_hierarchies[hierarchy_id]

        try:
            # Update context for all goals in hierarchy
            for goal in hierarchy.goals:
                goal.context.update(update_context)

        except Exception as e:
            self.logger.error(f"Error updating goal hierarchy context: {str(e)}")
            raise
