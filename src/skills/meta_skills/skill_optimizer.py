"""
Skill Optimizer Module
Author: Drmusab
Last Modified: 2025-07-05 22:18:05 UTC

This module provides functionality for optimizing skills through performance analysis,
parameter tuning, execution strategy optimization, and caching. It aims to improve
the efficiency, reliability, and effectiveness of skills in the AI assistant system.
"""

import ast
import copy
import importlib
import inspect
import json
import logging
import os
import random
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import asyncio
import numpy as np
import pandas as pd

# Assistant components
from src.assistant.component_manager import ComponentManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SkillExecutionCompleted,
    SkillExecutionFailed,
    SkillExecutionStarted,
    SkillOptimizationCompleted,
    SkillOptimizationFailed,
    SkillOptimizationStarted,
    SystemConfigurationChanged,
)

# Integrations
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.llm.model_router import ModelRouter

# Learning components
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.model_adaptation import ModelAdapter

# Memory and learning components
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory
from src.memory.operations.context_manager import ContextManager

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.profiling.cpu_profiler import CPUProfiler
from src.observability.profiling.memory_profiler import MemoryProfiler
from src.skills.skill_factory import SkillFactory

# Skill management
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator


class OptimizationType(Enum):
    """Types of optimization that can be performed on skills."""

    PERFORMANCE = "performance"  # Optimize for execution speed
    MEMORY = "memory"  # Optimize memory usage
    QUALITY = "quality"  # Optimize result quality
    CACHING = "caching"  # Optimize through result caching
    PARAMETERS = "parameters"  # Optimize skill parameters
    CODE = "code"  # Optimize skill code
    SELECTION = "selection"  # Optimize skill selection
    COMPREHENSIVE = "comprehensive"  # All optimizations combined


class OptimizationLevel(Enum):
    """Levels of optimization aggressiveness."""

    MINIMAL = "minimal"  # Safe, minimal changes
    MODERATE = "moderate"  # Balanced approach
    AGGRESSIVE = "aggressive"  # Maximum optimization, potentially breaking changes


class OptimizationMode(Enum):
    """Modes for applying optimizations."""

    ANALYZE = "analyze"  # Analyze only, no changes
    SUGGEST = "suggest"  # Suggest changes but don't apply
    APPLY = "apply"  # Apply changes automatically
    ADAPTIVE = "adaptive"  # Apply and continuously adapt


@dataclass
class SkillPerformanceMetrics:
    """Performance metrics for a skill."""

    skill_id: str
    execution_count: int = 0
    avg_execution_time: float = 0.0
    min_execution_time: float = float("inf")
    max_execution_time: float = 0.0
    success_rate: float = 1.0
    memory_usage: float = 0.0  # Average memory usage in MB
    error_count: int = 0
    last_executed: Optional[datetime] = None
    parameter_distributions: Dict[str, Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    execution_times: List[float] = field(default_factory=list)
    common_errors: Counter = field(default_factory=Counter)
    cached_execution_count: int = 0
    cache_hit_rate: float = 0.0
    avg_cached_execution_time: float = 0.0


@dataclass
class OptimizationSuggestion:
    """A suggestion for optimizing a skill."""

    skill_id: str
    optimization_type: OptimizationType
    description: str
    impact: str  # High, Medium, Low
    suggested_changes: Dict[str, Any] = field(default_factory=dict)
    code_changes: Optional[Dict[str, str]] = None  # Original -> New
    estimated_improvement: str = ""
    risk_level: str = "low"  # low, medium, high
    rationale: str = ""
    applied: bool = False


@dataclass
class OptimizationResult:
    """Results from an optimization operation."""

    skill_id: str
    optimization_id: str
    optimization_type: OptimizationType
    metrics_before: SkillPerformanceMetrics
    metrics_after: Optional[SkillPerformanceMetrics] = None
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    applied_changes: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SkillOptimizer:
    """
    Handles the optimization of skills for improved performance and effectiveness.

    This class provides functionality to:
    - Analyze skill performance and usage patterns
    - Optimize skill parameters for better results
    - Implement caching strategies for frequently used skills
    - Refactor skill code for better performance
    - Recommend skill selection strategies
    - Adapt skills based on usage patterns
    """

    def __init__(self, container: Container):
        """
        Initialize the skill optimizer.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Skill management components
        self.skill_registry = container.get(SkillRegistry)
        self.skill_factory = container.get(SkillFactory)
        self.skill_validator = container.get(SkillValidator)

        # System components
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)

        # Memory and learning components
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.episodic_memory = container.get(EpisodicMemory)

        # Learning components
        try:
            self.continual_learner = container.get(ContinualLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
            self.model_adapter = container.get(ModelAdapter)
        except Exception as e:
            self.logger.warning(f"Learning components not available: {str(e)}")
            self.continual_learner = None
            self.feedback_processor = None
            self.model_adapter = None

        # Integration components
        try:
            self.cache_strategy = container.get(CacheStrategy)
            self.model_router = container.get(ModelRouter)
        except Exception as e:
            self.logger.warning(f"Integration components not available: {str(e)}")
            self.cache_strategy = None
            self.model_router = None

        # Observability components
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Profiling components
        try:
            self.cpu_profiler = container.get(CPUProfiler)
            self.memory_profiler = container.get(MemoryProfiler)
        except Exception as e:
            self.logger.warning(f"Profiling components not available: {str(e)}")
            self.cpu_profiler = None
            self.memory_profiler = None

        # Set up cache for optimization results
        self.optimization_cache = {}

        # Register metrics
        if self.metrics:
            self.metrics.register_counter("skill_optimizations_total")
            self.metrics.register_counter("skill_optimizations_failed")
            self.metrics.register_histogram("skill_optimization_time_seconds")
            self.metrics.register_histogram("skill_performance_improvement_percent")

    async def optimize_skill(
        self,
        skill_id: str,
        optimization_type: OptimizationType = OptimizationType.COMPREHENSIVE,
        level: OptimizationLevel = OptimizationLevel.MODERATE,
        mode: OptimizationMode = OptimizationMode.SUGGEST,
    ) -> OptimizationResult:
        """
        Optimize a skill based on specified parameters.

        Args:
            skill_id: ID of the skill to optimize
            optimization_type: Type of optimization to perform
            level: Level of optimization aggressiveness
            mode: Mode for applying optimizations

        Returns:
            Optimization result
        """
        optimization_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment("skill_optimizations_total")

        # Emit optimization started event
        await self.event_bus.emit(
            SkillOptimizationStarted(
                optimization_id=optimization_id,
                skill_id=skill_id,
                optimization_type=optimization_type.value,
            )
        )

        try:
            with self.tracer.trace(f"skill_optimization_{skill_id}") if self.tracer else None:
                # Check if skill exists
                skill_info = self.skill_registry.get_skill_info(skill_id)
                if not skill_info:
                    raise ValueError(f"Skill '{skill_id}' is not registered")

                # Get current performance metrics
                current_metrics = await self._get_skill_performance_metrics(skill_id)

                # Initialize optimization result
                result = OptimizationResult(
                    skill_id=skill_id,
                    optimization_id=optimization_id,
                    optimization_type=optimization_type,
                    metrics_before=current_metrics,
                )

                # Perform the appropriate optimization
                if optimization_type == OptimizationType.PERFORMANCE:
                    await self._optimize_for_performance(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.MEMORY:
                    await self._optimize_for_memory(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.QUALITY:
                    await self._optimize_for_quality(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.CACHING:
                    await self._optimize_for_caching(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.PARAMETERS:
                    await self._optimize_parameters(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.CODE:
                    await self._optimize_code(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.SELECTION:
                    await self._optimize_selection(skill_id, level, mode, result)
                elif optimization_type == OptimizationType.COMPREHENSIVE:
                    # Perform all optimizations
                    await self._optimize_for_performance(skill_id, level, mode, result)
                    await self._optimize_for_memory(skill_id, level, mode, result)
                    await self._optimize_for_caching(skill_id, level, mode, result)
                    await self._optimize_parameters(skill_id, level, mode, result)
                    await self._optimize_code(skill_id, level, mode, result)
                    # Quality and selection are only analyzed but not applied automatically
                    await self._optimize_for_quality(
                        skill_id, OptimizationMode.SUGGEST, mode, result
                    )
                    await self._optimize_selection(skill_id, OptimizationMode.SUGGEST, mode, result)

                # If changes were applied, get updated metrics
                if (
                    mode in [OptimizationMode.APPLY, OptimizationMode.ADAPTIVE]
                    and result.applied_changes
                ):
                    result.metrics_after = await self._get_skill_performance_metrics(skill_id)

                # Calculate execution time
                end_time = datetime.now(timezone.utc)
                result.execution_time = (end_time - start_time).total_seconds()

                # Cache the optimization result
                self.optimization_cache[optimization_id] = result

                # Emit completion event
                await self.event_bus.emit(
                    SkillOptimizationCompleted(
                        optimization_id=optimization_id,
                        skill_id=skill_id,
                        execution_time=result.execution_time,
                        applied_changes=len(result.applied_changes) > 0,
                    )
                )

                if self.metrics:
                    self.metrics.record("skill_optimization_time_seconds", result.execution_time)

                    # If we have before and after metrics, calculate improvement
                    if result.metrics_after:
                        if result.metrics_before.avg_execution_time > 0:
                            improvement = (
                                (
                                    result.metrics_before.avg_execution_time
                                    - result.metrics_after.avg_execution_time
                                )
                                / result.metrics_before.avg_execution_time
                                * 100
                            )
                            if improvement > 0:
                                self.metrics.record(
                                    "skill_performance_improvement_percent", improvement
                                )

                return result

        except Exception as e:
            # Handle optimization failure
            error_msg = str(e)
            self.logger.error(f"Skill optimization failed: {error_msg}")

            # Emit failure event
            await self.event_bus.emit(
                SkillOptimizationFailed(
                    optimization_id=optimization_id, skill_id=skill_id, error_message=error_msg
                )
            )

            if self.metrics:
                self.metrics.increment("skill_optimizations_failed")

            # Return result with error
            return OptimizationResult(
                skill_id=skill_id,
                optimization_id=optimization_id,
                optimization_type=optimization_type,
                metrics_before=await self._get_skill_performance_metrics(skill_id),
                error=error_msg,
                execution_time=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

    async def get_optimization_results(self, optimization_id: str) -> Optional[OptimizationResult]:
        """
        Get the results of a previous optimization.

        Args:
            optimization_id: ID of the optimization operation

        Returns:
            Optimization result if found, None otherwise
        """
        return self.optimization_cache.get(optimization_id)

    async def get_skill_performance_report(self, skill_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for a skill.

        Args:
            skill_id: ID of the skill

        Returns:
            Performance report data
        """
        # Get skill information
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Get performance metrics
        metrics = await self._get_skill_performance_metrics(skill_id)

        # Get optimization history
        optimization_history = [
            result for result in self.optimization_cache.values() if result.skill_id == skill_id
        ]

        # Analyze skill code
        code_analysis = await self._analyze_skill_code(skill_id)

        # Generate performance insights
        insights = await self._generate_performance_insights(skill_id, metrics)

        # Combine into report
        report = {
            "skill_id": skill_id,
            "skill_info": skill_info,
            "performance_metrics": vars(metrics),
            "optimization_history": [vars(result) for result in optimization_history],
            "code_analysis": code_analysis,
            "insights": insights,
            "recommendations": await self._generate_optimization_recommendations(skill_id, metrics),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_id": str(uuid.uuid4()),
        }

        return report

    async def analyze_system_wide_performance(self) -> Dict[str, Any]:
        """
        Analyze system-wide skill performance and generate optimization recommendations.

        Returns:
            System performance analysis report
        """
        # Get all registered skills
        skills = self.skill_registry.list_skills()

        # Collect metrics for each skill
        skill_metrics = {}
        for skill in skills:
            skill_id = skill.get("name")
            try:
                metrics = await self._get_skill_performance_metrics(skill_id)
                skill_metrics[skill_id] = metrics
            except Exception as e:
                self.logger.warning(f"Error getting metrics for skill '{skill_id}': {str(e)}")

        # Calculate system-wide statistics
        total_execution_count = sum(m.execution_count for m in skill_metrics.values())
        avg_execution_time = (
            np.mean([m.avg_execution_time for m in skill_metrics.values() if m.execution_count > 0])
            if skill_metrics
            else 0
        )
        system_success_rate = (
            np.mean([m.success_rate for m in skill_metrics.values() if m.execution_count > 0])
            if skill_metrics
            else 1.0
        )

        # Find most used skills
        most_used_skills = sorted(
            [(skill_id, metrics) for skill_id, metrics in skill_metrics.items()],
            key=lambda x: x[1].execution_count,
            reverse=True,
        )[:10]

        # Find slowest skills
        slowest_skills = sorted(
            [
                (skill_id, metrics)
                for skill_id, metrics in skill_metrics.items()
                if metrics.execution_count > 0
            ],
            key=lambda x: x[1].avg_execution_time,
            reverse=True,
        )[:10]

        # Find skills with lowest success rate
        least_reliable_skills = sorted(
            [
                (skill_id, metrics)
                for skill_id, metrics in skill_metrics.items()
                if metrics.execution_count > 0
            ],
            key=lambda x: x[1].success_rate,
        )[:10]

        # Generate system-wide recommendations
        recommendations = await self._generate_system_wide_recommendations(skill_metrics)

        # Build report
        report = {
            "total_skills": len(skills),
            "skills_with_metrics": len(skill_metrics),
            "total_executions": total_execution_count,
            "average_execution_time": avg_execution_time,
            "system_success_rate": system_success_rate,
            "most_used_skills": [
                {
                    "skill_id": s[0],
                    "execution_count": s[1].execution_count,
                    "avg_execution_time": s[1].avg_execution_time,
                }
                for s in most_used_skills
            ],
            "slowest_skills": [
                {
                    "skill_id": s[0],
                    "avg_execution_time": s[1].avg_execution_time,
                    "execution_count": s[1].execution_count,
                }
                for s in slowest_skills
            ],
            "least_reliable_skills": [
                {
                    "skill_id": s[0],
                    "success_rate": s[1].success_rate,
                    "error_count": s[1].error_count,
                }
                for s in least_reliable_skills
            ],
            "recommendations": recommendations,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_id": str(uuid.uuid4()),
        }

        return report

    async def benchmark_skill(
        self, skill_id: str, test_parameters: List[Dict[str, Any]] = None, iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark a skill's performance with standardized tests.

        Args:
            skill_id: ID of the skill to benchmark
            test_parameters: List of parameter sets to test
            iterations: Number of iterations for each parameter set

        Returns:
            Benchmark results
        """
        # Check if skill exists
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Get the skill instance
        skill_instance = self.skill_factory.create_skill(skill_id, self.container)

        # If no test parameters provided, generate some based on skill_parameters schema
        if not test_parameters:
            test_parameters = await self._generate_test_parameters(skill_id, skill_instance)

        benchmark_results = []

        for param_set in test_parameters:
            iteration_results = []

            for i in range(iterations):
                # Prepare for profiling if available
                cpu_profile = None
                memory_profile = None

                if self.cpu_profiler:
                    self.cpu_profiler.start()

                if self.memory_profiler:
                    self.memory_profiler.start()

                # Execute the skill and measure performance
                start_time = time.time()
                try:
                    result = await skill_instance.execute(**param_set)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)

                execution_time = time.time() - start_time

                # Collect profiling data if available
                if self.cpu_profiler:
                    cpu_profile = self.cpu_profiler.stop()

                if self.memory_profiler:
                    memory_profile = self.memory_profiler.stop()

                # Record iteration result
                iteration_results.append(
                    {
                        "iteration": i,
                        "execution_time": execution_time,
                        "success": success,
                        "error": error,
                        "cpu_profile": cpu_profile,
                        "memory_profile": memory_profile,
                    }
                )

            # Calculate statistics for this parameter set
            successful_times = [r["execution_time"] for r in iteration_results if r["success"]]
            avg_time = np.mean(successful_times) if successful_times else None
            min_time = min(successful_times) if successful_times else None
            max_time = max(successful_times) if successful_times else None
            success_rate = len(successful_times) / iterations if iterations > 0 else 0

            benchmark_results.append(
                {
                    "parameters": param_set,
                    "iterations": iterations,
                    "avg_execution_time": avg_time,
                    "min_execution_time": min_time,
                    "max_execution_time": max_time,
                    "success_rate": success_rate,
                    "detailed_results": iteration_results,
                }
            )

        # Compile benchmark report
        benchmark_report = {
            "skill_id": skill_id,
            "skill_info": skill_info,
            "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_id": str(uuid.uuid4()),
            "parameter_sets_tested": len(test_parameters),
            "iterations_per_set": iterations,
            "results": benchmark_results,
            "overall_avg_time": np.mean(
                [
                    r["avg_execution_time"]
                    for r in benchmark_results
                    if r["avg_execution_time"] is not None
                ]
            ),
            "overall_success_rate": np.mean([r["success_rate"] for r in benchmark_results]),
        }

        return benchmark_report

    async def optimize_skill_selection(
        self, similar_skills: List[str], test_case: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize the selection of skills for a specific use case.

        Args:
            similar_skills: List of skill IDs with similar functionality
            test_case: Test parameters to evaluate the skills

        Returns:
            Skill selection recommendation
        """
        if not similar_skills:
            raise ValueError("Must provide at least one skill to evaluate")

        results = []

        for skill_id in similar_skills:
            # Check if skill exists
            skill_info = self.skill_registry.get_skill_info(skill_id)
            if not skill_info:
                self.logger.warning(f"Skill '{skill_id}' is not registered, skipping")
                continue

            # Get the skill instance
            skill_instance = self.skill_factory.create_skill(skill_id, self.container)

            # Execute the skill with the test case
            start_time = time.time()
            try:
                result = await skill_instance.execute(**test_case)
                success = True
                error = None
            except Exception as e:
                result = None
                success = False
                error = str(e)

            execution_time = time.time() - start_time

            # Get memory usage if profiler available
            memory_usage = None
            if self.memory_profiler:
                self.memory_profiler.start()
                await skill_instance.execute(**test_case)
                memory_profile = self.memory_profiler.stop()
                memory_usage = memory_profile.get("peak_memory_mb") if memory_profile else None

            # Get historical performance
            metrics = await self._get_skill_performance_metrics(skill_id)

            # Add to results
            results.append(
                {
                    "skill_id": skill_id,
                    "skill_info": skill_info,
                    "execution_time": execution_time,
                    "success": success,
                    "error": error,
                    "memory_usage": memory_usage,
                    "historical_metrics": vars(metrics),
                    "result": result,
                }
            )

        # Sort by execution time (fastest first) among successful executions
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            sorted_results = sorted(successful_results, key=lambda x: x["execution_time"])
        else:
            sorted_results = []

        # Build recommendation
        recommendation = {
            "test_case": test_case,
            "evaluated_skills": len(results),
            "successful_skills": len(successful_results),
            "recommended_skill": sorted_results[0]["skill_id"] if sorted_results else None,
            "recommendation_basis": "Fastest successful execution",
            "results": results,
            "recommendation_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return recommendation

    async def apply_optimization_suggestions(self, optimization_id: str) -> Dict[str, Any]:
        """
        Apply previously generated optimization suggestions.

        Args:
            optimization_id: ID of the optimization operation

        Returns:
            Result of applying the suggestions
        """
        # Get the optimization result
        optimization_result = self.optimization_cache.get(optimization_id)
        if not optimization_result:
            raise ValueError(f"Optimization ID '{optimization_id}' not found")

        # Check if suggestions exist
        if not optimization_result.suggestions:
            raise ValueError(f"No suggestions found for optimization ID '{optimization_id}'")

        # Copy the original result
        result = copy.deepcopy(optimization_result)
        result.optimization_id = str(uuid.uuid4())  # New ID for the application operation

        # Apply each suggestion
        applied_suggestions = []
        for suggestion in result.suggestions:
            if suggestion.applied:
                continue  # Skip already applied suggestions

            try:
                await self._apply_suggestion(suggestion, result)
                suggestion.applied = True
                applied_suggestions.append(suggestion)
                result.applied_changes.append(
                    f"Applied {suggestion.optimization_type.value} optimization: {suggestion.description}"
                )
            except Exception as e:
                self.logger.error(f"Error applying suggestion: {str(e)}")
                result.error = f"Error applying suggestion: {str(e)}"

        # Update metrics after
        if applied_suggestions:
            result.metrics_after = await self._get_skill_performance_metrics(result.skill_id)

        # Update cache with the new result
        self.optimization_cache[result.optimization_id] = result

        # Return application result
        return {
            "optimization_id": result.optimization_id,
            "original_optimization_id": optimization_id,
            "skill_id": result.skill_id,
            "applied_suggestions": len(applied_suggestions),
            "total_suggestions": len(result.suggestions),
            "applied_changes": result.applied_changes,
            "metrics_before": vars(result.metrics_before),
            "metrics_after": vars(result.metrics_after) if result.metrics_after else None,
            "error": result.error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _get_skill_performance_metrics(self, skill_id: str) -> SkillPerformanceMetrics:
        """
        Get performance metrics for a skill from execution history.

        Args:
            skill_id: ID of the skill

        Returns:
            Performance metrics
        """
        # Initialize metrics
        metrics = SkillPerformanceMetrics(skill_id=skill_id)

        # Get execution history from episodic memory
        # Query for skill executions
        execution_history = await self.episodic_memory.retrieve(
            query={"skill_id": skill_id},
            limit=1000,  # Reasonable limit to prevent memory issues
            sort_by="timestamp",
            ascending=False,
        )

        if not execution_history:
            return metrics  # Return empty metrics if no history

        # Process execution history
        execution_times = []
        errors = []
        parameters = defaultdict(Counter)
        success_count = 0
        cached_count = 0
        cached_times = []

        for execution in execution_history:
            # Extract basic metrics
            execution_time = execution.get("execution_time", 0)
            success = not execution.get("error")
            timestamp = execution.get("timestamp")

            if success:
                success_count += 1
                execution_times.append(execution_time)

                # Check if it was a cached execution
                if execution.get("cached", False):
                    cached_count += 1
                    cached_times.append(execution_time)
            else:
                errors.append(execution.get("error", "Unknown error"))

            # Track parameter distributions
            for param, value in execution.get("parameters", {}).items():
                # For simplicity, only track primitive types
                if isinstance(value, (str, int, float, bool)) or value is None:
                    parameters[param][str(value)] += 1

            # Update last executed time
            if timestamp and (metrics.last_executed is None or timestamp > metrics.last_executed):
                try:
                    metrics.last_executed = (
                        timestamp
                        if isinstance(timestamp, datetime)
                        else datetime.fromisoformat(timestamp)
                    )
                except (ValueError, TypeError):
                    pass

        # Update metrics
        metrics.execution_count = len(execution_history)
        metrics.error_count = metrics.execution_count - success_count
        metrics.success_rate = (
            success_count / metrics.execution_count if metrics.execution_count > 0 else 1.0
        )

        if execution_times:
            metrics.avg_execution_time = np.mean(execution_times)
            metrics.min_execution_time = min(execution_times)
            metrics.max_execution_time = max(execution_times)
            metrics.execution_times = execution_times

        if cached_times:
            metrics.cached_execution_count = cached_count
            metrics.cache_hit_rate = (
                cached_count / metrics.execution_count if metrics.execution_count > 0 else 0
            )
            metrics.avg_cached_execution_time = np.mean(cached_times)

        # Update parameter distributions
        metrics.parameter_distributions = parameters

        # Update common errors
        metrics.common_errors = Counter(errors)

        # Get memory usage if available
        if self.memory_profiler:
            try:
                # Try to profile memory usage
                skill_instance = self.skill_factory.create_skill(skill_id, self.container)

                # Use most common parameters if available
                test_params = {}
                for param, counter in parameters.items():
                    if counter:
                        most_common = counter.most_common(1)[0][0]
                        # Convert to appropriate type if possible
                        try:
                            if most_common.lower() == "true":
                                test_params[param] = True
                            elif most_common.lower() == "false":
                                test_params[param] = False
                            elif most_common.lower() == "none":
                                test_params[param] = None
                            elif most_common.isdigit():
                                test_params[param] = int(most_common)
                            elif most_common.replace(".", "", 1).isdigit():
                                test_params[param] = float(most_common)
                            else:
                                test_params[param] = most_common
                        except:
                            test_params[param] = most_common

                # Start profiling
                self.memory_profiler.start()
                # Execute skill with parameters (catch errors)
                try:
                    await asyncio.wait_for(skill_instance.execute(**test_params), timeout=5.0)
                except:
                    pass
                # Stop profiling and get results
                profile_results = self.memory_profiler.stop()

                # Update memory usage
                if profile_results and "peak_memory_mb" in profile_results:
                    metrics.memory_usage = profile_results["peak_memory_mb"]
            except Exception as e:
                self.logger.warning(
                    f"Error profiling memory usage for skill '{skill_id}': {str(e)}"
                )

        return metrics

    async def _optimize_for_performance(
        self,
        skill_id: str,
        level: OptimizationLevel,
        mode: OptimizationMode,
        result: OptimizationResult,
    ) -> None:
        """
        Optimize a skill for execution performance.

        Args:
            skill_id: ID of the skill
            level: Optimization level
            mode: Optimization mode
            result: Result to update
        """
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Get current code
        skill_path = Path(skill_info["path"])
        if not skill_path.exists():
            raise ValueError(f"Skill file not found: {skill_path}")

        # Performance suggestions
        suggestions = []

        # 1. Analyze code for performance issues
        code_analysis = await self._analyze_skill_code(skill_id)

        # Add suggestions based on code analysis
        if code_analysis.get("async_issues"):
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.PERFORMANCE,
                    description="Improve async code patterns",
                    impact="High",
                    suggested_changes={"async_patterns": code_analysis.get("async_issues")},
                    estimated_improvement="10-30% reduction in execution time",
                    risk_level="medium",
                    rationale="Inefficient async patterns can cause unnecessary waiting and resource usage",
                )
            )

        if code_analysis.get("io_operations") and level != OptimizationLevel.MINIMAL:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.PERFORMANCE,
                    description="Optimize I/O operations",
                    impact="Medium",
                    suggested_changes={"io_operations": code_analysis.get("io_operations")},
                    estimated_improvement="5-15% reduction in execution time for I/O bound skills",
                    risk_level="medium",
                    rationale="Blocking I/O operations can significantly slow down skill execution",
                )
            )

        # 2. Check for caching opportunities
        metrics = result.metrics_before
        if metrics.execution_count > 0 and metrics.cache_hit_rate < 0.5:
            # Check parameter distributions to see if there are common parameter sets
            repeatable_params = False
            for param, counter in metrics.parameter_distributions.items():
                most_common = counter.most_common(1)
                if most_common and most_common[0][1] > metrics.execution_count * 0.3:
                    repeatable_params = True
                    break

            if repeatable_params:
                suggestions.append(
                    OptimizationSuggestion(
                        skill_id=skill_id,
                        optimization_type=OptimizationType.CACHING,
                        description="Implement result caching",
                        impact="High",
                        suggested_changes={
                            "caching": {
                                "enabled": True,
                                "cache_key_params": [
                                    param
                                    for param, counter in metrics.parameter_distributions.items()
                                    if counter.most_common(1)
                                    and counter.most_common(1)[0][1] > metrics.execution_count * 0.3
                                ],
                            }
                        },
                        estimated_improvement=f"Up to {metrics.avg_execution_time * 0.9:.2f}s reduction for repeated calls",
                        risk_level="low",
                        rationale="Skill shows repeated executions with similar parameters",
                    )
                )

        # 3. Check for expensive computations
        if code_analysis.get("expensive_operations") and level != OptimizationLevel.MINIMAL:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.PERFORMANCE,
                    description="Optimize expensive computations",
                    impact="Medium",
                    suggested_changes={
                        "expensive_operations": code_analysis.get("expensive_operations")
                    },
                    estimated_improvement="5-20% reduction in execution time",
                    risk_level="medium",
                    rationale="Expensive computational operations can be optimized or parallelized",
                )
            )

        # 4. Check for model calls that could be optimized
        model_calls = code_analysis.get("model_calls", [])
        if model_calls and level != OptimizationLevel.MINIMAL:
            # Check if there are multiple sequential model calls that could be batched
            if len(model_calls) > 1:
                suggestions.append(
                    OptimizationSuggestion(
                        skill_id=skill_id,
                        optimization_type=OptimizationType.PERFORMANCE,
                        description="Batch model calls where possible",
                        impact="High",
                        suggested_changes={
                            "batch_model_calls": {
                                "model_calls": model_calls,
                                "suggested_batching": "Combine sequential model calls into batched requests",
                            }
                        },
                        estimated_improvement="20-40% reduction in execution time for skills with multiple model calls",
                        risk_level="medium",
                        rationale="Multiple sequential model calls have high latency overhead",
                    )
                )

        # Add suggestions to result
        result.suggestions.extend(suggestions)

        # Apply optimizations if in APPLY mode
        if mode == OptimizationMode.APPLY:
            applied_count = 0
            for suggestion in suggestions:
                if suggestion.optimization_type == OptimizationType.CACHING:
                    # Implement caching
                    await self._apply_caching_optimization(skill_id, suggestion)
                    applied_count += 1
                    result.applied_changes.append(
                        f"Applied caching optimization for skill '{skill_id}'"
                    )

                # Other optimizations that modify code would be applied here
                # This is a simplified implementation

            if applied_count > 0:
                self.logger.info(
                    f"Applied {applied_count} performance optimizations to skill '{skill_id}'"
                )

    async def _optimize_for_memory(
        self,
        skill_id: str,
        level: OptimizationLevel,
        mode: OptimizationMode,
        result: OptimizationResult,
    ) -> None:
        """
        Optimize a skill for memory usage.

        Args:
            skill_id: ID of the skill
            level: Optimization level
            mode: Optimization mode
            result: Result to update
        """
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Get current code
        skill_path = Path(skill_info["path"])
        if not skill_path.exists():
            raise ValueError(f"Skill file not found: {skill_path}")

        # Memory optimization suggestions
        suggestions = []

        # 1. Analyze code for memory issues
        code_analysis = await self._analyze_skill_code(skill_id)

        # Check for large data structures
        if code_analysis.get("large_data_structures"):
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.MEMORY,
                    description="Optimize large data structures",
                    impact="High",
                    suggested_changes={
                        "large_data_structures": code_analysis.get("large_data_structures")
                    },
                    estimated_improvement="20-50% reduction in memory usage",
                    risk_level="medium",
                    rationale="Large in-memory data structures can be replaced with more efficient alternatives",
                )
            )

        # Check for memory leaks or unreleased resources
        if code_analysis.get("potential_memory_leaks"):
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.MEMORY,
                    description="Fix potential memory leaks",
                    impact="High",
                    suggested_changes={
                        "potential_memory_leaks": code_analysis.get("potential_memory_leaks")
                    },
                    estimated_improvement="Prevents memory growth over time",
                    risk_level="low",
                    rationale="Resources are not properly released, causing memory leaks",
                )
            )

        # Check for excessive object creation
        if code_analysis.get("excessive_object_creation") and level != OptimizationLevel.MINIMAL:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.MEMORY,
                    description="Reduce excessive object creation",
                    impact="Medium",
                    suggested_changes={
                        "excessive_object_creation": code_analysis.get("excessive_object_creation")
                    },
                    estimated_improvement="10-30% reduction in memory usage",
                    risk_level="medium",
                    rationale="Creating many temporary objects increases memory pressure and GC overhead",
                )
            )

        # Suggest streaming for large responses if applicable
        if code_analysis.get("large_responses") and level == OptimizationLevel.AGGRESSIVE:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.MEMORY,
                    description="Implement streaming for large responses",
                    impact="High",
                    suggested_changes={"streaming": code_analysis.get("large_responses")},
                    estimated_improvement="50-80% reduction in peak memory usage",
                    risk_level="high",
                    rationale="Processing large responses in memory can be replaced with streaming processing",
                )
            )

        # Add suggestions to result
        result.suggestions.extend(suggestions)

        # Apply optimizations if in APPLY mode
        if mode == OptimizationMode.APPLY:
            applied_count = 0
            for suggestion in suggestions:
                # Apply specific memory optimizations based on suggestion type
                if suggestion.optimization_type == OptimizationType.MEMORY:
                    # Memory optimizations that modify code would be applied here
                    # This is a simplified implementation
                    pass

            if applied_count > 0:
                self.logger.info(
                    f"Applied {applied_count} memory optimizations to skill '{skill_id}'"
                )

    async def _optimize_for_quality(
        self,
        skill_id: str,
        level: OptimizationLevel,
        mode: OptimizationMode,
        result: OptimizationResult,
    ) -> None:
        """
        Optimize a skill for result quality.

        Args:
            skill_id: ID of the skill
            level: Optimization level
            mode: Optimization mode
            result: Result to update
        """
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Quality optimization suggestions
        suggestions = []

        # 1. Analyze error patterns
        metrics = result.metrics_before
        if metrics.error_count > 0 and metrics.execution_count > 0:
            error_rate = metrics.error_count / metrics.execution_count

            if error_rate > 0.1:  # More than 10% errors
                common_errors = metrics.common_errors.most_common(3)

                suggestions.append(
                    OptimizationSuggestion(
                        skill_id=skill_id,
                        optimization_type=OptimizationType.QUALITY,
                        description="Improve error handling",
                        impact="High",
                        suggested_changes={
                            "error_handling": {
                                "error_rate": error_rate,
                                "common_errors": common_errors,
                            }
                        },
                        estimated_improvement=f"Reduce error rate from {error_rate:.2%} to below 5%",
                        risk_level="low",
                        rationale="Skill has a high error rate with consistent error patterns",
                    )
                )

        # 2. Analyze parameter sensitivity
        if metrics.parameter_distributions:
            potential_improvements = []

            for param, counter in metrics.parameter_distributions.items():
                values = list(counter.keys())
                if len(values) > 1:  # Only consider parameters with variation
                    # This is a simplification - in a real system, you would analyze
                    # success rates by parameter value and correlate with results
                    potential_improvements.append(param)

            if potential_improvements and level != OptimizationLevel.MINIMAL:
                suggestions.append(
                    OptimizationSuggestion(
                        skill_id=skill_id,
                        optimization_type=OptimizationType.PARAMETERS,
                        description="Optimize parameter defaults and handling",
                        impact="Medium",
                        suggested_changes={
                            "parameter_tuning": {"parameters_to_tune": potential_improvements}
                        },
                        estimated_improvement="Improved result quality and success rate",
                        risk_level="medium",
                        rationale="Parameter sensitivity analysis shows potential for optimization",
                    )
                )

        # 3. Model selection optimization if using LLMs
        code_analysis = await self._analyze_skill_code(skill_id)
        model_calls = code_analysis.get("model_calls", [])

        if model_calls and level == OptimizationLevel.AGGRESSIVE:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.QUALITY,
                    description="Optimize model selection and parameters",
                    impact="High",
                    suggested_changes={
                        "model_optimization": {
                            "current_models": [
                                call.get("model") for call in model_calls if "model" in call
                            ],
                            "suggested_improvements": "Use more appropriate models or tune parameters for quality",
                        }
                    },
                    estimated_improvement="Significant improvement in result quality",
                    risk_level="medium",
                    rationale="Model selection and parameters can be optimized for this skill's specific needs",
                )
            )

        # Add suggestions to result
        result.suggestions.extend(suggestions)

        # Apply optimizations if in APPLY mode
        if mode == OptimizationMode.APPLY:
            applied_count = 0
            # Quality optimizations often require human review and are less amenable to automatic application
            # However, we could implement certain safe changes

            if applied_count > 0:
                self.logger.info(
                    f"Applied {applied_count} quality optimizations to skill '{skill_id}'"
                )

    async def _optimize_for_caching(
        self,
        skill_id: str,
        level: OptimizationLevel,
        mode: OptimizationMode,
        result: OptimizationResult,
    ) -> None:
        """
        Optimize a skill through caching strategies.

        Args:
            skill_id: ID of the skill
            level: Optimization level
            mode: Optimization mode
            result: Result to update
        """
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Caching optimization suggestions
        suggestions = []

        # 1. Analyze execution patterns
        metrics = result.metrics_before

        # Check if skill is executed frequently
        if metrics.execution_count > 10:
            # Check if there are repeated parameter sets
            parameter_repetition = False
            key_params = []

            for param, counter in metrics.parameter_distributions.items():
                most_common = counter.most_common(1)
                if most_common and most_common[0][1] > 1:  # Param value used more than once
                    parameter_repetition = True
                    if (
                        most_common[0][1] > metrics.execution_count * 0.2
                    ):  # Used in at least 20% of executions
                        key_params.append(param)

            if parameter_repetition:
                # Determine appropriate cache TTL based on data characteristics
                if metrics.last_executed:
                    # Check recency of executions
                    time_since_last = datetime.now(timezone.utc) - metrics.last_executed

                    # Determine cache TTL based on execution frequency
                    if time_since_last < timedelta(hours=1):
                        cache_ttl = 3600  # 1 hour
                    elif time_since_last < timedelta(days=1):
                        cache_ttl = 86400  # 1 day
                    else:
                        cache_ttl = 604800  # 1 week
                else:
                    cache_ttl = 3600  # Default 1 hour

                # Create caching suggestion
                suggestions.append(
                    OptimizationSuggestion(
                        skill_id=skill_id,
                        optimization_type=OptimizationType.CACHING,
                        description="Implement result caching",
                        impact="High",
                        suggested_changes={
                            "caching": {
                                "enabled": True,
                                "cache_key_params": key_params,
                                "ttl_seconds": cache_ttl,
                                "cache_strategy": "default",
                            }
                        },
                        estimated_improvement=f"Up to {metrics.avg_execution_time * 0.9:.2f}s reduction for repeated calls",
                        risk_level="low",
                        rationale="Skill shows pattern of repeated executions with similar parameters",
                    )
                )

        # 2. Check if already cached but with low hit rate
        if metrics.cache_hit_rate > 0 and metrics.cache_hit_rate < 0.5:
            # Suggest cache key optimization
            code_analysis = await self._analyze_skill_code(skill_id)
            current_cache_keys = code_analysis.get("cache_keys", [])

            if current_cache_keys:
                # Analyze parameter distributions to find better cache keys
                improved_keys = []
                for param, counter in metrics.parameter_distributions.items():
                    if (
                        param not in current_cache_keys
                        and len(counter) < metrics.execution_count * 0.5
                    ):
                        # Parameter has repeated values and isn't already a cache key
                        improved_keys.append(param)

                if improved_keys:
                    suggestions.append(
                        OptimizationSuggestion(
                            skill_id=skill_id,
                            optimization_type=OptimizationType.CACHING,
                            description="Optimize cache key selection",
                            impact="Medium",
                            suggested_changes={
                                "cache_key_optimization": {
                                    "current_keys": current_cache_keys,
                                    "suggested_additional_keys": improved_keys,
                                }
                            },
                            estimated_improvement=f"Increase cache hit rate from {metrics.cache_hit_rate:.2%} to ~70%",
                            risk_level="low",
                            rationale="Current cache keys don't capture all parameter repetition patterns",
                        )
                    )

        # 3. For LLM-heavy skills, suggest semantic caching if appropriate
        code_analysis = await self._analyze_skill_code(skill_id)
        model_calls = code_analysis.get("model_calls", [])

        if model_calls and level == OptimizationLevel.AGGRESSIVE:
            # Suggest semantic caching for LLM calls
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.CACHING,
                    description="Implement semantic caching for LLM calls",
                    impact="High",
                    suggested_changes={
                        "semantic_caching": {
                            "enabled": True,
                            "similarity_threshold": 0.92,
                            "embedding_model": "default",
                            "target_model_calls": [
                                call.get("location", "unknown") for call in model_calls
                            ],
                        }
                    },
                    estimated_improvement="20-50% reduction in LLM API costs and latency",
                    risk_level="medium",
                    rationale="Skill makes frequent similar LLM calls that could benefit from semantic caching",
                )
            )

        # Add suggestions to result
        result.suggestions.extend(suggestions)

        # Apply optimizations if in APPLY mode
        if mode == OptimizationMode.APPLY:
            applied_count = 0

            for suggestion in suggestions:
                if suggestion.optimization_type == OptimizationType.CACHING:
                    # Apply caching optimization
                    await self._apply_caching_optimization(skill_id, suggestion)
                    applied_count += 1
                    result.applied_changes.append(
                        f"Applied caching optimization: {suggestion.description}"
                    )

            if applied_count > 0:
                self.logger.info(
                    f"Applied {applied_count} caching optimizations to skill '{skill_id}'"
                )

    async def _optimize_parameters(
        self,
        skill_id: str,
        level: OptimizationLevel,
        mode: OptimizationMode,
        result: OptimizationResult,
    ) -> None:
        """
        Optimize skill parameters based on historical performance.

        Args:
            skill_id: ID of the skill
            level: Optimization level
            mode: Optimization mode
            result: Result to update
        """
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Parameter optimization suggestions
        suggestions = []

        # 1. Analyze parameter impact on performance and success
        metrics = result.metrics_before

        if metrics.execution_count > 10:
            # Get skill instance to access parameter schema
            skill_instance = self.skill_factory.create_skill(skill_id, self.container)
            parameter_schema = getattr(skill_instance, "get_skill_parameters", lambda: {})()

            # Analyze each parameter
            for param_name, distributions in metrics.parameter_distributions.items():
                if not distributions:
                    continue

                # Get parameter schema if available
                param_schema = parameter_schema.get(param_name, {})

                # Check if this is a parameter with performance impact
                if len(distributions) > 1 and metrics.execution_times:
                    # This is a simplification - in a real system, you would correlate
                    # parameter values with execution times and success rates
                    # For this implementation, we'll just identify parameters with variation

                    # Recommend optimal value based on success rates
                    suggested_value = distributions.most_common(1)[0][0]

                    suggestions.append(
                        OptimizationSuggestion(
                            skill_id=skill_id,
                            optimization_type=OptimizationType.PARAMETERS,
                            description=f"Optimize '{param_name}' parameter",
                            impact="Medium",
                            suggested_changes={
                                "parameter_optimization": {
                                    "parameter": param_name,
                                    "current_default": param_schema.get("default", "not specified"),
                                    "suggested_value": suggested_value,
                                    "distribution": dict(distributions),
                                }
                            },
                            estimated_improvement="Improved performance and success rate",
                            risk_level="low",
                            rationale=f"Parameter '{param_name}' shows significant variation with performance impact",
                        )
                    )

        # 2. Check for unused parameters
        code_analysis = await self._analyze_skill_code(skill_id)
        unused_params = code_analysis.get("unused_parameters", [])

        if unused_params and level != OptimizationLevel.MINIMAL:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.PARAMETERS,
                    description="Remove or document unused parameters",
                    impact="Low",
                    suggested_changes={"unused_parameters": unused_params},
                    estimated_improvement="Cleaner API and improved documentation",
                    risk_level="low",
                    rationale="Parameters are defined but not used in skill implementation",
                )
            )

        # 3. Check for parameter validation issues
        validation_issues = code_analysis.get("parameter_validation_issues", [])

        if validation_issues:
            suggestions.append(
                OptimizationSuggestion(
                    skill_id=skill_id,
                    optimization_type=OptimizationType.PARAMETERS,
                    description="Improve parameter validation",
                    impact="Medium",
                    suggested_changes={"parameter_validation": validation_issues},
                    estimated_improvement="Reduced errors and improved robustness",
                    risk_level="low",
                    rationale="Missing or insufficient parameter validation can lead to errors",
                )
            )

        # Add suggestions to result
        result.suggestions.extend(suggestions)

        # Apply optimizations if in APPLY mode
        if mode == OptimizationMode.APPLY:
            applied_count = 0

            for suggestion in suggestions:
                if suggestion.optimization_type == OptimizationType.PARAMETERS:
                    # Parameter optimizations often require code changes
                    if "parameter_optimization" in suggestion.suggested_changes:
                        param_info = suggestion.suggested_changes["parameter_optimization"]
                        await self._apply_parameter_optimization(skill_id, param_info)
                        applied_count += 1
                        result.applied_changes.append(
                            f"Optimized parameter '{param_info['parameter']}' with value '{param_info['suggested_value']}'"
                        )

            if applied_count > 0:
                self.logger.info(
                    f"Applied {applied_count} parameter optimizations to skill '{skill_id}'"
                )

    async def _optimize_code(
        self,
        skill_id: str,
        level: OptimizationLevel,
        mode: OptimizationMode,
        result: OptimizationResult,
    ) -> None:
        """
        Optimize skill code for better performance and quality.

        Args:
            skill_id: ID of the skill
            level: Optimization level
            mode: Optimization mode
            result: Result to update
        """
        skill_info = self.skill_registry.get_skill_info(skill_id)
        if not skill_info:
            raise ValueError(f"Skill '{skill_id}' is not registered")

        # Get current code
        skill_path = Path(skill_info["path"])
        if not skill_path.exists():
            raise ValueError(f"Skill file not found: {skill_path}")

        with open(skill_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        # Code optimization suggestions
        suggestions = []

        # 1. Analyze code quality
        code_analysis = await self._analyze_skill_code(skill_id)

        # Check for code quality issues
        code_quality_issues = code_analysis.get("code_quality_issues", [])
        if code_quality_issues:
            # Create code improvement suggestions
            for issue in code_quality_issues:
                issue_type = issue.get("type", "unknown")
                location = issue.get("location", "unknown")
                description = issue.get("description", "Code quality issue")
                suggested_fix = issue.get("suggested_fix")

                if suggested_fix:
                    suggestions.append(
                        OptimizationSuggestion(
                            skill_id=skill_id,
                            optimization_type=OptimizationType.CODE,
                            description=f"Fix {issue_type} issue at {location}",
                            impact="Medium",
                            code_changes={issue.get("original_code", ""): suggested_fix},
                            estimated_improvement="Improved code quality and maintainability",
                            risk_level="medium",
                            rationale=description,
                        )
                    )

        # 2. Check for error handling improvements
        error_handling_issues = code_analysis.get("error_handling_issues", [])
        if error_handling_issues:
            for issue in error_handling_issues:
                location = issue.get("location", "unknown")
                description = issue.get("description", "Error handling issue")
                suggested_fix = issue.get("suggested_fix")
