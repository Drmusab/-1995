"""
Skill Composer Module
Author: Drmusab
Last Modified: 2025-07-05 22:06:35 UTC

This module provides functionality for composing multiple skills into more
complex workflows and creating new composite skills. It enables the dynamic
creation of skill pipelines, parallel skill execution, conditional branching,
and skill chaining for advanced AI capabilities.
"""

import copy
import importlib
import inspect
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import asyncio
import networkx as nx
import yaml

# Assistant components
from src.assistant.component_manager import ComponentManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SkillCompositionCompleted,
    SkillCompositionFailed,
    SkillCompositionStarted,
    SkillExecutionCompleted,
    SkillExecutionFailed,
    SkillExecutionStarted,
    SystemConfigurationChanged,
)

# Memory components
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.reasoning.planning.goal_decomposer import GoalDecomposer

# Reasoning
from src.reasoning.planning.task_planner import TaskPlanner
from src.skills.skill_factory import SkillFactory

# Skill management
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator


class CompositionType(Enum):
    """Types of skill compositions."""

    SEQUENTIAL = "sequential"  # Execute skills in sequence
    PARALLEL = "parallel"  # Execute skills in parallel
    CONDITIONAL = "conditional"  # Execute skills based on conditions
    ITERATIVE = "iterative"  # Execute skills in a loop
    AGGREGATED = "aggregated"  # Combine results from multiple skills
    DYNAMIC = "dynamic"  # Determine skills to execute at runtime


class ExecutionMode(Enum):
    """Execution modes for composite skills."""

    ALL = "all"  # Execute all component skills
    FIRST_SUCCESS = "first_success"  # Stop after first successful skill
    BEST_RESULT = "best_result"  # Execute all and pick best result
    FALLBACK = "fallback"  # Try skills in order until success
    CONSENSUS = "consensus"  # Execute all and use consensus result


class DataFlow(Enum):
    """Data flow patterns between skills."""

    DIRECT = "direct"  # Output directly to input
    TRANSFORMED = "transformed"  # Transform output before input
    MERGED = "merged"  # Merge multiple outputs
    FILTERED = "filtered"  # Filter output before input
    MAPPED = "mapped"  # Map specific output fields to input fields


@dataclass
class SkillNode:
    """A node representing a skill in a composition."""

    skill_id: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    transform: Optional[str] = None
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    required: bool = True
    timeout_seconds: int = 30
    retry_count: int = 0
    fallback_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeSkillDefinition:
    """Definition of a composite skill."""

    name: str
    description: str
    version: str = "0.1.0"
    author: str = "system"
    composition_type: CompositionType = CompositionType.SEQUENTIAL
    execution_mode: ExecutionMode = ExecutionMode.ALL
    skills: List[SkillNode] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    category: str = "custom"
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    data_flow: DataFlow = DataFlow.DIRECT
    max_execution_time: int = 300  # seconds
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SkillComposer:
    """
    Handles the composition of skills into more complex workflows.

    This class provides functionality to:
    - Combine multiple skills into a single composite skill
    - Define execution flow between skills (sequential, parallel, conditional)
    - Manage data transformations between skills
    - Create reusable skill workflows
    - Validate skill compositions for correctness
    - Register composite skills with the skill registry
    """

    def __init__(self, container: Container):
        """
        Initialize the skill composer.

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

        # Memory components
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)

        # Reasoning components
        try:
            self.task_planner = container.get(TaskPlanner)
            self.goal_decomposer = container.get(GoalDecomposer)
        except Exception as e:
            self.logger.warning(f"Reasoning components not available: {str(e)}")
            self.task_planner = None
            self.goal_decomposer = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Set up paths
        self.base_skills_path = self._get_skills_directory()
        self.composite_skills_path = self.base_skills_path / "custom" / "composite"

        # Ensure required directories exist
        self._ensure_directories()

        # Register metrics
        if self.metrics:
            self.metrics.register_counter("skill_compositions_total")
            self.metrics.register_counter("skill_compositions_failed")
            self.metrics.register_counter("composite_skill_executions_total")
            self.metrics.register_histogram("skill_composition_time_seconds")
            self.metrics.register_histogram("composite_skill_execution_time_seconds")

    def _get_skills_directory(self) -> Path:
        """Get the base directory for skills."""
        # First try to get from config
        skills_dir = self.config.get("skills.directory", None)
        if skills_dir:
            return Path(skills_dir)

        # Default to src/skills
        return Path("src") / "skills"

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.composite_skills_path.mkdir(parents=True, exist_ok=True)
        # Ensure __init__.py exists
        init_file = self.composite_skills_path / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    async def create_composite_skill(self, definition: CompositeSkillDefinition) -> Dict[str, Any]:
        """
        Create a new composite skill based on the provided definition.

        Args:
            definition: Composite skill definition

        Returns:
            Information about the created composite skill
        """
        composition_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment("skill_compositions_total")

        # Emit composition started event
        await self.event_bus.emit(
            SkillCompositionStarted(
                composition_id=composition_id,
                skill_name=definition.name,
                composition_type=definition.composition_type.value,
            )
        )

        result = {
            "composition_id": composition_id,
            "skill_name": definition.name,
            "status": "pending",
            "started_at": start_time.isoformat(),
            "completed_at": None,
            "duration_seconds": None,
            "error": None,
            "skill_path": None,
        }

        try:
            # Validate the composition
            await self._validate_composition(definition)

            # Generate the composite skill class
            skill_path = await self._generate_composite_skill(definition)
            result["skill_path"] = str(skill_path)

            # Register the composite skill
            await self._register_composite_skill(definition, skill_path)

            # Successful composition
            result["status"] = "completed"

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration

            # Log success
            self.logger.info(
                f"Successfully created composite skill {definition.name} "
                f"of type {definition.composition_type.value} in {duration:.2f} seconds"
            )

            # Emit completion event
            await self.event_bus.emit(
                SkillCompositionCompleted(
                    composition_id=composition_id,
                    skill_name=definition.name,
                    execution_time=duration,
                )
            )

            if self.metrics:
                self.metrics.record("skill_composition_time_seconds", duration)

            return result

        except Exception as e:
            # Handle composition failure
            result["status"] = "failed"
            result["error"] = str(e)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration

            # Log error
            self.logger.error(f"Skill composition failed: {str(e)}")

            # Emit failure event
            await self.event_bus.emit(
                SkillCompositionFailed(
                    composition_id=composition_id,
                    skill_name=definition.name,
                    error_message=str(e),
                    execution_time=duration,
                )
            )

            if self.metrics:
                self.metrics.increment("skill_compositions_failed")

            return result

    async def execute_composite_skill(
        self, skill_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a composite skill by name with the provided parameters.

        Args:
            skill_name: Name of the composite skill
            parameters: Input parameters for the skill

        Returns:
            Execution results
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment("composite_skill_executions_total")

        # Emit execution started event
        await self.event_bus.emit(
            SkillExecutionStarted(
                skill_id=skill_name, execution_id=execution_id, parameters=parameters
            )
        )

        try:
            with (
                self.tracer.trace(f"composite_skill_execution_{skill_name}")
                if self.tracer
                else None
            ):
                # Get the skill instance
                skill_instance = self.skill_factory.create_skill(skill_name, self.container)

                # Execute the skill
                result = await skill_instance.execute(**parameters)

                # Calculate execution time
                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                # Add execution metadata
                if isinstance(result, dict):
                    result["execution_metadata"] = {
                        "execution_id": execution_id,
                        "skill_name": skill_name,
                        "execution_time": duration,
                        "executed_at": end_time.isoformat(),
                    }

                # Emit completion event
                await self.event_bus.emit(
                    SkillExecutionCompleted(
                        skill_id=skill_name, execution_id=execution_id, execution_time=duration
                    )
                )

                if self.metrics:
                    self.metrics.record("composite_skill_execution_time_seconds", duration)

                return result

        except Exception as e:
            # Handle execution failure
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Log error
            self.logger.error(f"Composite skill execution failed: {str(e)}")

            # Emit failure event
            await self.event_bus.emit(
                SkillExecutionFailed(
                    skill_id=skill_name, execution_id=execution_id, error_message=str(e)
                )
            )

            # Return error result
            return {
                "error": str(e),
                "execution_metadata": {
                    "execution_id": execution_id,
                    "skill_name": skill_name,
                    "execution_time": duration,
                    "executed_at": end_time.isoformat(),
                    "status": "failed",
                },
            }

    async def list_composite_skills(self) -> List[Dict[str, Any]]:
        """
        List all registered composite skills.

        Returns:
            List of composite skill information
        """
        all_skills = self.skill_registry.list_skills()
        composite_skills = []

        for skill in all_skills:
            try:
                # Check if it's a composite skill
                skill_info = self.skill_registry.get_skill_info(skill["name"])
                if not skill_info:
                    continue

                skill_path = Path(skill_info.get("path", ""))
                if not skill_path.exists():
                    continue

                # Look for composite skill marker or metadata
                with open(skill_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "CompositeSkill" in content or "composite_skill_definition" in content:
                        # Extract definition from the file
                        definition = await self._extract_composite_definition(skill_path)

                        # Add to list with additional information
                        composite_skills.append(
                            {
                                **skill,
                                "composition_type": definition.get("composition_type", "unknown"),
                                "component_skills": definition.get("component_skills", []),
                                "description": definition.get("description", ""),
                                "author": definition.get("author", "unknown"),
                                "created_at": definition.get("created_at", ""),
                            }
                        )
            except Exception as e:
                self.logger.warning(f"Error processing skill {skill['name']}: {str(e)}")

        return composite_skills

    async def get_composite_skill_details(self, skill_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific composite skill.

        Args:
            skill_name: Name of the composite skill

        Returns:
            Detailed information about the composite skill
        """
        skill_info = self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            raise ValueError(f"Skill '{skill_name}' is not installed")

        skill_path = Path(skill_info.get("path", ""))
        if not skill_path.exists():
            raise ValueError(f"Skill path '{skill_path}' does not exist")

        # Extract definition
        definition = await self._extract_composite_definition(skill_path)

        # Get source code
        with open(skill_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Build dependency graph
        graph = await self._build_dependency_graph(definition)

        return {
            "name": skill_name,
            "path": str(skill_path),
            "definition": definition,
            "source_code": source_code,
            "dependency_graph": graph,
            "component_skills": definition.get("component_skills", []),
        }

    async def update_composite_skill(
        self, skill_name: str, updated_definition: CompositeSkillDefinition
    ) -> Dict[str, Any]:
        """
        Update an existing composite skill with a new definition.

        Args:
            skill_name: Name of the existing composite skill
            updated_definition: New skill definition

        Returns:
            Update result information
        """
        # Check if skill exists
        skill_info = self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            raise ValueError(f"Composite skill '{skill_name}' does not exist")

        # Ensure the names match
        if skill_name != updated_definition.name:
            raise ValueError(
                f"Cannot change skill name from '{skill_name}' to '{updated_definition.name}'"
            )

        # Create a new composite skill (will replace the old one)
        result = await self.create_composite_skill(updated_definition)

        return {**result, "updated": True, "previous_version": skill_info.get("version", "unknown")}

    async def delete_composite_skill(self, skill_name: str) -> Dict[str, Any]:
        """
        Delete a composite skill.

        Args:
            skill_name: Name of the composite skill to delete

        Returns:
            Deletion result information
        """
        # Check if skill exists
        skill_info = self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            raise ValueError(f"Composite skill '{skill_name}' does not exist")

        skill_path = Path(skill_info.get("path", ""))
        if not skill_path.exists():
            raise ValueError(f"Skill path '{skill_path}' does not exist")

        # Verify it's a composite skill
        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "CompositeSkill" not in content and "composite_skill_definition" not in content:
                raise ValueError(f"Skill '{skill_name}' is not a composite skill")

        # Unregister the skill
        self.skill_registry.unregister_skill(skill_name)

        # Delete the file
        skill_path.unlink()

        return {
            "skill_name": skill_name,
            "status": "deleted",
            "deleted_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _validate_composition(self, definition: CompositeSkillDefinition) -> None:
        """
        Validate a skill composition for correctness.

        Args:
            definition: Composite skill definition

        Raises:
            ValueError: If validation fails
        """
        # Check for empty composition
        if not definition.skills:
            raise ValueError("Composition must contain at least one skill")

        # Check for duplicate skill name
        existing_skill = self.skill_registry.get_skill_info(definition.name)
        if existing_skill:
            # Allow for updates to existing composite skills
            skill_path = Path(existing_skill.get("path", ""))
            if skill_path.exists():
                with open(skill_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if (
                        "CompositeSkill" not in content
                        and "composite_skill_definition" not in content
                    ):
                        raise ValueError(
                            f"A non-composite skill with name '{definition.name}' already exists"
                        )

        # Validate each component skill
        for node in definition.skills:
            skill_info = self.skill_registry.get_skill_info(node.skill_id)
            if not skill_info:
                raise ValueError(
                    f"Skill '{node.skill_id}' is not registered and cannot be used in composition"
                )

        # Validate conditional nodes
        if definition.composition_type == CompositionType.CONDITIONAL:
            for node in definition.skills:
                if not node.condition:
                    raise ValueError(
                        f"Skill '{node.skill_id}' in conditional composition must have a condition"
                    )

        # Validate dependency graph for cycles
        try:
            graph = await self._build_dependency_graph(vars(definition))
            if graph.get("has_cycle", False):
                raise ValueError("Composition contains cyclic dependencies, which is not supported")
        except Exception as e:
            raise ValueError(f"Error validating dependency graph: {str(e)}")

        # Additional validation for specific composition types
        if definition.composition_type == CompositionType.SEQUENTIAL:
            # No additional validation needed
            pass
        elif definition.composition_type == CompositionType.PARALLEL:
            # Check for overlapping output mappings
            output_keys = set()
            for node in definition.skills:
                node_outputs = set(node.output_mapping.values())
                if node_outputs & output_keys:
                    overlap = node_outputs & output_keys
                    raise ValueError(f"Overlapping output mappings found: {overlap}")
                output_keys.update(node_outputs)
        elif definition.composition_type == CompositionType.ITERATIVE:
            # Ensure there's a stop condition
            has_stop_condition = False
            for node in definition.skills:
                if node.condition and "stop" in node.condition.lower():
                    has_stop_condition = True
                    break
            if not has_stop_condition:
                raise ValueError("Iterative composition must have a stop condition")

    async def _generate_composite_skill(self, definition: CompositeSkillDefinition) -> Path:
        """
        Generate a Python file for the composite skill.

        Args:
            definition: Composite skill definition

        Returns:
            Path to the generated skill file
        """
        # Create safe file name
        safe_name = definition.name.replace(" ", "_").replace("-", "_").lower()
        skill_file = self.composite_skills_path / f"{safe_name}.py"

        # Generate the skill class code
        skill_code = self._generate_skill_code(definition)

        # Write to file
        with open(skill_file, "w", encoding="utf-8") as f:
            f.write(skill_code)

        return skill_file

    def _generate_skill_code(self, definition: CompositeSkillDefinition) -> str:
        """
        Generate Python code for a composite skill class.

        Args:
            definition: Composite skill definition

        Returns:
            Generated Python code
        """
        # Convert definition to serializable dict
        definition_dict = {
            "name": definition.name,
            "description": definition.description,
            "version": definition.version,
            "author": definition.author,
            "composition_type": definition.composition_type.value,
            "execution_mode": definition.execution_mode.value,
            "skills": [vars(skill) for skill in definition.skills],
            "dependencies": definition.dependencies,
            "category": definition.category,
            "input_schema": definition.input_schema,
            "output_schema": definition.output_schema,
            "data_flow": definition.data_flow.value,
            "max_execution_time": definition.max_execution_time,
            "tags": definition.tags,
            "metadata": definition.metadata,
            "created_at": definition.created_at,
            "component_skills": [skill.skill_id for skill in definition.skills],
        }

        # Format as JSON for embedding in the code
        definition_json = json.dumps(definition_dict, indent=4)

        # Generate code with appropriate imports and class definition
        code = f'''"""
{definition.name} - Composite Skill
Author: {definition.author}
Version: {definition.version}
Created: {definition.created_at}

{definition.description}
"""

import asyncio
import json
import uuid
import copy
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum

# Core imports
from src.core.dependency_injection import Container
from src.core.error_handling import handle_exceptions

# For evaluating conditions
from src.reasoning.logic_engine import LogicEngine

# Base classes
from src.skills.skill_factory import SkillFactory


class CompositeSkill:
    """
    {definition.name} - A composite skill of type {definition.composition_type.value}.
    
    {definition.description}
    """
    
    def __init__(self, container: Container):
        """Initialize the composite skill."""
        self.container = container
        self.skill_factory = container.get(SkillFactory)
        
        # Try to get additional components
        try:
            self.logic_engine = container.get(LogicEngine)
        except Exception:
            self.logic_engine = None
        
        # Observability components
        try:
            from src.observability.logging.config import get_logger
            from src.observability.monitoring.metrics import MetricsCollector
            from src.observability.monitoring.tracing import TraceManager
            
            self.logger = get_logger("{definition.name}")
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger = None
            self.metrics = None
            self.tracer = None
        
        # Load composite skill definition
        self.composite_skill_definition = {definition_json}
    
    def get_skill_id(self) -> str:
        """Get the unique skill identifier."""
        return "{definition.name}"
    
    def get_skill_description(self) -> str:
        """Get the skill description."""
        return "{definition.description}"
    
    def get_skill_category(self) -> str:
        """Get the skill category."""
        return "{definition.category}"
    
    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the skill parameters schema."""
        return self.composite_skill_definition.get("input_schema", {{}})
    
    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the composite skill.
        
        Returns:
            Execution results
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Log execution start
        if self.logger:
            self.logger.info(f"Executing composite skill {{self.get_skill_id()}} with execution_id {{execution_id}}")
        
        # Track metrics
        if self.metrics:
            self.metrics.increment(f"skill_{{self.get_skill_id()}}_executions_total")
        
        # Initialize execution context
        context = {{
            "input": kwargs,
            "output": {{}},
            "intermediate_results": {{}},
            "execution_path": [],
            "errors": [],
            "start_time": start_time,
            "execution_id": execution_id
        }}
        
        try:
            # Execute based on composition type
            composition_type = self.composite_skill_definition.get("composition_type")
            
            if composition_type == "{CompositionType.SEQUENTIAL.value}":
                result = await self._execute_sequential(context)
            elif composition_type == "{CompositionType.PARALLEL.value}":
                result = await self._execute_parallel(context)
            elif composition_type == "{CompositionType.CONDITIONAL.value}":
                result = await self._execute_conditional(context)
            elif composition_type == "{CompositionType.ITERATIVE.value}":
                result = await self._execute_iterative(context)
            elif composition_type == "{CompositionType.AGGREGATED.value}":
                result = await self._execute_aggregated(context)
            elif composition_type == "{CompositionType.DYNAMIC.value}":
                result = await self._execute_dynamic(context)
            else:
                # Default to sequential
                result = await self._execute_sequential(context)
            
            # Calculate execution time
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            # Add execution metadata
            result["metadata"] = {{
                "execution_id": execution_id,
                "composition_type": composition_type,
                "execution_time": duration,
                "execution_path": context["execution_path"],
                "errors": context["errors"] if context["errors"] else None
            }}
            
            # Log completion
            if self.logger:
                self.logger.info(
                    f"Completed composite skill {{self.get_skill_id()}} "
                    f"in {{duration:.2f}} seconds"
                )
            
            # Record metrics
            if self.metrics:
                self.metrics.record(f"skill_{{self.get_skill_id()}}_execution_time_seconds", duration)
            
            return result
            
        except Exception as e:
            # Handle execution error
            if self.logger:
                self.logger.error(f"Error executing composite skill {{self.get_skill_id()}}: {{str(e)}}")
            
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            
            return {{
                "error": str(e),
                "metadata": {{
                    "execution_id": execution_id,
                    "execution_time": duration,
                    "execution_path": context["execution_path"],
                    "errors": context["errors"] + [str(e)] if context["errors"] else [str(e)]
                }}
            }}
    
    async def _execute_sequential(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skills in sequence."""
        skills = self.composite_skill_definition.get("skills", [])
        current_input = copy.deepcopy(context["input"])
        
        for i, skill_node in enumerate(skills):
            try:
                # Get skill instance
                skill_id = skill_node["skill_id"]
                skill_instance = self.skill_factory.create_skill(skill_id, self.container)
                
                # Prepare input parameters
                params = self._prepare_skill_input(skill_node, current_input, context)
                
                # Execute skill
                context["execution_path"].append(skill_id)
                skill_result = await skill_instance.execute(**params)
                
                # Process output
                if isinstance(skill_result, dict):
                    # Save intermediate result
                    context["intermediate_results"][skill_id] = skill_result
                    
                    # Apply output mapping
                    mapped_result = self._apply_output_mapping(skill_node, skill_result)
                    
                    # Update current input for next skill and output
                    current_input.update(mapped_result)
                    context["output"].update(mapped_result)
                else:
                    context["errors"].append(f"Skill {{skill_id}} returned non-dict result: {{type(skill_result)}}")
            
            except Exception as e:
                context["errors"].append(f"Error executing skill {{skill_node['skill_id']}}: {{str(e)}}")
                
                # Check if this skill is required
                if skill_node.get("required", True):
                    raise
        
        return context["output"]
    
    async def _execute_parallel(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skills in parallel."""
        skills = self.composite_skill_definition.get("skills", [])
        execution_mode = self.composite_skill_definition.get("execution_mode")
        
        # Prepare tasks
        tasks = []
        for skill_node in skills:
            # Get skill instance
            skill_id = skill_node["skill_id"]
            skill_instance = self.skill_factory.create_skill(skill_id, self.container)
            
            # Prepare input parameters
            params = self._prepare_skill_input(skill_node, context["input"], context)
            
            # Create task
            task = asyncio.create_task(self._execute_skill_with_metadata(
                skill_instance, skill_id, params, skill_node
            ))
            tasks.append(task)
        
        # Execute tasks based on execution mode
        if execution_mode == "{ExecutionMode.FIRST_SUCCESS.value}":
            # Wait for first successful result
            result = None
            for future in asyncio.as_completed(tasks):
                try:
                    skill_result = await future
                    if not skill_result.get("error"):
                        result = skill_result
                        context["execution_path"].append(skill_result["metadata"]["skill_id"])
                        break
                except Exception as e:
                    context["errors"].append(str(e))
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            if result:
                context["output"].update(result["result"])
            else:
                raise RuntimeError("All parallel skills failed to execute")
        
        elif execution_mode == "{ExecutionMode.FALLBACK.value}":
            # Try skills in order until success
            for task in tasks:
                try:
                    skill_result = await task
                    skill_id = skill_result["metadata"]["skill_id"]
                    context["execution_path"].append(skill_id)
                    
                    if not skill_result.get("error"):
                        context["output"].update(skill_result["result"])
                        break
                    else:
                        context["errors"].append(skill_result["error"])
                except Exception as e:
                    context["errors"].append(str(e))
            
            if not context["output"]:
                raise RuntimeError("All fallback skills failed to execute")
        
        else:  # Default: ALL or others
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    context["errors"].append(str(result))
                else:
                    skill_id = result["metadata"]["skill_id"]
                    context["execution_path"].append(skill_id)
                    
                    if not result.get("error"):
                        # Store intermediate result
                        context["intermediate_results"][skill_id] = result["result"]
                        
                        # Apply output mapping and update output
                        skill_node = next((s for s in skills if s["skill_id"] == skill_id), None)
                        if skill_node:
                            mapped_result = self._apply_output_mapping(skill_node, result["result"])
                            context["output"].update(mapped_result)
                    else:
                        context["errors"].append(result["error"])
        
        return context["output"]
    
    async def _execute_conditional(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skills based on conditions."""
        skills = self.composite_skill_definition.get("skills", [])
        
        for skill_node in skills:
            # Evaluate condition
            condition = skill_node.get("condition")
            if not condition:
                continue
            
            condition_met = self._evaluate_condition(condition, context)
            if not condition_met:
                continue
            
            try:
                # Get skill instance
                skill_id = skill_node["skill_id"]
                skill_instance = self.skill_factory.create_skill(skill_id, self.container)
                
                # Prepare input parameters
                params = self._prepare_skill_input(skill_node, context["input"], context)
                
                # Execute skill
                context["execution_path"].append(skill_id)
                skill_result = await skill_instance.execute(**params)
                
                # Process output
                if isinstance(skill_result, dict):
                    # Save intermediate result
                    context["intermediate_results"][skill_id] = skill_result
                    
                    # Apply output mapping
                    mapped_result = self._apply_output_mapping(skill_node, skill_result)
                    
                    # Update output
                    context["output"].update(mapped_result)
                    context["input"].update(mapped_result)  # Update input for next conditions
                else:
                    context["errors"].append(f"Skill {{skill_id}} returned non-dict result: {{type(skill_result)}}")
            
            except Exception as e:
                context["errors"].append(f"Error executing skill {{skill_node['skill_id']}}: {{str(e)}}")
                
                # Check if this skill is required
                if skill_node.get("required", True):
                    raise
        
        return context["output"]
    
    async def _execute_iterative(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skills in a loop until a condition is met."""
        skills = self.composite_skill_definition.get("skills", [])
        max_iterations = self.composite_skill_definition.get("metadata", {{}}).get("max_iterations", 10)
        
        iteration = 0
        current_input = copy.deepcopy(context["input"])
        
        # Add iteration to context
        context["iteration"] = iteration
        
        while iteration < max_iterations:
            # Check stop condition before executing skills
            for skill_node in skills:
                stop_condition = skill_node.get("condition", "")
                if "stop" in stop_condition.lower():
                    if self._evaluate_condition(stop_condition, context):
                        # Stop condition met, exit loop
                        return context["output"]
            
            # Execute one iteration of skills
            for skill_node in skills:
                # Skip stop condition nodes
                if skill_node.get("condition", "").lower().startswith("stop"):
                    continue
                
                # Check if this skill should be executed in this iteration
                condition = skill_node.get("condition")
                if condition and not self._evaluate_condition(condition, context):
                    continue
                
                try:
                    # Get skill instance
                    skill_id = skill_node["skill_id"]
                    skill_instance = self.skill_factory.create_skill(skill_id, self.container)
                    
                    # Prepare input parameters
                    params = self._prepare_skill_input(skill_node, current_input, context)
                    
                    # Execute skill
                    iteration_skill_id = f"{{skill_id}}_iter{{iteration}}"
                    context["execution_path"].append(iteration_skill_id)
                    skill_result = await skill_instance.execute(**params)
                    
                    # Process output
                    if isinstance(skill_result, dict):
                        # Save intermediate result
                        context["intermediate_results"][iteration_skill_id] = skill_result
                        
                        # Apply output mapping
                        mapped_result = self._apply_output_mapping(skill_node, skill_result)
                        
                        # Update current input for next skill and output
                        current_input.update(mapped_result)
                        context["output"].update(mapped_result)
                    else:
                        context["errors"].append(
                            f"Skill {{skill_id}} (iteration {{iteration}}) "
                            f"returned non-dict result: {{type(skill_result)}}"
                        )
                
                except Exception as e:
                    context["errors"].append(
                        f"Error executing skill {{skill_node['skill_id']}} "
                        f"(iteration {{iteration}}): {{str(e)}}"
                    )
                    
                    # Check if this skill is required
                    if skill_node.get("required", True):
                        raise
            
            # Update iteration counter
            iteration += 1
            context["iteration"] = iteration
        
        # Max iterations reached
        context["errors"].append(f"Max iterations ({{max_iterations}}) reached")
        return context["output"]
    
    async def _execute_aggregated(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from multiple skills."""
        skills = self.composite_skill_definition.get("skills", [])
        data_flow = self.composite_skill_definition.get("data_flow")
        
        # Execute skills in parallel
        tasks = []
        for skill_node in skills:
            # Get skill instance
            skill_id = skill_node["skill_id"]
            skill_instance = self.skill_factory.create_skill(skill_id, self.container)
            
            # Prepare input parameters
            params = self._prepare_skill_input(skill_node, context["input"], context)
            
            # Create task
            task = asyncio.create_task(self._execute_skill_with_metadata(
                skill_instance, skill_id, params, skill_node
            ))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results based on data flow
        if data_flow == "{DataFlow.MERGED.value}":
            # Merge all results
            for result in results:
                if isinstance(result, Exception):
                    context["errors"].append(str(result))
                else:
                    skill_id = result["metadata"]["skill_id"]
                    context["execution_path"].append(skill_id)
                    
                    if not result.get("error"):
                        # Store intermediate result
                        context["intermediate_results"][skill_id] = result["result"]
                        
                        # Apply output mapping and update output
                        skill_node = next((s for s in skills if s["skill_id"] == skill_id), None)
                        if skill_node:
                            mapped_result = self._apply_output_mapping(skill_node, result["result"])
                            
                            # Merge nested dictionaries
                            self._deep_merge(context["output"], mapped_result)
                    else:
                        context["errors"].append(result["error"])
        
        elif data_flow == "{DataFlow.FILTERED.value}":
            # Filter and combine results
            for result in results:
                if isinstance(result, Exception):
                    context["errors"].append(str(result))
                    continue
                
                skill_id = result["metadata"]["skill_id"]
                context["execution_path"].append(skill_id)
                
                if result.get("error"):
                    context["errors"].append(result["error"])
                    continue
                
                # Store intermediate result
                context["intermediate_results"][skill_id] = result["result"]
                
                # Apply output mapping and filtering
                skill_node = next((s for s in skills if s["skill_id"] == skill_id), None)
                if skill_node:
                    # Apply any transforms
                    transform = skill_node.get("transform")
                    if transform:
                        result["result"] = self._apply_transform(transform, result["result"], context)
                    
                    # Apply output mapping
                    mapped_result = self._apply_output_mapping(skill_node, result["result"])
                    
                    # Update output
                    context["output"].update(mapped_result)
        
        else:  # Default or DIRECT
            # Simply combine all results
            for result in results:
                if isinstance(result, Exception):
                    context["errors"].append(str(result))
                else:
                    skill_id = result["metadata"]["skill_id"]
                    context["execution_path"].append(skill_id)
                    
                    if not result.get("error"):
                        # Store intermediate result
                        context["intermediate_results"][skill_id] = result["result"]
                        
                        # Apply output mapping and update output
                        skill_node = next((s for s in skills if s["skill_id"] == skill_id), None)
                        if skill_node:
                            mapped_result = self._apply_output_mapping(skill_node, result["result"])
                            context["output"].update(mapped_result)
                    else:
                        context["errors"].append(result["error"])
        
        return context["output"]
    
    async def _execute_dynamic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamically determine skills to execute."""
        # This is a placeholder for custom dynamic execution logic
        # In a real implementation, this would determine skills to execute based on input or other factors
        
        # For now, we'll fall back to sequential execution
        return await self._execute_sequential(context)
    
    async def _execute_skill_with_metadata(
        self, skill_instance: Any, skill_id: str, params: Dict[str, Any], skill_node: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a skill and return result with metadata."""
        try:
            # Set timeout if specified
            timeout = skill_node.get("timeout_seconds", 30)
            
            # Execute with timeout
            skill_result = await asyncio.wait_for(
                skill_instance.execute(**params),
                timeout=timeout
            )
            
            # Return with metadata
            return {{
                "result": skill_result,
                "metadata": {{
                    "skill_id": skill_id,
                    "execution_time": None  # We don't track individual execution time for now
                }}
            }}
        except asyncio.TimeoutError:
            # Handle timeout
            error_msg = f"Skill execution timed out after {{timeout}} seconds"
            
            # Return fallback value if specified
            fallback_value = skill_node.get("fallback_value")
            if fallback_value is not None:
                return {{
                    "result": fallback_value,
                    "error": error_msg,
                    "metadata": {{
                        "skill_id": skill_id,
                        "execution_time": timeout,
                        "fallback_used": True
                    }}
                }}
            
            return {{
                "error": error_msg,
                "metadata": {{
                    "skill_id": skill_id,
                    "execution_time": timeout
                }}
            }}
        except Exception as e:
            # Handle other errors
            error_msg = f"Error executing skill {{skill_id}}: {{str(e)}}"
            
            # Return fallback value if specified
            fallback_value = skill_node.get("fallback_value")
            if fallback_value is not None:
                return {{
                    "result": fallback_value,
                    "error": error_msg,
                    "metadata": {{
                        "skill_id": skill_id,
                        "fallback_used": True
                    }}
                }}
            
            return {{
                "error": error_msg,
                "metadata": {{
                    "skill_id": skill_id
                }}
            }}
    
    def _prepare_skill_input(self, skill_node: Dict[str, Any], 
                           current_input: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input parameters for a skill."""
        # Start with parameters defined in the skill node
        params = copy.deepcopy(skill_node.get("parameters", {{}}))
        
        # Apply input mapping
        input_mapping = skill_node.get("input_mapping", {{}})
        
        if input_mapping:
            # Map specific input fields
            for target_key, source_key in input_mapping.items():
                # Handle nested keys with dot notation
                if "." in source_key:
                    value = current_input
                    for part in source_key.split("."):
                        if part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    
                    if value is not None:
                        params[target_key] = value
                else:
                    if source_key in current_input:
                        params[target_key] = current_input[source_key]
        else:
            # No mapping, pass all current input
            params.update(current_input)
        
        # Add context variables if specified
        if skill_node.get("metadata", {{}}).get("include_context", False):
            params["_context"] = {{
                "execution_id": context["execution_id"],
                "execution_path": context["execution_path"],
                "intermediate_results": context.get("intermediate_results", {{}})
            }}
            
            # Add iteration counter for iterative compositions
            if "iteration" in context:
                params["_context"]["iteration"] = context["iteration"]
        
        return params
    
    def _apply_output_mapping(self, skill_node: Dict[str, Any], 
                             skill_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply output mapping to skill result."""
        output_mapping = skill_node.get("output_mapping", {{}})
        
        if not output_mapping:
            # No mapping, return result as is
            return skill_result
        
        mapped_result = {{}}
        
        for source_key, target_key in output_mapping.items():
            # Handle nested keys with dot notation in source
            if "." in source_key:
                value = skill_result
                for part in source_key.split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                
                if value is not None:
                    mapped_result[target_key] = value
            else:
                if source_key in skill_result:
                    mapped_result[target_key] = skill_result[source_key]
        
        return mapped_result
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression."""
        try:
            # If we have a logic engine, use it
            if self.logic_engine:
                # Prepare variables for evaluation
                variables = {{
                    "input": context["input"],
                    "output": context["output"],
                    "intermediate_results": context["intermediate_results"]
                }}
                
                # Add iteration for iterative compositions
                if "iteration" in context:
                    variables["iteration"] = context["iteration"]
                
                # Evaluate using logic engine
                return self.logic_engine.evaluate_expression(condition, variables)
            
            # Simple fallback evaluation
            # For security, we use a limited evaluation approach
            # This is a simplified version - in production, use proper secure evaluation
            
            # Create a safe evaluation context
            eval_globals = {{"__builtins__": {{}}}}
            eval_locals = {{
                "input": context["input"],
                "output": context["output"],
                "intermediate_results": context["intermediate_results"]
            }}
            
            # Add iteration for iterative compositions
            if "iteration" in context:
                eval_locals["iteration"] = context["iteration"]
                
                # Add common iteration conditions as helpers
                eval_locals["is_first_iteration"] = (context["iteration"] == 0)
                eval_locals["is_last_iteration"] = False  # We don't know this in advance
            
            # Handle common condition patterns
            if condition.startswith("stop"):
                # Extract the actual condition after "stop"
                actual_condition = condition[4:].strip()
                if actual_condition.startswith("if") or actual_condition.startswith("when"):
                    actual_condition = actual_condition[2:].strip()
                
                return eval(actual_condition, eval_globals, eval_locals)
            
            # Regular condition
            return eval(condition, eval_globals, eval_locals)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error evaluating condition '{{condition}}': {{str(e)}}")
            return False
    
    def _apply_transform(self, transform: str, data: Any, context: Dict[str, Any]) -> Any:
        """Apply a transformation to data."""
        try:
            # Simple transformation using eval (in production, use a more secure method)
            eval_globals = {{"__builtins__": {{}}}}
            eval_locals = {{
                "data": data,
                "context": context
            }}
            
            # Recognize common transform patterns
            if transform.startswith("filter"):
                # Filter pattern: filter(data, condition)
                # Example: "filter(data, x['score'] > 0.5)"
                filter_condition = transform[transform.find("(")+1:transform.rfind(")")]
                
                # Split into data and condition parts
                parts = filter_condition.split(",", 1)
                if len(parts) == 2:
                    data_var, condition = parts
                    data_var = data_var.strip()
                    condition = condition.strip()
                    
                    # Apply filter
                    if isinstance(data, list):
                        # Create a lambda for filtering
                        filter_code = f"lambda x: {condition}"
                        filter_func = eval(filter_code, eval_globals, eval_locals)
                        return list(filter(filter_func, data))
                    else:
                        return data
            
            elif transform.startswith("map"):
                # Map pattern: map(data, expression)
                # Example: "map(data, x['value'] * 2)"
                map_expression = transform[transform.find("(")+1:transform.rfind(")")]
                
                # Split into data and expression parts
                parts = map_expression.split(",", 1)
                if len(parts) == 2:
                    data_var, expression = parts
                    data_var = data_var.strip()
                    expression = expression.strip()
                    
                    # Apply map
                    if isinstance(data, list):
                        # Create a lambda for mapping
                        map_code = f"lambda x: {expression}"
                        map_func = eval(map_code, eval_globals, eval_locals)
                        return list(map(map_func, data))
                    else:
                        return data
            
            else:
                # Generic transform - evaluate as expression
                eval_locals["data"] = data
                return eval(transform, eval_globals, eval_locals)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error applying transform '{{transform}}': {{str(e)}}")
            return data
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge dictionaries, modifying target in-place."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_merge(target[key], value)
            else:
                # Overwrite or add key
                target[key] = value
'''

        return code

    async def _register_composite_skill(
        self, definition: CompositeSkillDefinition, skill_path: Path
    ) -> None:
        """
        Register the composite skill with the skill registry.

        Args:
            definition: Composite skill definition
            skill_path: Path to the generated skill file
        """
        # Determine module path
        module_path = str(skill_path).replace(os.sep, ".").replace(".py", "")
        if module_path.startswith("src."):
            module_path = module_path[4:]  # Remove "src." prefix

        self.logger.info(f"Registering composite skill {definition.name} from module {module_path}")

        # Register with skill registry
        self.skill_registry.register_skill(
            name=definition.name,
            module_path=module_path,
            path=str(skill_path),
            version=definition.version,
            category=definition.category,
        )

        # Emit configuration changed event
        await self.event_bus.emit(
            SystemConfigurationChanged(
                component="skills",
                change_type="skill_composed",
                details={
                    "skill_name": definition.name,
                    "skill_version": definition.version,
                    "skill_category": definition.category,
                    "composition_type": definition.composition_type.value,
                    "component_skills": [skill.skill_id for skill in definition.skills],
                },
            )
        )

    async def _extract_composite_definition(self, skill_path: Path) -> Dict[str, Any]:
        """
        Extract composite skill definition from a skill file.

        Args:
            skill_path: Path to the skill file

        Returns:
            Extracted definition as a dictionary
        """
        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for composite_skill_definition in the code
        definition_match = re.search(
            r"self\.composite_skill_definition\s*=\s*(\{.*?\})", content, re.DOTALL
        )
        if not definition_match:
            # Look for it assigned as a multi-line json
            definition_match = re.search(
                r'self\.composite_skill_definition\s*=\s*"""(\{.*?\})"""', content, re.DOTALL
            )

        if not definition_match:
            # Return minimal definition
            return {
                "name": Path(skill_path).stem,
                "description": "",
                "composition_type": "unknown",
                "component_skills": [],
            }

        # Parse the definition JSON
        definition_json = definition_match.group(1)
        try:
            definition = json.loads(definition_json)
            return definition
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse composite skill definition in {skill_path}")
            return {
                "name": Path(skill_path).stem,
                "description": "",
                "composition_type": "unknown",
                "component_skills": [],
            }

    async def _build_dependency_graph(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a dependency graph for a composite skill.

        Args:
            definition: Skill definition

        Returns:
            Graph representation
        """
        graph = nx.DiGraph()

        # Add nodes for each skill
        skills = definition.get("skills", [])
        for skill in skills:
            skill_id = skill.get("skill_id", "unknown")
            graph.add_node(skill_id, **skill)

        # Add edges based on composition type
        composition_type = definition.get("composition_type")

        if composition_type == CompositionType.SEQUENTIAL.value:
            # Sequential: each skill depends on the previous one
            for i in range(1, len(skills)):
                prev_skill = skills[i - 1].get("skill_id")
                curr_skill = skills[i].get("skill_id")
                graph.add_edge(prev_skill, curr_skill)

        elif composition_type == CompositionType.CONDITIONAL.value:
            # No explicit dependencies, could add based on condition analysis
            pass

        elif composition_type == CompositionType.ITERATIVE.value:
            # Potential cycle for iterative, mark special
            if len(skills) > 1:
                first_skill = skills[0].get("skill_id")
                last_skill = skills[-1].get("skill_id")
                graph.add_edge(last_skill, first_skill, is_cycle=True)

        # Check for cycles
        has_cycle = False
        try:
            nx.find_cycle(graph)
            has_cycle = True
        except nx.NetworkXNoCycle:
            has_cycle = False

        # Convert to serializable format
        nodes = [{"id": node, **graph.nodes[node]} for node in graph.nodes]
        edges = [{"source": u, "target": v, **graph.edges[u, v]} for u, v in graph.edges]

        return {
            "nodes": nodes,
            "edges": edges,
            "has_cycle": has_cycle,
            "composition_type": composition_type,
        }


class SkillComposerFactory:
    """Factory for creating SkillComposer instances."""

    @staticmethod
    def create(container: Container) -> SkillComposer:
        """
        Create a SkillComposer instance.

        Args:
            container: Dependency injection container

        Returns:
            SkillComposer instance
        """
        return SkillComposer(container)
