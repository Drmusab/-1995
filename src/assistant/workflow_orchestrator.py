"""
Workflow Orchestrator for AI Assistant System

This module manages and orchestrates complex workflows and task execution.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import uuid
import logging

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    WorkflowStarted,
    WorkflowCompleted,
    WorkflowStepStarted,
    WorkflowStepCompleted
)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a workflow step."""
    step_id: str
    name: str
    step_type: str  # "skill", "api_call", "condition", "loop", etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkflowDefinition:
    """Defines a workflow."""
    workflow_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """
    Orchestrates workflow execution with support for complex control flows,
    error handling, and monitoring.
    """

    def __init__(self, container: Container):
        """Initialize the workflow orchestrator."""
        self.container = container
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.active_executions: Set[str] = set()
        self.event_bus = container.get(EventBus) if container else None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the workflow orchestrator."""
        self.logger.info("Initializing Workflow Orchestrator")
        
        # Register built-in workflows
        await self._register_builtin_workflows()
        
        self.logger.info("Workflow Orchestrator initialized successfully")

    async def _register_builtin_workflows(self) -> None:
        """Register built-in workflows."""
        # Simple greeting workflow
        greeting_workflow = WorkflowDefinition(
            workflow_id="greeting_workflow",
            name="Greeting Workflow",
            description="Simple greeting and introduction workflow",
            steps=[
                WorkflowStep(
                    step_id="greet_user",
                    name="Greet User",
                    step_type="skill",
                    parameters={"skill_name": "greeting_skill"}
                ),
                WorkflowStep(
                    step_id="get_user_info",
                    name="Get User Information",
                    step_type="skill",
                    parameters={"skill_name": "user_info_skill"},
                    dependencies=["greet_user"]
                )
            ]
        )
        
        # Task management workflow
        task_workflow = WorkflowDefinition(
            workflow_id="task_management",
            name="Task Management Workflow",
            description="Workflow for managing user tasks and reminders",
            steps=[
                WorkflowStep(
                    step_id="parse_task",
                    name="Parse Task Request",
                    step_type="nlp",
                    parameters={"parser_type": "task_parser"}
                ),
                WorkflowStep(
                    step_id="create_task",
                    name="Create Task",
                    step_type="skill",
                    parameters={"skill_name": "task_creator"},
                    dependencies=["parse_task"]
                ),
                WorkflowStep(
                    step_id="schedule_reminder",
                    name="Schedule Reminder",
                    step_type="skill",
                    parameters={"skill_name": "reminder_scheduler"},
                    dependencies=["create_task"]
                )
            ]
        )
        
        self.workflows["greeting_workflow"] = greeting_workflow
        self.workflows["task_management"] = task_workflow
        
        self.logger.info(f"Registered {len(self.workflows)} built-in workflows")

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Execute a workflow.
        
        Args:
            workflow_id: Workflow identifier
            input_data: Input data for the workflow
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            Execution identifier
        """
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            session_id=session_id,
            user_id=user_id,
            input_data=input_data,
            started_at=datetime.now(timezone.utc)
        )
        
        self.executions[execution_id] = execution
        self.active_executions.add(execution_id)
        
        if self.event_bus:
            await self.event_bus.emit(
                WorkflowStarted(
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    user_id=user_id
                )
            )
        
        # Start execution in background
        asyncio.create_task(self._execute_workflow(execution_id))
        
        self.logger.info(f"Started workflow execution {execution_id}")
        return execution_id

    async def _execute_workflow(self, execution_id: str) -> None:
        """Execute a workflow instance."""
        execution = self.executions.get(execution_id)
        workflow = self.workflows.get(execution.workflow_id) if execution else None
        
        if not execution or not workflow:
            return
        
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Execute steps in dependency order
            for step in workflow.steps:
                if execution.status != WorkflowStatus.RUNNING:
                    break
                
                await self._execute_step(execution_id, step)
            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            
            if self.event_bus:
                await self.event_bus.emit(
                    WorkflowCompleted(
                        workflow_id=workflow.workflow_id,
                        execution_id=execution_id,
                        success=True,
                        duration=(execution.completed_at - execution.started_at).total_seconds()
                    )
                )
            
            self.logger.info(f"Workflow execution {execution_id} completed successfully")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            
            self.logger.error(f"Workflow execution {execution_id} failed: {e}")
            
        finally:
            self.active_executions.discard(execution_id)

    async def _execute_step(self, execution_id: str, step: WorkflowStep) -> None:
        """Execute a workflow step."""
        execution = self.executions[execution_id]
        
        if self.event_bus:
            await self.event_bus.emit(
                WorkflowStepStarted(
                    execution_id=execution_id,
                    step_id=step.step_id,
                    step_name=step.name
                )
            )
        
        try:
            # Check dependencies
            for dep_step_id in step.dependencies:
                if dep_step_id not in execution.step_results:
                    raise RuntimeError(f"Dependency {dep_step_id} not satisfied")
            
            # Execute step based on type
            result = await self._execute_step_by_type(step, execution)
            
            # Store result
            execution.step_results[step.step_id] = result
            
            if self.event_bus:
                await self.event_bus.emit(
                    WorkflowStepCompleted(
                        execution_id=execution_id,
                        step_id=step.step_id,
                        step_name=step.name,
                        success=True
                    )
                )
            
            self.logger.debug(f"Step {step.step_id} completed successfully")
            
        except Exception as e:
            execution.step_results[step.step_id] = {"error": str(e)}
            self.logger.error(f"Step {step.step_id} failed: {e}")
            raise

    async def _execute_step_by_type(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a step based on its type."""
        # Stub implementation for different step types
        if step.step_type == "skill":
            return await self._execute_skill_step(step, execution)
        elif step.step_type == "nlp":
            return await self._execute_nlp_step(step, execution)
        elif step.step_type == "api_call":
            return await self._execute_api_step(step, execution)
        else:
            return {"result": f"Executed {step.step_type} step", "success": True}

    async def _execute_skill_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute a skill step."""
        # Stub implementation
        skill_name = step.parameters.get("skill_name", "unknown")
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "skill": skill_name,
            "result": f"Executed skill {skill_name}",
            "success": True
        }

    async def _execute_nlp_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute an NLP step."""
        # Stub implementation
        await asyncio.sleep(0.1)  # Simulate processing
        return {
            "parsed_intent": "task_creation",
            "entities": {"task": "example task"},
            "success": True
        }

    async def _execute_api_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute an API call step."""
        # Stub implementation
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "api_response": {"status": "success"},
            "success": True
        }

    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        execution = self.executions.get(execution_id)
        if not execution:
            return {"error": "Execution not found"}
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "step_results": execution.step_results,
            "output_data": execution.output_data,
            "error_message": execution.error_message
        }

    def list_workflows(self) -> List[str]:
        """List available workflows."""
        return list(self.workflows.keys())

    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return list(self.active_executions)

    async def shutdown_all(self) -> None:
        """Shutdown the workflow orchestrator."""
        self.logger.info("Shutting down Workflow Orchestrator")
        
        # Cancel active executions
        for execution_id in list(self.active_executions):
            execution = self.executions.get(execution_id)
            if execution:
                execution.status = WorkflowStatus.CANCELLED
        
        self.active_executions.clear()
        
        self.logger.info("Workflow Orchestrator shutdown complete")