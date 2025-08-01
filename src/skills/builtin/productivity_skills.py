"""
Advanced Productivity Skills for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:28:17 UTC

This module provides comprehensive productivity-focused skills for the AI assistant,
including task management, note taking, reminders, time tracking, file organization,
and calendar integration with seamless memory persistence and personalization.
"""

import datetime
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple, Union

import asyncio

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CalendarEventCreated,
    CalendarEventUpdated,
    NoteCreated,
    NoteUpdated,
    ReminderCreated,
    ReminderTriggered,
    SearchPerformed,
    SkillExecutionCompleted,
    SkillExecutionFailed,
    SkillExecutionStarted,
    TaskCompleted,
    TaskCreated,
    TaskUpdated,
)
from src.integrations.external_apis.calendar_api import CalendarAPI
from src.integrations.external_apis.web_search import WebSearchAPI

# Integrations
from src.integrations.storage.database import DatabaseManager

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.storage.vector_store import VectorStore
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Skills management
from src.skills.skill_registry import SkillRegistry


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TaskStatus(Enum):
    """Task status values."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class ReminderRecurrence(Enum):
    """Reminder recurrence patterns."""

    NONE = "none"
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class Task:
    """Task data structure."""

    task_id: str
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.NOT_STARTED
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    project: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None
    estimated_time_minutes: Optional[int] = None
    actual_time_minutes: Optional[int] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Note:
    """Note data structure."""

    note_id: str
    title: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    category: Optional[str] = None
    is_pinned: bool = False
    is_archived: bool = False
    references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Reminder:
    """Reminder data structure."""

    reminder_id: str
    title: str
    description: Optional[str] = None
    due_time: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_completed: bool = False
    recurrence: ReminderRecurrence = ReminderRecurrence.NONE
    recurrence_pattern: Optional[str] = None
    user_id: Optional[str] = None
    notify_before_minutes: int = 15
    tags: List[str] = field(default_factory=list)
    linked_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalendarEvent:
    """Calendar event data structure."""

    event_id: str
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    location: Optional[str] = None
    is_all_day: bool = False
    recurrence: Optional[str] = None
    attendees: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    calendar_id: Optional[str] = None
    reminders: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProductivityError(Exception):
    """Custom exception for productivity skills."""

    def __init__(
        self,
        message: str,
        skill_name: Optional[str] = None,
        error_code: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.skill_name = skill_name
        self.error_code = error_code
        self.user_id = user_id
        self.timestamp = datetime.now(timezone.utc)


class BaseProductivitySkill:
    """Base class for all productivity skills."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        context_manager: ContextManager,
        event_bus: EventBus,
        config: ConfigLoader,
        database: DatabaseManager,
        metrics: MetricsCollector,
        tracer: TraceManager,
    ):
        self.memory_manager = memory_manager
        self.context_manager = context_manager
        self.event_bus = event_bus
        self.config = config
        self.database = database
        self.metrics = metrics
        self.tracer = tracer
        self.logger = get_logger(__name__)

        # Skill metadata
        self.skill_id = f"{self.__class__.__name__.lower()}"
        self.version = "1.0.0"
        self.description = "Base productivity skill"

        # Initialize metrics
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup skill-specific metrics."""
        try:
            self.metrics.register_counter(f"skill_{self.skill_id}_executions_total")
            self.metrics.register_counter(f"skill_{self.skill_id}_errors_total")
            self.metrics.register_histogram(f"skill_{self.skill_id}_execution_time_seconds")
        except Exception as e:
            self.logger.warning(f"Failed to setup metrics for {self.skill_id}: {str(e)}")

    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the skill with standard logging and metrics."""
        start_time = time.time()
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")

        # Emit execution started event
        await self.event_bus.emit(
            SkillExecutionStarted(skill_id=self.skill_id, user_id=user_id, session_id=session_id)
        )

        try:
            with self.tracer.trace(f"skill_{self.skill_id}_execution") as span:
                if user_id:
                    span.set_attribute("user_id", user_id)
                if session_id:
                    span.set_attribute("session_id", session_id)

                # Execute skill implementation
                result = await self._execute(**kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Emit execution completed event
                await self.event_bus.emit(
                    SkillExecutionCompleted(
                        skill_id=self.skill_id,
                        execution_time=execution_time,
                        user_id=user_id,
                        session_id=session_id,
                    )
                )

                # Update metrics
                self.metrics.increment(f"skill_{self.skill_id}_executions_total")
                self.metrics.record(f"skill_{self.skill_id}_execution_time_seconds", execution_time)

                return {
                    "success": True,
                    "skill_id": self.skill_id,
                    "execution_time": execution_time,
                    **result,
                }

        except Exception as e:
            # Calculate execution time even for failures
            execution_time = time.time() - start_time

            # Emit execution failed event
            await self.event_bus.emit(
                SkillExecutionFailed(
                    skill_id=self.skill_id,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    user_id=user_id,
                    session_id=session_id,
                )
            )

            # Update error metrics
            self.metrics.increment(f"skill_{self.skill_id}_errors_total")

            self.logger.error(f"Error executing skill {self.skill_id}: {str(e)}")
            raise ProductivityError(f"Skill execution failed: {str(e)}", self.skill_id) from e

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Skill-specific implementation to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute method")

    def get_metadata(self) -> Dict[str, Any]:
        """Get skill metadata."""
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "description": self.description,
            "category": "productivity",
        }


class TaskManagementSkill(BaseProductivitySkill):
    """Task management skill for creating, updating, and tracking tasks."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        context_manager: ContextManager,
        event_bus: EventBus,
        config: ConfigLoader,
        database: DatabaseManager,
        metrics: MetricsCollector,
        tracer: TraceManager,
        vector_store: VectorStore,
    ):
        super().__init__(
            memory_manager, context_manager, event_bus, config, database, metrics, tracer
        )
        self.vector_store = vector_store
        self.description = "Create, manage, and organize tasks and projects"

        # Task storage
        self._initialize_storage()

        # Task-specific metrics
        self.metrics.register_counter("tasks_created_total")
        self.metrics.register_counter("tasks_completed_total")
        self.metrics.register_gauge("active_tasks")

    def _initialize_storage(self) -> None:
        """Initialize task storage."""
        try:
            # Create tasks table if it doesn't exist
            self.database.execute_sync(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    due_date TIMESTAMP,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    tags TEXT,
                    user_id TEXT,
                    project TEXT,
                    subtasks TEXT,
                    parent_task TEXT,
                    estimated_time INTEGER,
                    actual_time INTEGER,
                    notes TEXT,
                    metadata TEXT
                )
            """
            )

            # Create index on user_id for faster queries
            self.database.execute_sync(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id)
            """
            )

            # Create index on status for filtering
            self.database.execute_sync(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
            """
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize task storage: {str(e)}")

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute task management operations."""
        operation = kwargs.get("operation", "list")
        user_id = kwargs.get("user_id")

        # Route to appropriate operation
        if operation == "create":
            return await self._create_task(**kwargs)
        elif operation == "update":
            return await self._update_task(**kwargs)
        elif operation == "complete":
            return await self._complete_task(**kwargs)
        elif operation == "delete":
            return await self._delete_task(**kwargs)
        elif operation == "get":
            return await self._get_task(**kwargs)
        elif operation == "list":
            return await self._list_tasks(**kwargs)
        elif operation == "search":
            return await self._search_tasks(**kwargs)
        else:
            raise ProductivityError(f"Unknown operation: {operation}", self.skill_id)

    async def _create_task(self, **kwargs) -> Dict[str, Any]:
        """Create a new task."""
        user_id = kwargs.get("user_id")
        title = kwargs.get("title")
        description = kwargs.get("description")
        priority = kwargs.get("priority", "medium")
        due_date_str = kwargs.get("due_date")
        tags = kwargs.get("tags", [])
        project = kwargs.get("project")

        if not title:
            raise ProductivityError("Task title is required", self.skill_id)

        # Parse due date if provided
        due_date = None
        if due_date_str:
            try:
                due_date = datetime.fromisoformat(due_date_str)
            except ValueError:
                # Try natural language parsing
                try:
                    # This would use a date parsing utility in a real implementation
                    due_date = self._parse_date_expression(due_date_str)
                except Exception as e:
                    raise ProductivityError(f"Invalid due date format: {str(e)}", self.skill_id)

        # Create task
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            status=TaskStatus.NOT_STARTED,
            priority=TaskPriority(priority.lower()) if priority else TaskPriority.MEDIUM,
            due_date=due_date,
            created_at=now,
            updated_at=now,
            tags=tags,
            user_id=user_id,
            project=project,
        )

        # Store task in database
        await self._store_task(task)

        # Store in vector database for semantic search
        if self.vector_store:
            task_text = f"{task.title} {task.description or ''} {' '.join(task.tags)}"
            await self.vector_store.store(
                task_id,
                task_text,
                metadata={
                    "type": "task",
                    "user_id": user_id,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                },
            )

        # Emit task created event
        await self.event_bus.emit(
            TaskCreated(
                task_id=task_id,
                user_id=user_id,
                title=title,
                due_date=due_date.isoformat() if due_date else None,
            )
        )

        # Update metrics
        self.metrics.increment("tasks_created_total")

        # Update task count for user
        active_tasks = await self._count_active_tasks(user_id)
        self.metrics.set("active_tasks", active_tasks, tags={"user_id": user_id})

        return {
            "task_id": task_id,
            "message": f"Task '{title}' created successfully",
            "task": asdict(task),
        }

    async def _update_task(self, **kwargs) -> Dict[str, Any]:
        """Update an existing task."""
        task_id = kwargs.get("task_id")
        user_id = kwargs.get("user_id")

        if not task_id:
            raise ProductivityError("Task ID is required for updates", self.skill_id)

        # Get existing task
        task = await self._get_task_by_id(task_id)
        if not task:
            raise ProductivityError(f"Task not found: {task_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and task.user_id and task.user_id != user_id:
            raise ProductivityError("Not authorized to update this task", self.skill_id)

        # Update fields
        if "title" in kwargs:
            task.title = kwargs["title"]
        if "description" in kwargs:
            task.description = kwargs["description"]
        if "status" in kwargs:
            task.status = TaskStatus(kwargs["status"])
        if "priority" in kwargs:
            task.priority = TaskPriority(kwargs["priority"])
        if "due_date" in kwargs:
            due_date_str = kwargs["due_date"]
            if due_date_str:
                try:
                    task.due_date = datetime.fromisoformat(due_date_str)
                except ValueError:
                    task.due_date = self._parse_date_expression(due_date_str)
            else:
                task.due_date = None
        if "tags" in kwargs:
            task.tags = kwargs["tags"]
        if "project" in kwargs:
            task.project = kwargs["project"]
        if "notes" in kwargs:
            task.notes = kwargs["notes"]

        # Update timestamp
        task.updated_at = datetime.now(timezone.utc)

        # Store updated task
        await self._store_task(task)

        # Update vector store
        if self.vector_store:
            task_text = f"{task.title} {task.description or ''} {' '.join(task.tags)}"
            await self.vector_store.update(
                task_id,
                task_text,
                metadata={
                    "type": "task",
                    "user_id": task.user_id,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                },
            )

        # Emit task updated event
        await self.event_bus.emit(
            TaskUpdated(task_id=task_id, user_id=user_id, status=task.status.value)
        )

        return {
            "task_id": task_id,
            "message": f"Task '{task.title}' updated successfully",
            "task": asdict(task),
        }

    async def _complete_task(self, **kwargs) -> Dict[str, Any]:
        """Mark a task as completed."""
        task_id = kwargs.get("task_id")
        user_id = kwargs.get("user_id")

        if not task_id:
            raise ProductivityError("Task ID is required", self.skill_id)

        # Get existing task
        task = await self._get_task_by_id(task_id)
        if not task:
            raise ProductivityError(f"Task not found: {task_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and task.user_id and task.user_id != user_id:
            raise ProductivityError("Not authorized to complete this task", self.skill_id)

        # Mark as completed
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.updated_at = datetime.now(timezone.utc)

        # Store updated task
        await self._store_task(task)

        # Update vector store
        if self.vector_store:
            await self.vector_store.update(
                task_id,
                metadata={
                    "status": task.status.value,
                    "completed_at": task.completed_at.isoformat(),
                },
            )

        # Emit task completed event
        await self.event_bus.emit(
            TaskCompleted(
                task_id=task_id,
                user_id=user_id,
                title=task.title,
                completion_time=task.completed_at.isoformat(),
            )
        )

        # Update metrics
        self.metrics.increment("tasks_completed_total")

        # Update task count for user
        active_tasks = await self._count_active_tasks(user_id)
        self.metrics.set("active_tasks", active_tasks, tags={"user_id": user_id})

        return {
            "task_id": task_id,
            "message": f"Task '{task.title}' marked as completed",
            "completed_at": task.completed_at.isoformat(),
        }

    async def _delete_task(self, **kwargs) -> Dict[str, Any]:
        """Delete a task."""
        task_id = kwargs.get("task_id")
        user_id = kwargs.get("user_id")

        if not task_id:
            raise ProductivityError("Task ID is required", self.skill_id)

        # Get existing task
        task = await self._get_task_by_id(task_id)
        if not task:
            raise ProductivityError(f"Task not found: {task_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and task.user_id and task.user_id != user_id:
            raise ProductivityError("Not authorized to delete this task", self.skill_id)

        # Delete from database
        await self.database.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

        # Delete from vector store
        if self.vector_store:
            await self.vector_store.delete(task_id)

        # Update metrics
        if task.status != TaskStatus.COMPLETED:
            active_tasks = await self._count_active_tasks(user_id)
            self.metrics.set("active_tasks", active_tasks, tags={"user_id": user_id})

        return {"task_id": task_id, "message": f"Task '{task.title}' deleted successfully"}

    async def _get_task(self, **kwargs) -> Dict[str, Any]:
        """Get a specific task."""
        task_id = kwargs.get("task_id")
        user_id = kwargs.get("user_id")

        if not task_id:
            raise ProductivityError("Task ID is required", self.skill_id)

        task = await self._get_task_by_id(task_id)
        if not task:
            raise ProductivityError(f"Task not found: {task_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and task.user_id and task.user_id != user_id:
            raise ProductivityError("Not authorized to view this task", self.skill_id)

        return {"task": asdict(task)}

    async def _list_tasks(self, **kwargs) -> Dict[str, Any]:
        """List tasks with filtering."""
        user_id = kwargs.get("user_id")
        status = kwargs.get("status")
        priority = kwargs.get("priority")
        project = kwargs.get("project")
        tag = kwargs.get("tag")
        due_date_before = kwargs.get("due_date_before")
        due_date_after = kwargs.get("due_date_after")
        limit = kwargs.get("limit", 50)
        offset = kwargs.get("offset", 0)

        # Build query
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if status:
            query += " AND status = ?"
            params.append(status)

        if priority:
            query += " AND priority = ?"
            params.append(priority)

        if project:
            query += " AND project = ?"
            params.append(project)

        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")

        if due_date_before:
            query += " AND due_date <= ?"
            params.append(due_date_before)

        if due_date_after:
            query += " AND due_date >= ?"
            params.append(due_date_after)

        # Add ordering and pagination
        query += " ORDER BY due_date ASC, priority ASC, created_at DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute query
        results = await self.database.fetch_all(query, tuple(params))

        # Convert to Task objects
        tasks = []
        for row in results:
            task = self._row_to_task(row)
            tasks.append(asdict(task))

        # Get total count for pagination
        count_query = query.split(" ORDER BY")[0].replace("SELECT *", "SELECT COUNT(*)")
        count_params = params[:-2]  # Remove limit and offset
        count_result = await self.database.fetch_one(count_query, tuple(count_params))
        total_count = count_result[0] if count_result else 0

        return {"tasks": tasks, "total_count": total_count, "limit": limit, "offset": offset}

    async def _search_tasks(self, **kwargs) -> Dict[str, Any]:
        """Search tasks using semantic search."""
        query = kwargs.get("query")
        user_id = kwargs.get("user_id")
        limit = kwargs.get("limit", 10)

        if not query:
            raise ProductivityError("Search query is required", self.skill_id)

        if not self.vector_store:
            raise ProductivityError("Vector store not available for semantic search", self.skill_id)

        # Perform semantic search
        search_filter = {"type": "task"}
        if user_id:
            search_filter["user_id"] = user_id

        search_results = await self.vector_store.search(query, filter=search_filter, limit=limit)

        # Fetch full task details
        tasks = []
        for result in search_results:
            task_id = result["id"]
            task = await self._get_task_by_id(task_id)
            if task:
                task_dict = asdict(task)
                task_dict["relevance_score"] = result.get("score", 0)
                tasks.append(task_dict)

        return {"query": query, "tasks": tasks, "result_count": len(tasks)}

    async def _store_task(self, task: Task) -> None:
        """Store a task in the database."""
        query = """
            INSERT OR REPLACE INTO tasks (
                task_id, title, description, status, priority, due_date,
                created_at, updated_at, completed_at, tags, user_id, project,
                subtasks, parent_task, estimated_time, actual_time, notes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            task.task_id,
            task.title,
            task.description,
            task.status.value,
            task.priority.value,
            task.due_date.isoformat() if task.due_date else None,
            task.created_at.isoformat(),
            task.updated_at.isoformat(),
            task.completed_at.isoformat() if task.completed_at else None,
            json.dumps(task.tags),
            task.user_id,
            task.project,
            json.dumps(task.subtasks),
            task.parent_task,
            task.estimated_time_minutes,
            task.actual_time_minutes,
            task.notes,
            json.dumps(task.metadata),
        )

        await self.database.execute(query, params)

    async def _get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        result = await self.database.fetch_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))

        if not result:
            return None

        return self._row_to_task(result)

    def _row_to_task(self, row: tuple) -> Task:
        """Convert a database row to a Task object."""
        # Extract fields from row
        (
            task_id,
            title,
            description,
            status,
            priority,
            due_date,
            created_at,
            updated_at,
            completed_at,
            tags,
            user_id,
            project,
            subtasks,
            parent_task,
            estimated_time,
            actual_time,
            notes,
            metadata,
        ) = row

        # Parse dates
        created_at_dt = (
            datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
        )
        updated_at_dt = (
            datetime.fromisoformat(updated_at) if updated_at else datetime.now(timezone.utc)
        )
        completed_at_dt = datetime.fromisoformat(completed_at) if completed_at else None
        due_date_dt = datetime.fromisoformat(due_date) if due_date else None

        # Parse JSON fields
        tags_list = json.loads(tags) if tags else []
        subtasks_list = json.loads(subtasks) if subtasks else []
        metadata_dict = json.loads(metadata) if metadata else {}

        return Task(
            task_id=task_id,
            title=title,
            description=description,
            status=TaskStatus(status),
            priority=TaskPriority(priority),
            due_date=due_date_dt,
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            completed_at=completed_at_dt,
            tags=tags_list,
            user_id=user_id,
            project=project,
            subtasks=subtasks_list,
            parent_task=parent_task,
            estimated_time_minutes=estimated_time,
            actual_time_minutes=actual_time,
            notes=notes,
            metadata=metadata_dict,
        )

    async def _count_active_tasks(self, user_id: Optional[str]) -> int:
        """Count active tasks for a user."""
        if not user_id:
            return 0

        query = """
            SELECT COUNT(*) FROM tasks 
            WHERE user_id = ? AND status != ?
        """

        result = await self.database.fetch_one(query, (user_id, TaskStatus.COMPLETED.value))
        return result[0] if result else 0

    def _parse_date_expression(self, expression: str) -> datetime:
        """Parse natural language date expressions."""
        now = datetime.now(timezone.utc)
        expression = expression.lower().strip()

        # Common patterns
        if expression == "today":
            return datetime.combine(now.date(), datetime.min.time()).replace(tzinfo=timezone.utc)
        elif expression == "tomorrow":
            return datetime.combine(now.date() + timedelta(days=1), datetime.min.time()).replace(
                tzinfo=timezone.utc
            )
        elif expression == "next week":
            return datetime.combine(now.date() + timedelta(days=7), datetime.min.time()).replace(
                tzinfo=timezone.utc
            )
        elif expression == "next month":
            if now.month == 12:
                return datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                return datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)

        # Handle "in X days/weeks/months"
        in_match = re.match(r"in (\d+) (day|days|week|weeks|month|months)", expression)
        if in_match:
            amount = int(in_match.group(1))
            unit = in_match.group(2)

            if unit in ["day", "days"]:
                return now + timedelta(days=amount)
            elif unit in ["week", "weeks"]:
                return now + timedelta(days=amount * 7)
            elif unit in ["month", "months"]:
                # Approximation for months
                return now + timedelta(days=amount * 30)

        # Could not parse
        raise ProductivityError(f"Could not parse date expression: {expression}", self.skill_id)


class NoteManagementSkill(BaseProductivitySkill):
    """Note management skill for creating, organizing, and retrieving notes."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        context_manager: ContextManager,
        event_bus: EventBus,
        config: ConfigLoader,
        database: DatabaseManager,
        metrics: MetricsCollector,
        tracer: TraceManager,
        vector_store: VectorStore,
    ):
        super().__init__(
            memory_manager, context_manager, event_bus, config, database, metrics, tracer
        )
        self.vector_store = vector_store
        self.description = "Create, organize, and retrieve notes and information"

        # Note storage
        self._initialize_storage()

        # Note-specific metrics
        self.metrics.register_counter("notes_created_total")
        self.metrics.register_counter("notes_updated_total")
        self.metrics.register_gauge("total_notes")

    def _initialize_storage(self) -> None:
        """Initialize note storage."""
        try:
            # Create notes table if it doesn't exist
            self.database.execute_sync(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    note_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    tags TEXT,
                    user_id TEXT,
                    category TEXT,
                    is_pinned BOOLEAN,
                    is_archived BOOLEAN,
                    references TEXT,
                    metadata TEXT
                )
            """
            )

            # Create index on user_id for faster queries
            self.database.execute_sync(
                """
                CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes(user_id)
            """
            )

            # Create index on category for filtering
            self.database.execute_sync(
                """
                CREATE INDEX IF NOT EXISTS idx_notes_category ON notes(category)
            """
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize note storage: {str(e)}")

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute note management operations."""
        operation = kwargs.get("operation", "list")
        user_id = kwargs.get("user_id")

        # Route to appropriate operation
        if operation == "create":
            return await self._create_note(**kwargs)
        elif operation == "update":
            return await self._update_note(**kwargs)
        elif operation == "delete":
            return await self._delete_note(**kwargs)
        elif operation == "get":
            return await self._get_note(**kwargs)
        elif operation == "list":
            return await self._list_notes(**kwargs)
        elif operation == "search":
            return await self._search_notes(**kwargs)
        else:
            raise ProductivityError(f"Unknown operation: {operation}", self.skill_id)

    async def _create_note(self, **kwargs) -> Dict[str, Any]:
        """Create a new note."""
        user_id = kwargs.get("user_id")
        title = kwargs.get("title")
        content = kwargs.get("content")
        category = kwargs.get("category")
        tags = kwargs.get("tags", [])
        is_pinned = kwargs.get("is_pinned", False)

        if not title:
            raise ProductivityError("Note title is required", self.skill_id)

        if not content:
            raise ProductivityError("Note content is required", self.skill_id)

        # Create note
        note_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        note = Note(
            note_id=note_id,
            title=title,
            content=content,
            created_at=now,
            updated_at=now,
            tags=tags,
            user_id=user_id,
            category=category,
            is_pinned=is_pinned,
            is_archived=False,
        )

        # Store note in database
        await self._store_note(note)

        # Store in vector database for semantic search
        if self.vector_store:
            note_text = f"{note.title} {note.content} {' '.join(note.tags)}"
            await self.vector_store.store(
                note_id,
                note_text,
                metadata={
                    "type": "note",
                    "user_id": user_id,
                    "category": category,
                    "is_pinned": is_pinned,
                    "is_archived": False,
                },
            )

        # Emit note created event
        await self.event_bus.emit(NoteCreated(note_id=note_id, user_id=user_id, title=title))

        # Update metrics
        self.metrics.increment("notes_created_total")

        # Update note count for user
        total_notes = await self._count_notes(user_id)
        self.metrics.set("total_notes", total_notes, tags={"user_id": user_id})

        return {
            "note_id": note_id,
            "message": f"Note '{title}' created successfully",
            "note": asdict(note),
        }

    async def _update_note(self, **kwargs) -> Dict[str, Any]:
        """Update an existing note."""
        note_id = kwargs.get("note_id")
        user_id = kwargs.get("user_id")

        if not note_id:
            raise ProductivityError("Note ID is required for updates", self.skill_id)

        # Get existing note
        note = await self._get_note_by_id(note_id)
        if not note:
            raise ProductivityError(f"Note not found: {note_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and note.user_id and note.user_id != user_id:
            raise ProductivityError("Not authorized to update this note", self.skill_id)

        # Update fields
        if "title" in kwargs:
            note.title = kwargs["title"]
        if "content" in kwargs:
            note.content = kwargs["content"]
        if "category" in kwargs:
            note.category = kwargs["category"]
        if "tags" in kwargs:
            note.tags = kwargs["tags"]
        if "is_pinned" in kwargs:
            note.is_pinned = kwargs["is_pinned"]
        if "is_archived" in kwargs:
            note.is_archived = kwargs["is_archived"]

        # Update timestamp
        note.updated_at = datetime.now(timezone.utc)

        # Store updated note
        await self._store_note(note)

        # Update vector store
        if self.vector_store:
            note_text = f"{note.title} {note.content} {' '.join(note.tags)}"
            await self.vector_store.update(
                note_id,
                note_text,
                metadata={
                    "type": "note",
                    "user_id": note.user_id,
                    "category": note.category,
                    "is_pinned": note.is_pinned,
                    "is_archived": note.is_archived,
                },
            )

        # Emit note updated event
        await self.event_bus.emit(NoteUpdated(note_id=note_id, user_id=user_id, title=note.title))

        # Update metrics
        self.metrics.increment("notes_updated_total")

        return {
            "note_id": note_id,
            "message": f"Note '{note.title}' updated successfully",
            "note": asdict(note),
        }

    async def _delete_note(self, **kwargs) -> Dict[str, Any]:
        """Delete a note."""
        note_id = kwargs.get("note_id")
        user_id = kwargs.get("user_id")

        if not note_id:
            raise ProductivityError("Note ID is required", self.skill_id)

        # Get existing note
        note = await self._get_note_by_id(note_id)
        if not note:
            raise ProductivityError(f"Note not found: {note_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and note.user_id and note.user_id != user_id:
            raise ProductivityError("Not authorized to delete this note", self.skill_id)

        # Delete from database
        await self.database.execute("DELETE FROM notes WHERE note_id = ?", (note_id,))

        # Delete from vector store
        if self.vector_store:
            await self.vector_store.delete(note_id)

        # Update metrics
        total_notes = await self._count_notes(user_id)
        self.metrics.set("total_notes", total_notes, tags={"user_id": user_id})

        return {"note_id": note_id, "message": f"Note '{note.title}' deleted successfully"}

    async def _get_note(self, **kwargs) -> Dict[str, Any]:
        """Get a specific note."""
        note_id = kwargs.get("note_id")
        user_id = kwargs.get("user_id")

        if not note_id:
            raise ProductivityError("Note ID is required", self.skill_id)

        note = await self._get_note_by_id(note_id)
        if not note:
            raise ProductivityError(f"Note not found: {note_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and note.user_id and note.user_id != user_id:
            raise ProductivityError("Not authorized to view this note", self.skill_id)

        return {"note": asdict(note)}

    async def _list_notes(self, **kwargs) -> Dict[str, Any]:
        """List notes with filtering."""
        user_id = kwargs.get("user_id")
        category = kwargs.get("category")
        tag = kwargs.get("tag")
        is_pinned = kwargs.get("is_pinned")
        is_archived = kwargs.get("is_archived")
        limit = kwargs.get("limit", 50)
        offset = kwargs.get("offset", 0)

        # Build query
        query = "SELECT * FROM notes WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        if category:
            query += " AND category = ?"
            params.append(category)

        if tag:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")

        if is_pinned is not None:
            query += " AND is_pinned = ?"
            params.append(1 if is_pinned else 0)

        if is_archived is not None:
            query += " AND is_archived = ?"
            params.append(1 if is_archived else 0)

        # Add ordering and pagination
        query += " ORDER BY is_pinned DESC, updated_at DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute query
        results = await self.database.fetch_all(query, tuple(params))

        # Convert to Note objects
        notes = []
        for row in results:
            note = self._row_to_note(row)
            notes.append(asdict(note))

        # Get total count for pagination
        count_query = query.split(" ORDER BY")[0].replace("SELECT *", "SELECT COUNT(*)")
        count_params = params[:-2]  # Remove limit and offset
        count_result = await self.database.fetch_one(count_query, tuple(count_params))
        total_count = count_result[0] if count_result else 0

        return {"notes": notes, "total_count": total_count, "limit": limit, "offset": offset}

    async def _search_notes(self, **kwargs) -> Dict[str, Any]:
        """Search notes using semantic search."""
        query = kwargs.get("query")
        user_id = kwargs.get("user_id")
        limit = kwargs.get("limit", 10)

        if not query:
            raise ProductivityError("Search query is required", self.skill_id)

        if not self.vector_store:
            raise ProductivityError("Vector store not available for semantic search", self.skill_id)

        # Perform semantic search
        search_filter = {"type": "note"}
        if user_id:
            search_filter["user_id"] = user_id

        search_results = await self.vector_store.search(query, filter=search_filter, limit=limit)

        # Fetch full note details
        notes = []
        for result in search_results:
            note_id = result["id"]
            note = await self._get_note_by_id(note_id)
            if note:
                note_dict = asdict(note)
                note_dict["relevance_score"] = result.get("score", 0)
                notes.append(note_dict)

        return {"query": query, "notes": notes, "result_count": len(notes)}

    async def _store_note(self, note: Note) -> None:
        """Store a note in the database."""
        query = """
            INSERT OR REPLACE INTO notes (
                note_id, title, content, created_at, updated_at,
                tags, user_id, category, is_pinned, is_archived, references, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        params = (
            note.note_id,
            note.title,
            note.content,
            note.created_at.isoformat(),
            note.updated_at.isoformat(),
            json.dumps(note.tags),
            note.user_id,
            note.category,
            1 if note.is_pinned else 0,
            1 if note.is_archived else 0,
            json.dumps(note.references),
            json.dumps(note.metadata),
        )

        await self.database.execute(query, params)

    async def _get_note_by_id(self, note_id: str) -> Optional[Note]:
        """Get a note by ID."""
        result = await self.database.fetch_one("SELECT * FROM notes WHERE note_id = ?", (note_id,))

        if not result:
            return None

        return self._row_to_note(result)

    def _row_to_note(self, row: tuple) -> Note:
        """Convert a database row to a Note object."""
        # Extract fields from row
        (
            note_id,
            title,
            content,
            created_at,
            updated_at,
            tags,
            user_id,
            category,
            is_pinned,
            is_archived,
            references,
            metadata,
        ) = row

        # Parse dates
        created_at_dt = (
            datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
        )
        updated_at_dt = (
            datetime.fromisoformat(updated_at) if updated_at else datetime.now(timezone.utc)
        )

        # Parse JSON fields
        tags_list = json.loads(tags) if tags else []
        references_list = json.loads(references) if references else []
        metadata_dict = json.loads(metadata) if metadata else {}

        return Note(
            note_id=note_id,
            title=title,
            content=content,
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            tags=tags_list,
            user_id=user_id,
            category=category,
            is_pinned=bool(is_pinned),
            is_archived=bool(is_archived),
            references=references_list,
            metadata=metadata_dict,
        )

    async def _count_notes(self, user_id: Optional[str]) -> int:
        """Count notes for a user."""
        if not user_id:
            return 0

        query = "SELECT COUNT(*) FROM notes WHERE user_id = ?"
        result = await self.database.fetch_one(query, (user_id,))
        return result[0] if result else 0


class ReminderManagementSkill(BaseProductivitySkill):
    """Reminder management skill for creating and managing time-based reminders."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        context_manager: ContextManager,
        event_bus: EventBus,
        config: ConfigLoader,
        database: DatabaseManager,
        metrics: MetricsCollector,
        tracer: TraceManager,
        calendar_api: Optional[CalendarAPI] = None,
    ):
        super().__init__(
            memory_manager, context_manager, event_bus, config, database, metrics, tracer
        )
        self.calendar_api = calendar_api
        self.description = "Create and manage time-based reminders and alerts"

        # Reminder storage
        self._initialize_storage()

        # Reminder-specific metrics
        self.metrics.register_counter("reminders_created_total")
        self.metrics.register_counter("reminders_triggered_total")
        self.metrics.register_gauge("active_reminders")

        # Start reminder checker background task
        self.reminder_task = asyncio.create_task(self._reminder_check_loop())

    def _initialize_storage(self) -> None:
        """Initialize reminder storage."""
        try:
            # Create reminders table if it doesn't exist
            self.database.execute_sync(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    due_time TIMESTAMP NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    is_completed BOOLEAN,
                    recurrence TEXT,
                    recurrence_pattern TEXT,
                    user_id TEXT,
                    notify_before_minutes INTEGER,
                    tags TEXT,
                    linked_tasks TEXT,
                    metadata TEXT
                )
            """
            )

            # Create index on user_id for faster queries
            self.database.execute_sync(
                """
                CREATE INDEX IF NOT EXISTS idx_reminders_user_id ON reminders(user_id)
            """
            )

            # Create index on due_time for faster checking
            self.database.execute_sync(
                """
                CREATE INDEX IF NOT EXISTS idx_reminders_due_time ON reminders(due_time)
            """
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize reminder storage: {str(e)}")

    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """Execute reminder management operations."""
        operation = kwargs.get("operation", "list")
        user_id = kwargs.get("user_id")

        # Route to appropriate operation
        if operation == "create":
            return await self._create_reminder(**kwargs)
        elif operation == "update":
            return await self._update_reminder(**kwargs)
        elif operation == "complete":
            return await self._complete_reminder(**kwargs)
        elif operation == "delete":
            return await self._delete_reminder(**kwargs)
        elif operation == "get":
            return await self._get_reminder(**kwargs)
        elif operation == "list":
            return await self._list_reminders(**kwargs)
        elif operation == "upcoming":
            return await self._get_upcoming_reminders(**kwargs)
        else:
            raise ProductivityError(f"Unknown operation: {operation}", self.skill_id)

    async def _create_reminder(self, **kwargs) -> Dict[str, Any]:
        """Create a new reminder."""
        user_id = kwargs.get("user_id")
        title = kwargs.get("title")
        description = kwargs.get("description")
        due_time_str = kwargs.get("due_time")
        recurrence = kwargs.get("recurrence", "none")
        recurrence_pattern = kwargs.get("recurrence_pattern")
        notify_before_minutes = kwargs.get("notify_before_minutes", 15)
        tags = kwargs.get("tags", [])
        linked_tasks = kwargs.get("linked_tasks", [])

        if not title:
            raise ProductivityError("Reminder title is required", self.skill_id)

        if not due_time_str:
            raise ProductivityError("Reminder due time is required", self.skill_id)

        # Parse due time
        try:
            if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", due_time_str):
                due_time = datetime.fromisoformat(due_time_str)
            else:
                # Try natural language parsing
                due_time = self._parse_date_time_expression(due_time_str)
        except Exception as e:
            raise ProductivityError(f"Invalid due time format: {str(e)}", self.skill_id)

        # Create reminder
        reminder_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        reminder = Reminder(
            reminder_id=reminder_id,
            title=title,
            description=description,
            due_time=due_time,
            created_at=now,
            updated_at=now,
            is_completed=False,
            recurrence=ReminderRecurrence(recurrence),
            recurrence_pattern=recurrence_pattern,
            user_id=user_id,
            notify_before_minutes=notify_before_minutes,
            tags=tags,
            linked_tasks=linked_tasks,
        )

        # Store reminder in database
        await self._store_reminder(reminder)

        # Create calendar event if integration is available and requested
        if self.calendar_api and kwargs.get("add_to_calendar", False):
            try:
                calendar_event = {
                    "title": title,
                    "description": description,
                    "start_time": due_time.isoformat(),
                    "end_time": (due_time + timedelta(minutes=30)).isoformat(),
                    "location": kwargs.get("location"),
                    "reminders": [notify_before_minutes],
                }

                event_result = await self.calendar_api.create_event(user_id, calendar_event)

                # Store calendar event ID in reminder metadata
                reminder.metadata["calendar_event_id"] = event_result["event_id"]
                await self._store_reminder(reminder)
            except Exception as e:
                self.logger.warning(f"Failed to create calendar event: {str(e)}")

        # Emit reminder created event
        await self.event_bus.emit(
            ReminderCreated(
                reminder_id=reminder_id, user_id=user_id, title=title, due_time=due_time.isoformat()
            )
        )

        # Update metrics
        self.metrics.increment("reminders_created_total")

        # Update reminder count for user
        active_reminders = await self._count_active_reminders(user_id)
        self.metrics.set("active_reminders", active_reminders, tags={"user_id": user_id})

        return {
            "reminder_id": reminder_id,
            "message": f"Reminder '{title}' created successfully",
            "reminder": asdict(reminder),
        }

    async def _update_reminder(self, **kwargs) -> Dict[str, Any]:
        """Update an existing reminder."""
        reminder_id = kwargs.get("reminder_id")
        user_id = kwargs.get("user_id")

        if not reminder_id:
            raise ProductivityError("Reminder ID is required for updates", self.skill_id)

        # Get existing reminder
        reminder = await self._get_reminder_by_id(reminder_id)
        if not reminder:
            raise ProductivityError(f"Reminder not found: {reminder_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and reminder.user_id and reminder.user_id != user_id:
            raise ProductivityError("Not authorized to update this reminder", self.skill_id)

        # Update fields
        if "title" in kwargs:
            reminder.title = kwargs["title"]
        if "description" in kwargs:
            reminder.description = kwargs["description"]
        if "due_time" in kwargs:
            due_time_str = kwargs["due_time"]
            if due_time_str:
                try:
                    if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", due_time_str):
                        reminder.due_time = datetime.fromisoformat(due_time_str)
                    else:
                        reminder.due_time = self._parse_date_time_expression(due_time_str)
                except Exception as e:
                    raise ProductivityError(f"Invalid due time format: {str(e)}", self.skill_id)
        if "recurrence" in kwargs:
            reminder.recurrence = ReminderRecurrence(kwargs["recurrence"])
        if "recurrence_pattern" in kwargs:
            reminder.recurrence_pattern = kwargs["recurrence_pattern"]
        if "notify_before_minutes" in kwargs:
            reminder.notify_before_minutes = kwargs["notify_before_minutes"]
        if "tags" in kwargs:
            reminder.tags = kwargs["tags"]
        if "linked_tasks" in kwargs:
            reminder.linked_tasks = kwargs["linked_tasks"]

        # Update timestamp
        reminder.updated_at = datetime.now(timezone.utc)

        # Store updated reminder
        await self._store_reminder(reminder)

        # Update calendar event if integration is available and calendar event exists
        if self.calendar_api and reminder.metadata.get("calendar_event_id"):
            try:
                calendar_event = {
                    "title": reminder.title,
                    "description": reminder.description,
                    "start_time": reminder.due_time.isoformat(),
                    "end_time": (reminder.due_time + timedelta(minutes=30)).isoformat(),
                    "reminders": [reminder.notify_before_minutes],
                }

                await self.calendar_api.update_event(
                    user_id, reminder.metadata["calendar_event_id"], calendar_event
                )
            except Exception as e:
                self.logger.warning(f"Failed to update calendar event: {str(e)}")

        return {
            "reminder_id": reminder_id,
            "message": f"Reminder '{reminder.title}' updated successfully",
            "reminder": asdict(reminder),
        }

    async def _complete_reminder(self, **kwargs) -> Dict[str, Any]:
        """Mark a reminder as completed."""
        reminder_id = kwargs.get("reminder_id")
        user_id = kwargs.get("user_id")

        if not reminder_id:
            raise ProductivityError("Reminder ID is required", self.skill_id)

        # Get existing reminder
        reminder = await self._get_reminder_by_id(reminder_id)
        if not reminder:
            raise ProductivityError(f"Reminder not found: {reminder_id}", self.skill_id)

        # Check ownership if user_id provided
        if user_id and reminder.user_id and reminder.user_id != user_id:
            raise ProductivityError("Not authorized to complete this reminder", self.skill_id)

        # Mark as completed
        reminder.is_completed = True
        reminder.updated_at = datetime.now(timezone.utc)

        # Create next reminder if recurring
        next_reminder = None
        if reminder.recurrence != ReminderRecurrence.NONE:
            next_reminder = await self._create_next_recurring_reminder(reminder)

        # Store updated reminder
        await self._store_reminder(reminder)

        # Update metrics
