"""
Procedural Memory Implementation
Author: Drmusab
Last Modified: 2025-08-09 21:30:00 UTC

This module implements procedural memory - a system for storing skills, procedures,
and how-to knowledge. It provides step-by-step procedure storage, skill execution
tracking, performance optimization, and integration with the skills system.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryItemRetrieved,
    MemoryItemStored,
    MemoryItemUpdated,
)

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    BaseMemoryStore,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryRetentionPolicy,
    MemoryType,
    MemoryUtils,
    SimpleMemoryQuery,
    memory_operation_span,
)

# Skills integration
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator

# Learning integration
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class ProcedureStatus(Enum):
    """Status of a stored procedure."""
    DRAFT = "draft"
    VALIDATED = "validated"
    OPTIMIZING = "optimizing"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class SkillCategory(Enum):
    """Categories of procedural skills."""
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    CREATIVITY = "creativity"
    PRODUCTIVITY = "productivity"
    TECHNICAL = "technical"
    SOCIAL = "social"
    LEARNING = "learning"
    SYSTEM = "system"


@dataclass
class ProcedureStep:
    """Individual step in a procedure."""
    step_id: str
    description: str
    action: str
    prerequisites: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None
    error_handlers: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[float] = None
    retries: int = 0
    optional: bool = False


@dataclass
class ExecutionRecord:
    """Record of a procedure execution."""
    execution_id: str
    procedure_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    steps_completed: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    feedback: Optional[Dict[str, Any]] = None


class ProceduralMemoryConfig:
    """Configuration for procedural memory."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize configuration from config loader."""
        self.max_cache_size = config_loader.get("memory.procedural.cache_size", 500)
        self.optimization_threshold = config_loader.get("memory.procedural.optimization_threshold", 10)
        self.performance_tracking = config_loader.get("memory.procedural.performance_tracking", True)
        self.auto_optimization = config_loader.get("memory.procedural.auto_optimization", True)
        self.skill_integration = config_loader.get("memory.procedural.skill_integration", True)
        self.execution_history_limit = config_loader.get("memory.procedural.execution_history_limit", 100)
        self.optimization_interval = config_loader.get("memory.procedural.optimization_interval", 3600)
        self.validation_required = config_loader.get("memory.procedural.validation_required", True)


class ProceduralMemory(BaseMemory):
    """
    Procedural memory implementation - stores skills, procedures, and how-to knowledge.
    
    This implementation provides:
    - Step-by-step procedure storage and retrieval
    - Skill execution tracking and performance monitoring
    - Automatic procedure optimization based on execution history
    - Integration with the skills system
    - Learning from execution feedback
    - Procedure validation and testing
    - Performance metrics and analytics
    """

    def __init__(
        self,
        container: Container,
        memory_store: BaseMemoryStore,
    ):
        """
        Initialize procedural memory.

        Args:
            container: Dependency injection container
            memory_store: Memory storage backend
        """
        self.container = container
        self.memory_store = memory_store
        self.logger = get_logger(__name__)

        # Get core dependencies
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Load configuration
        config_loader = container.get(ConfigLoader)
        self.config = ProceduralMemoryConfig(config_loader)

        # Get optional dependencies
        try:
            self.skill_registry = container.get(SkillRegistry)
            self.skill_validator = container.get(SkillValidator)
        except Exception:
            self.logger.warning("Skills system not available")
            self.skill_registry = None
            self.skill_validator = None

        try:
            self.continual_learner = container.get(ContinualLearner)
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.logger.warning("Learning components not available")
            self.continual_learner = None
            self.feedback_processor = None

        # Get monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            self.logger.warning("Monitoring components not available")

        # Initialize indices and storage
        self._initialize_storage()

        # Cache
        self._procedure_cache: Dict[str, MemoryItem] = {}
        self._cache_lock = asyncio.Lock()

        # Background task management
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self.logger.info("ProceduralMemory initialized successfully")

    def _initialize_storage(self) -> None:
        """Initialize storage structures for procedural memory."""
        # Indices for efficient retrieval
        self._skill_index: Dict[str, List[str]] = defaultdict(list)
        self._category_index: Dict[SkillCategory, List[str]] = defaultdict(list)
        self._dependency_index: Dict[str, Set[str]] = defaultdict(set)
        self._status_index: Dict[ProcedureStatus, Set[str]] = defaultdict(set)
        
        # Execution tracking
        self._execution_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.execution_history_limit))
        self._performance_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._optimization_queue: deque = deque()
        
        # Skill relationships
        self._skill_hierarchy: Dict[str, Set[str]] = defaultdict(set)
        self._skill_aliases: Dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize procedural memory and start background tasks."""
        try:
            self.logger.info("Initializing procedural memory...")

            # Initialize storage backend if needed
            if hasattr(self.memory_store, 'initialize'):
                await self.memory_store.initialize()

            # Start background tasks
            if self.config.auto_optimization:
                self._background_tasks.append(
                    asyncio.create_task(self._optimization_loop())
                )

            if self.config.performance_tracking:
                self._background_tasks.append(
                    asyncio.create_task(self._performance_tracking_loop())
                )

            # Rebuild indices from existing data
            await self._rebuild_indices()

            # Sync with skill registry if available
            if self.config.skill_integration and self.skill_registry:
                await self._sync_with_skill_registry()

            self.logger.info("Procedural memory initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize procedural memory: {str(e)}")
            raise MemoryError(f"Procedural memory initialization failed: {str(e)}")

    async def shutdown(self) -> None:
        """Shutdown procedural memory and cleanup resources."""
        self.logger.info("Shutting down procedural memory...")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()

        # Cancel and wait for background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Save any pending optimizations
        await self._save_pending_optimizations()

        # Clear caches and indices
        self._procedure_cache.clear()
        self._clear_indices()

        self.logger.info("Procedural memory shutdown complete")

    @handle_exceptions
    async def store(self, data: Any, **kwargs) -> str:
        """
        Store a procedure in procedural memory.

        Args:
            data: Procedure data (can be dict with steps or a Procedure object)
            **kwargs: Additional parameters including:
                skill_name: Name of the skill
                category: Skill category
                dependencies: List of required skills/procedures
                version: Procedure version
                author: Procedure author
                tags: Additional tags

        Returns:
            Memory ID of the stored procedure
        """
        async with memory_operation_span(self.tracer, "store_procedural"):
            # Generate memory ID
            memory_id = MemoryUtils.generate_memory_id()

            # Parse and validate procedure data
            procedure_data = await self._parse_procedure_data(data, **kwargs)

            # Create metadata
            metadata = await self._create_metadata(procedure_data, **kwargs)

            # Create memory item
            memory_item = MemoryItem(
                memory_id=memory_id,
                content=procedure_data,
                memory_type=MemoryType.PROCEDURAL,
                owner_id=kwargs.get("owner_id"),
                context_id=kwargs.get("context_id"),
                metadata=metadata,
            )

            # Validate procedure if required
            if self.config.validation_required:
                validation_result = await self._validate_procedure(procedure_data)
                if not validation_result["valid"]:
                    raise MemoryError(f"Procedure validation failed: {validation_result['errors']}")

            # Store in backend
            await self.memory_store.store_item(memory_item)

            # Update indices
            await self._update_indices_on_store(memory_item, **kwargs)

            # Register with skill system if available
            if self.config.skill_integration and self.skill_registry and kwargs.get("skill_name"):
                await self._register_skill(memory_id, kwargs["skill_name"], procedure_data)

            # Add to cache
            await self._add_to_cache(memory_item)

            # Emit event
            await self.event_bus.emit(
                MemoryItemStored(
                    memory_id=memory_id,
                    memory_type=MemoryType.PROCEDURAL.value,
                    owner_id=kwargs.get("owner_id"),
                    context_id=kwargs.get("context_id"),
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("procedural_memory_items_stored")
                self.metrics.gauge("procedural_skills_count", len(self._skill_index))

            self.logger.debug(f"Stored procedural memory {memory_id}")
            return memory_id

    @handle_exceptions
    async def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a procedure from procedural memory.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory item or None if not found
        """
        async with memory_operation_span(self.tracer, "retrieve_procedural", memory_id):
            # Check cache first
            async with self._cache_lock:
                if memory_id in self._procedure_cache:
                    item = self._procedure_cache[memory_id]
                    item.metadata.update_access()
                    return item

            # Retrieve from store
            item = await self.memory_store.get_item(memory_id)
            
            if item:
                # Update metadata
                item.metadata.update_access()
                
                # Add to cache
                await self._add_to_cache(item)
                
                # Update access in store
                await self.memory_store.update_item(memory_id, {"metadata": item.metadata})
                
                # Emit event
                await self.event_bus.emit(
                    MemoryItemRetrieved(
                        memory_id=memory_id,
                        memory_type=MemoryType.PROCEDURAL.value,
                        owner_id=item.owner_id,
                    )
                )
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("procedural_memory_items_retrieved")

            return item

    async def update(self, memory_id: str, data: Any) -> bool:
        """
        Update existing procedure.

        Args:
            memory_id: Memory identifier
            data: New procedure data

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "update_procedural", memory_id):
            # Retrieve existing memory
            memory_item = await self.retrieve(memory_id)
            if not memory_item:
                return False

            # Parse and validate new data
            procedure_data = await self._parse_procedure_data(data)

            # Update content
            memory_item.content = procedure_data
            memory_item.metadata.update_modification()

            # Update status to draft if it was active
            if memory_item.metadata.custom_metadata.get("status") == ProcedureStatus.ACTIVE.value:
                memory_item.metadata.custom_metadata["status"] = ProcedureStatus.DRAFT.value

            # Store updated item
            await self.memory_store.store_item(memory_item)

            # Update cache
            await self._add_to_cache(memory_item)

            # Update indices
            await self._update_indices_on_update(memory_item)

            # Emit event
            await self.event_bus.emit(
                MemoryItemUpdated(
                    memory_id=memory_id,
                    memory_type=MemoryType.PROCEDURAL.value,
                    owner_id=memory_item.owner_id,
                )
            )

            return True

    async def delete(self, memory_id: str) -> bool:
        """
        Delete procedure.

        Args:
            memory_id: Memory identifier

        Returns:
            True if successful
        """
        async with memory_operation_span(self.tracer, "delete_procedural", memory_id):
            # Remove from indices
            await self._remove_from_indices(memory_id)

            # Remove from cache
            async with self._cache_lock:
                self._procedure_cache.pop(memory_id, None)

            # Remove execution history
            self._execution_history.pop(memory_id, None)
            self._performance_stats.pop(memory_id, None)

            # Unregister from skill system if needed
            if self.skill_registry:
                await self._unregister_skill(memory_id)

            # Delete from store
            return await self.memory_store.delete_item(memory_id)

    async def search(self, query: Any) -> List[MemoryItem]:
        """
        Search procedural memory.

        Args:
            query: Search query (can be string, dict, or SimpleMemoryQuery)

        Returns:
            List of matching memory items
        """
        async with memory_operation_span(self.tracer, "search_procedural"):
            if isinstance(query, SimpleMemoryQuery):
                return await self.memory_store.query(query)

            elif isinstance(query, str):
                return await self._handle_string_query(query)

            elif isinstance(query, dict):
                return await self._handle_dict_query(query)

            else:
                raise MemoryError(f"Unsupported query type for procedural memory: {type(query)}")

    async def clear(self) -> None:
        """Clear all procedural memory."""
        # Clear indices
        self._clear_indices()

        # Clear execution tracking
        self._execution_history.clear()
        self._performance_stats.clear()
        self._optimization_queue.clear()

        # Clear cache
        async with self._cache_lock:
            self._procedure_cache.clear()

        # Clear underlying store
        query = SimpleMemoryQuery(memory_type=MemoryType.PROCEDURAL)
        items = await self.memory_store.query(query)

        for item in items:
            await self.memory_store.delete_item(item.memory_id)

        self.logger.info(f"Cleared all procedural memory ({len(items)} items)")

    async def get_stats(self) -> Dict[str, Any]:
        """Get procedural memory statistics."""
        total_procedures = sum(len(procedures) for procedures in self._skill_index.values())
        
        stats = {
            "total_procedures": total_procedures,
            "skills_count": len(self._skill_index),
            "categories": {
                category.value: len(procedures) 
                for category, procedures in self._category_index.items()
            },
            "status_breakdown": {
                status.value: len(procedures)
                for status, procedures in self._status_index.items()
            },
            "cache_size": len(self._procedure_cache),
            "execution_history_size": sum(len(history) for history in self._execution_history.values()),
            "optimization_queue_size": len(self._optimization_queue),
            "memory_type": MemoryType.PROCEDURAL.value,
        }

        # Add performance summary
        if self._performance_stats:
            stats["performance_summary"] = self._get_performance_summary()

        return stats

    # Additional public methods specific to procedural memory

    async def execute_procedure(
        self, procedure_id: str, context: Dict[str, Any], **kwargs
    ) -> ExecutionRecord:
        """
        Execute a stored procedure and track its performance.

        Args:
            procedure_id: Procedure identifier
            context: Execution context
            **kwargs: Additional execution parameters

        Returns:
            Execution record
        """
        # Retrieve procedure
        procedure_item = await self.retrieve(procedure_id)
        if not procedure_item:
            raise MemoryError(f"Procedure {procedure_id} not found")

        # Create execution record
        execution_record = ExecutionRecord(
            execution_id=MemoryUtils.generate_memory_id(),
            procedure_id=procedure_id,
            start_time=datetime.now(timezone.utc),
            context=context,
        )

        try:
            # Execute procedure steps
            procedure_data = procedure_item.content
            steps = procedure_data.get("steps", [])

            for step in steps:
                step_result = await self._execute_step(step, context, execution_record)
                
                if step_result["success"]:
                    execution_record.steps_completed.append(step["step_id"])
                else:
                    if not step.get("optional", False):
                        execution_record.status = "failed"
                        execution_record.errors.append(step_result["error"])
                        break

            # Mark as completed if all required steps done
            if execution_record.status != "failed":
                execution_record.status = "completed"

        except Exception as e:
            execution_record.status = "error"
            execution_record.errors.append({
                "type": "execution_error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        finally:
            execution_record.end_time = datetime.now(timezone.utc)
            
            # Calculate performance metrics
            duration = (execution_record.end_time - execution_record.start_time).total_seconds()
            execution_record.performance_metrics["duration"] = duration
            execution_record.performance_metrics["steps_completed"] = len(execution_record.steps_completed)
            execution_record.performance_metrics["success_rate"] = (
                len(execution_record.steps_completed) / len(steps) if steps else 0
            )

            # Store execution history
            self._execution_history[procedure_id].append(execution_record)

            # Update performance stats
            await self._update_performance_stats(procedure_id, execution_record)

            # Schedule for optimization if threshold met
            if len(self._execution_history[procedure_id]) >= self.config.optimization_threshold:
                self._optimization_queue.append(procedure_id)

        return execution_record

    async def get_procedure_by_skill(self, skill_name: str) -> Optional[MemoryItem]:
        """Get procedure associated with a skill name."""
        if skill_name not in self._skill_index:
            # Check aliases
            if skill_name in self._skill_aliases:
                skill_name = self._skill_aliases[skill_name]
            else:
                return None

        procedure_ids = self._skill_index[skill_name]
        if not procedure_ids:
            return None

        # Return the most recent or highest version
        items = []
        for proc_id in procedure_ids:
            item = await self.retrieve(proc_id)
            if item:
                items.append(item)

        if not items:
            return None

        # Sort by version or modification time
        items.sort(
            key=lambda x: (
                x.metadata.custom_metadata.get("version", "0"),
                x.metadata.last_modified or x.metadata.created_at
            ),
            reverse=True
        )

        return items[0]

    async def get_procedures_by_category(
        self, category: Union[str, SkillCategory], limit: int = 10
    ) -> List[MemoryItem]:
        """Get procedures in a specific category."""
        if isinstance(category, str):
            try:
                category = SkillCategory(category)
            except ValueError:
                return []

        if category not in self._category_index:
            return []

        procedure_ids = list(self._category_index[category])[:limit]
        return await self._retrieve_multiple(procedure_ids)

    async def get_procedure_dependencies(self, procedure_id: str) -> List[str]:
        """Get dependencies for a procedure."""
        procedure = await self.retrieve(procedure_id)
        if not procedure:
            return []

        return procedure.metadata.custom_metadata.get("dependencies", [])

    async def get_dependent_procedures(self, procedure_id: str) -> List[str]:
        """Get procedures that depend on this one."""
        return list(self._dependency_index.get(procedure_id, set()))

    async def get_execution_history(
        self, procedure_id: str, limit: Optional[int] = None
    ) -> List[ExecutionRecord]:
        """Get execution history for a procedure."""
        history = list(self._execution_history.get(procedure_id, []))
        
        if limit:
            history = history[-limit:]
        
        return history

    async def get_performance_metrics(self, procedure_id: str) -> Dict[str, Any]:
        """Get performance metrics for a procedure."""
        if procedure_id not in self._performance_stats:
            return {}

        stats = self._performance_stats[procedure_id].copy()
        
        # Add execution history summary
        history = self._execution_history.get(procedure_id, [])
        if history:
            stats["execution_count"] = len(history)
            stats["last_execution"] = history[-1].end_time.isoformat() if history[-1].end_time else None
            
            # Calculate success rate
            successful = sum(1 for h in history if h.status == "completed")
            stats["success_rate"] = successful / len(history) if history else 0

        return stats

    async def optimize_procedure(self, procedure_id: str) -> Dict[str, Any]:
        """
        Optimize a procedure based on execution history.

        Args:
            procedure_id: Procedure identifier

        Returns:
            Optimization results
        """
        procedure = await self.retrieve(procedure_id)
        if not procedure:
            return {"status": "error", "message": "Procedure not found"}

        history = self._execution_history.get(procedure_id, [])
        if len(history) < self.config.optimization_threshold:
            return {
                "status": "skipped",
                "message": f"Insufficient execution history ({len(history)}/{self.config.optimization_threshold})"
            }

        try:
            # Analyze execution patterns
            analysis = await self._analyze_execution_patterns(procedure_id, history)

            # Generate optimization suggestions
            suggestions = await self._generate_optimization_suggestions(procedure, analysis)

            # Apply optimizations if auto-optimization enabled
            if self.config.auto_optimization and suggestions:
                applied = await self._apply_optimizations(procedure_id, suggestions)
                
                return {
                    "status": "optimized",
                    "analysis": analysis,
                    "suggestions": suggestions,
                    "applied": applied,
                }
            else:
                return {
                    "status": "analyzed",
                    "analysis": analysis,
                    "suggestions": suggestions,
                }

        except Exception as e:
            self.logger.error(f"Error optimizing procedure {procedure_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def validate_procedure(self, procedure_id: str) -> Dict[str, Any]:
        """
        Validate a stored procedure.

        Args:
            procedure_id: Procedure identifier

        Returns:
            Validation results
        """
        procedure = await self.retrieve(procedure_id)
        if not procedure:
            return {"valid": False, "errors": ["Procedure not found"]}

        return await self._validate_procedure(procedure.content)

    # Private helper methods

    async def _parse_procedure_data(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Parse and structure procedure data."""
        if isinstance(data, dict):
            # Ensure required fields
            procedure_data = {
                "name": data.get("name", kwargs.get("skill_name", "unnamed_procedure")),
                "description": data.get("description", ""),
                "steps": data.get("steps", []),
                "parameters": data.get("parameters", {}),
                "prerequisites": data.get("prerequisites", []),
                "outputs": data.get("outputs", {}),
                "error_handlers": data.get("error_handlers", {}),
                "metadata": data.get("metadata", {}),
            }
        elif isinstance(data, str):
            # Parse as simple procedure description
            procedure_data = {
                "name": kwargs.get("skill_name", "unnamed_procedure"),
                "description": data,
                "steps": self._parse_steps_from_text(data),
                "parameters": {},
                "prerequisites": [],
                "outputs": {},
                "error_handlers": {},
                "metadata": {},
            }
        else:
            raise MemoryError(f"Unsupported procedure data type: {type(data)}")

        # Validate and convert steps
        procedure_data["steps"] = await self._process_steps(procedure_data["steps"])

        return procedure_data

    async def _create_metadata(self, procedure_data: Dict[str, Any], **kwargs) -> MemoryMetadata:
        """Create metadata for a procedural memory item."""
        # Extract metadata
        skill_name = procedure_data.get("name", kwargs.get("skill_name"))
        category = kwargs.get("category", SkillCategory.TECHNICAL)
        dependencies = kwargs.get("dependencies", procedure_data.get("prerequisites", []))
        
        if isinstance(category, str):
            try:
                category = SkillCategory(category)
            except ValueError:
                category = SkillCategory.TECHNICAL

        # Create base metadata
        metadata = MemoryMetadata(
            retention_policy=MemoryRetentionPolicy.EXTENDED,
            tags=set([skill_name, category.value] + dependencies),
            importance=kwargs.get("importance", 0.8),
            custom_metadata={
                "skill_name": skill_name,
                "category": category.value,
                "dependencies": dependencies,
                "version": kwargs.get("version", "1.0.0"),
                "author": kwargs.get("author"),
                "status": ProcedureStatus.DRAFT.value,
                "complexity": self._calculate_complexity(procedure_data),
            }
        )

        return metadata

    async def _process_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        """Process and validate procedure steps."""
        processed_steps = []
        
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                processed_step = {
                    "step_id": step.get("step_id", f"step_{i+1}"),
                    "description": step.get("description", ""),
                    "action": step.get("action", ""),
                    "prerequisites": step.get("prerequisites", []),
                    "parameters": step.get("parameters", {}),
                    "expected_outcome": step.get("expected_outcome"),
                    "error_handlers": step.get("error_handlers", {}),
                    "timeout": step.get("timeout"),
                    "retries": step.get("retries", 0),
                    "optional": step.get("optional", False),
                }
            elif isinstance(step, str):
                processed_step = {
                    "step_id": f"step_{i+1}",
                    "description": step,
                    "action": self._extract_action_from_description(step),
                    "prerequisites": [],
                    "parameters": {},
                    "expected_outcome": None,
                    "error_handlers": {},
                    "timeout": None,
                    "retries": 0,
                    "optional": False,
                }
            else:
                continue
            
            processed_steps.append(processed_step)
        
        return processed_steps

    def _parse_steps_from_text(self, text: str) -> List[str]:
        """Parse steps from text description."""
        # Simple implementation - split by newlines or numbered points
        lines = text.strip().split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering or bullet points
                step_text = line.lstrip('0123456789.-* \t')
                if step_text:
                    steps.append(step_text)
        
        return steps if steps else [text]

    def _extract_action_from_description(self, description: str) -> str:
        """Extract action identifier from step description."""
        # Simple implementation - extract verb or first word
        words = description.split()
        if words:
            return words[0].lower()
        return "execute"

    def _calculate_complexity(self, procedure_data: Dict[str, Any]) -> float:
        """Calculate complexity score for a procedure."""
        steps = procedure_data.get("steps", [])
        
        # Factors contributing to complexity
        step_count = len(steps)
        prerequisite_count = len(procedure_data.get("prerequisites", []))
        param_count = len(procedure_data.get("parameters", {}))
        
        # Calculate complexity score (0-1)
        complexity = min(1.0, (
            (step_count / 20) * 0.5 +
            (prerequisite_count / 5) * 0.3 +
            (param_count / 10) * 0.2
        ))
        
        return complexity

    async def _validate_procedure(self, procedure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a procedure."""
        errors = []
        warnings = []
        
        # Check required fields
        if not procedure_data.get("name"):
            errors.append("Procedure name is required")
        
        if not procedure_data.get("steps"):
            errors.append("Procedure must have at least one step")
        
        # Validate steps
        steps = procedure_data.get("steps", [])
        for i, step in enumerate(steps):
            if not step.get("description") and not step.get("action"):
                errors.append(f"Step {i+1} must have description or action")
            
            # Check for circular dependencies
            if step.get("prerequisites"):
                for prereq in step["prerequisites"]:
                    if prereq == step.get("step_id"):
                        errors.append(f"Step {i+1} has circular dependency")
        
        # Validate with skill validator if available
        if self.skill_validator:
            skill_validation = await self.skill_validator.validate_procedure(procedure_data)
            if not skill_validation.get("valid"):
                errors.extend(skill_validation.get("errors", []))
                warnings.extend(skill_validation.get("warnings", []))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    async def _update_indices_on_store(self, memory_item: MemoryItem, **kwargs) -> None:
        """Update indices when storing a procedure."""
        memory_id = memory_item.memory_id
        metadata = memory_item.metadata.custom_metadata
        
        # Update skill index
        skill_name = metadata.get("skill_name")
        if skill_name:
            self._skill_index[skill_name].append(memory_id)
        
        # Update category index
        category_str = metadata.get("category")
        if category_str:
            try:
                category = SkillCategory(category_str)
                self._category_index[category].append(memory_id)
            except ValueError:
                pass
        
        # Update dependency index
        dependencies = metadata.get("dependencies", [])
        for dep in dependencies:
            self._dependency_index[dep].add(memory_id)
        
        # Update status index
        status_str = metadata.get("status", ProcedureStatus.DRAFT.value)
        try:
            status = ProcedureStatus(status_str)
            self._status_index[status].add(memory_id)
        except ValueError:
            pass

    async def _update_indices_on_update(self, memory_item: MemoryItem) -> None:
        """Update indices when updating a procedure."""
        # Remove from old indices
        await self._remove_from_indices(memory_item.memory_id)
        
        # Add to new indices
        await self._update_indices_on_store(memory_item)

    async def _remove_from_indices(self, memory_id: str) -> None:
        """Remove a procedure from all indices."""
        # Retrieve memory to get metadata
        memory = await self.memory_store.get_item(memory_id)
        if not memory:
            return
        
        metadata = memory.metadata.custom_metadata
        
        # Remove from skill index
        skill_name = metadata.get("skill_name")
        if skill_name and skill_name in self._skill_index:
            self._skill_index[skill_name] = [
                mid for mid in self._skill_index[skill_name] if mid != memory_id
            ]
        
        # Remove from category index
        category_str = metadata.get("category")
        if category_str:
            try:
                category = SkillCategory(category_str)
                if category in self._category_index:
                    self._category_index[category] = [
                        mid for mid in self._category_index[category] if mid != memory_id
                    ]
            except ValueError:
                pass
        
        # Remove from dependency index
        for dep_list in self._dependency_index.values():
            dep_list.discard(memory_id)
        
        # Remove from status index
        for status_set in self._status_index.values():
            status_set.discard(memory_id)

    def _clear_indices(self) -> None:
        """Clear all procedural memory indices."""
        self._skill_index.clear()
        self._category_index.clear()
        self._dependency_index.clear()
        self._status_index.clear()
        self._skill_hierarchy.clear()
        self._skill_aliases.clear()

    async def _rebuild_indices(self) -> None:
        """Rebuild indices from stored procedures."""
        self.logger.info("Rebuilding procedural memory indices...")
        
        # Clear existing indices
        self._clear_indices()
        
        # Query all procedural memories
        query = SimpleMemoryQuery(memory_type=MemoryType.PROCEDURAL, limit=10000)
        items = await self.memory_store.query(query)
        
        # Rebuild indices
        for item in items:
            await self._update_indices_on_store(item)
        
        self.logger.info(f"Rebuilt indices for {len(items)} procedures")

    async def _sync_with_skill_registry(self) -> None:
        """Sync with skill registry to ensure consistency."""
        if not self.skill_registry:
            return
        
        try:
            # Get all registered skills
            registered_skills = await self.skill_registry.get_all_skills()
            
            # Update skill aliases
            for skill in registered_skills:
                if skill.get("aliases"):
                    for alias in skill["aliases"]:
                        self._skill_aliases[alias] = skill["name"]
            
            self.logger.info(f"Synced with skill registry: {len(registered_skills)} skills")
            
        except Exception as e:
            self.logger.warning(f"Failed to sync with skill registry: {str(e)}")

    async def _add_to_cache(self, item: MemoryItem) -> None:
        """Add item to procedure cache."""
        async with self._cache_lock:
            # Enforce cache size limit
            if len(self._procedure_cache) >= self.config.max_cache_size:
                # Remove oldest item
                if self._procedure_cache:
                    self._procedure_cache.pop(next(iter(self._procedure_cache)))
            
            # Add to cache
            self._procedure_cache[item.memory_id] = item

    async def _retrieve_multiple(self, memory_ids: List[str]) -> List[MemoryItem]:
        """Retrieve multiple memory items efficiently."""
        items = []
        
        for memory_id in memory_ids:
            item = await self.retrieve(memory_id)
            if item:
                items.append(item)
        
        return items

    async def _handle_string_query(self, query: str) -> List[MemoryItem]:
        """Handle string-based queries."""
        query_lower = query.lower()
        
        if query.startswith("skill:"):
            skill_name = query.split(":", 1)[1].strip()
            item = await self.get_procedure_by_skill(skill_name)
            return [item] if item else []
        
        elif query.startswith("category:"):
            category = query.split(":", 1)[1].strip()
            return await self.get_procedures_by_category(category)
        
        elif query.startswith("status:"):
            status_str = query.split(":", 1)[1].strip()
            try:
                status = ProcedureStatus(status_str)
                procedure_ids = list(self._status_index.get(status, set()))[:10]
                return await self._retrieve_multiple(procedure_ids)
            except ValueError:
                return []
        
        else:
            # Search across skill names and descriptions
            matching_ids = set()
            
            # Search in skill names
            for skill_name, proc_ids in self._skill_index.items():
                if query_lower in skill_name.lower():
                    matching_ids.update(proc_ids)
            
            # Retrieve and filter by description
            items = await self._retrieve_multiple(list(matching_ids)[:20])
            
            # Additional filtering by content
            filtered_items = []
            for item in items:
                content = item.content
                if isinstance(content, dict):
                    description = content.get("description", "")
                    if query_lower in description.lower():
                        filtered_items.append(item)
                elif isinstance(content, str) and query_lower in content.lower():
                    filtered_items.append(item)
            
            return filtered_items[:10]

    async def _handle_dict_query(self, query: dict) -> List[MemoryItem]:
        """Handle dictionary-based queries."""
        # Extract query parameters
        skill_name = query.get("skill_name")
        category = query.get("category")
        status = query.get("status")
        dependencies = query.get("dependencies", [])
        
        # Find matching procedure IDs
        matching_ids = None
        
        # Filter by skill name
        if skill_name:
            if skill_name in self._skill_index:
                matching_ids = set(self._skill_index[skill_name])
            else:
                return []
        
        # Filter by category
        if category:
            try:
                category_enum = SkillCategory(category)
                category_ids = set(self._category_index.get(category_enum, []))
                if matching_ids is None:
                    matching_ids = category_ids
                else:
                    matching_ids = matching_ids.intersection(category_ids)
            except ValueError:
                pass
        
        # Filter by status
        if status:
            try:
                status_enum = ProcedureStatus(status)
                status_ids = self._status_index.get(status_enum, set())
                if matching_ids is None:
                    matching_ids = status_ids
                else:
                    matching_ids = matching_ids.intersection(status_ids)
            except ValueError:
                pass
        
        # Filter by dependencies
        if dependencies:
            dep_ids = set()
            for dep in dependencies:
                if dep in self._dependency_index:
                    dep_ids.update(self._dependency_index[dep])
            if matching_ids is None:
                matching_ids = dep_ids
            else:
                matching_ids = matching_ids.intersection(dep_ids)
        
        if matching_ids is None:
            return []
        
        # Apply limit
        limit = query.get("limit", 10)
        procedure_ids = list(matching_ids)[:limit]
        
        return await self._retrieve_multiple(procedure_ids)

    async def _execute_step(
        self, step: Dict[str, Any], context: Dict[str, Any], execution_record: ExecutionRecord
    ) -> Dict[str, Any]:
        """Execute a single procedure step."""
        start_time = time.time()
        
        try:
            # Check prerequisites
            prerequisites = step.get("prerequisites", [])
            for prereq in prerequisites:
                if prereq not in execution_record.steps_completed:
                    return {
                        "success": False,
                        "error": {
                            "type": "prerequisite_not_met",
                            "message": f"Prerequisite step '{prereq}' not completed",
                            "step_id": step["step_id"],
                        }
                    }
            
            # Simulate step execution (in practice, would call actual implementation)
            # This is where integration with skills system would happen
            action = step.get("action", "execute")
            
            # Add artificial delay for simulation
            await asyncio.sleep(0.1)
            
            # Record success
            duration = time.time() - start_time
            
            return {
                "success": True,
                "duration": duration,
                "output": f"Executed {action} for step {step['step_id']}",
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": {
                    "type": "execution_error",
                    "message": str(e),
                    "step_id": step["step_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }

    async def _update_performance_stats(
        self, procedure_id: str, execution_record: ExecutionRecord
    ) -> None:
        """Update performance statistics for a procedure."""
        if procedure_id not in self._performance_stats:
            self._performance_stats[procedure_id] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0,
            }
        
        stats = self._performance_stats[procedure_id]
        duration = execution_record.performance_metrics.get("duration", 0)
        
        # Update counters
        stats["total_executions"] += 1
        if execution_record.status == "completed":
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
        
        # Update duration stats
        stats["total_duration"] += duration
        stats["avg_duration"] = stats["total_duration"] / stats["total_executions"]
        stats["min_duration"] = min(stats["min_duration"], duration)
        stats["max_duration"] = max(stats["max_duration"], duration)
        
        # Update metrics if available
        if self.metrics:
            self.metrics.increment("procedural_executions_total")
            if execution_record.status == "completed":
                self.metrics.increment("procedural_executions_successful")
            else:
                self.metrics.increment("procedural_executions_failed")
            self.metrics.record("procedural_execution_duration", duration)

    async def _register_skill(
        self, procedure_id: str, skill_name: str, procedure_data: Dict[str, Any]
    ) -> None:
        """Register procedure with skill registry."""
        if not self.skill_registry:
            return
        
        try:
            await self.skill_registry.register_skill({
                "name": skill_name,
                "procedure_id": procedure_id,
                "description": procedure_data.get("description", ""),
                "category": procedure_data.get("category", SkillCategory.TECHNICAL.value),
                "parameters": procedure_data.get("parameters", {}),
            })
        except Exception as e:
            self.logger.warning(f"Failed to register skill {skill_name}: {str(e)}")

    async def _unregister_skill(self, procedure_id: str) -> None:
        """Unregister procedure from skill registry."""
        if not self.skill_registry:
            return
        
        try:
            # Find skill name for procedure
            for skill_name, proc_ids in self._skill_index.items():
                if procedure_id in proc_ids:
                    await self.skill_registry.unregister_skill(skill_name)
                    break
        except Exception as e:
            self.logger.warning(f"Failed to unregister skill for {procedure_id}: {str(e)}")

    async def _analyze_execution_patterns(
        self, procedure_id: str, history: List[ExecutionRecord]
    ) -> Dict[str, Any]:
        """Analyze execution patterns from history."""
        analysis = {
            "total_executions": len(history),
            "success_rate": sum(1 for h in history if h.status == "completed") / len(history),
            "common_errors": defaultdict(int),
            "step_performance": defaultdict(lambda: {"count": 0, "total_duration": 0}),
            "bottlenecks": [],
            "recommendations": [],
        }
        
        # Analyze errors
        for record in history:
            for error in record.errors:
                error_type = error.get("type", "unknown")
                analysis["common_errors"][error_type] += 1
        
        # Analyze step performance (simplified)
        for record in history:
            for step_id in record.steps_completed:
                analysis["step_performance"][step_id]["count"] += 1
        
        # Identify bottlenecks (steps that frequently fail or timeout)
        # This is a simplified implementation
        
        return analysis

    async def _generate_optimization_suggestions(
        self, procedure: MemoryItem, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Check success rate
        if analysis["success_rate"] < 0.8:
            suggestions.append({
                "type": "improve_reliability",
                "priority": "high",
                "description": f"Success rate is {analysis['success_rate']:.1%}, consider adding error handling",
            })
        
        # Check for common errors
        for error_type, count in analysis["common_errors"].items():
            if count > 2:
                suggestions.append({
                    "type": "handle_error",
                    "priority": "medium",
                    "description": f"Error '{error_type}' occurred {count} times",
                    "error_type": error_type,
                })
        
        # Add learning-based suggestions if available
        if self.continual_learner:
            learned_suggestions = await self.continual_learner.suggest_optimizations(
                procedure.content, analysis
            )
            suggestions.extend(learned_suggestions)
        
        return suggestions

    async def _apply_optimizations(
        self, procedure_id: str, suggestions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply optimization suggestions to a procedure."""
        applied = []
        
        # This is a simplified implementation
        # In practice, would modify the procedure based on suggestions
        
        for suggestion in suggestions:
            if suggestion["priority"] == "high":
                # Apply high priority optimizations
                applied.append({
                    "suggestion": suggestion,
                    "status": "applied",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        
        # Update procedure status if optimizations applied
        if applied:
            procedure = await self.retrieve(procedure_id)
            if procedure:
                procedure.metadata.custom_metadata["status"] = ProcedureStatus.OPTIMIZING.value
                procedure.metadata.custom_metadata["last_optimized"] = datetime.now(timezone.utc).isoformat()
                await self.memory_store.store_item(procedure)
        
        return applied

    async def _save_pending_optimizations(self) -> None:
        """Save any pending optimizations before shutdown."""
        if not self._optimization_queue:
            return
        
        self.logger.info(f"Saving {len(self._optimization_queue)} pending optimizations")
        
        # Process remaining optimizations
        while self._optimization_queue:
            procedure_id = self._optimization_queue.popleft()
            try:
                await self.optimize_procedure(procedure_id)
            except Exception as e:
                self.logger.error(f"Error saving optimization for {procedure_id}: {str(e)}")

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance across all procedures."""
        if not self._performance_stats:
            return {}
        
        total_executions = sum(s["total_executions"] for s in self._performance_stats.values())
        successful_executions = sum(s["successful_executions"] for s in self._performance_stats.values())
        
        return {
            "total_procedures": len(self._performance_stats),
            "total_executions": total_executions,
            "overall_success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "avg_duration": sum(s["avg_duration"] for s in self._performance_stats.values()) / len(self._performance_stats) if self._performance_stats else 0,
        }

    async def _optimization_loop(self) -> None:
        """Background task for periodic procedure optimization."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for optimization interval
                await asyncio.sleep(self.config.optimization_interval)
                
                if not self._shutdown_event.is_set() and self._optimization_queue:
                    self.logger.info(f"Processing {len(self._optimization_queue)} procedures for optimization")
                    
                    # Process optimization queue
                    procedures_to_optimize = list(self._optimization_queue)
                    self._optimization_queue.clear()
                    
                    for procedure_id in procedures_to_optimize:
                        try:
                            await self.optimize_procedure(procedure_id)
                        except Exception as e:
                            self.logger.error(f"Error optimizing procedure {procedure_id}: {str(e)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {str(e)}")
                await asyncio.sleep(300)  # Retry after 5 minutes on error

    async def _performance_tracking_loop(self) -> None:
        """Background task for performance tracking and reporting."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for tracking interval (5 minutes)
                await asyncio.sleep(300)
                
                if not self._shutdown_event.is_set():
                    # Generate performance report
                    summary = self._get_performance_summary()
                    
                    if summary and self.metrics:
                        # Update metrics
                        self.metrics.gauge("procedural_total_procedures", summary["total_procedures"])
                        self.metrics.gauge("procedural_overall_success_rate", summary["overall_success_rate"])
                        self.metrics.gauge("procedural_avg_duration", summary["avg_duration"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance tracking loop: {str(e)}")
                await asyncio.sleep(60)
