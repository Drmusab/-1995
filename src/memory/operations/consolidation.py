"""
Memory Consolidation System
Author: Drmusab
Last Modified: 2025-07-05 10:21:06 UTC

This module provides memory consolidation functionality for the AI assistant,
transferring and organizing information between different memory systems.
It implements cognitive-inspired consolidation processes that summarize,
abstract, and integrate memories, supporting long-term knowledge retention
and efficient recall.
"""

import json
import logging
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import asyncio

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    MemoryConsolidationCompleted,
    MemoryConsolidationFailed,
    MemoryConsolidationStarted,
    MemoryGraphUpdated,
    MemoryItemStored,
    MemoryItemUpdated,
    SemanticMemoryExtracted,
)

# Integration components
from src.integrations.llm.model_router import ModelRouter

# Memory system imports
from src.memory.core_memory.base_memory import (
    BaseMemory,
    MemoryError,
    MemoryItem,
    MemoryMetadata,
    MemoryRetentionPolicy,
    MemoryStorageType,
    MemoryType,
)
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory
from src.memory.storage.memory_graph import (
    GraphEdge,
    GraphNode,
    GraphNodeType,
    MemoryGraphStore,
    RelationshipType,
)
from src.memory.storage.vector_store import VectorMemoryStore
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Processing components
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.natural_language.intent_manager import IntentManager


class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation."""

    SESSION_BASED = "session_based"  # Consolidate when a session ends
    THRESHOLD_BASED = "threshold_based"  # Consolidate when memory exceeds capacity
    TEMPORAL = "temporal"  # Regular time-based consolidation
    PRIORITY_BASED = "priority_based"  # Consolidate important memories first
    HYBRID = "hybrid"  # Combination of multiple strategies


class ConsolidationLevel(Enum):
    """Levels of memory consolidation processing."""

    BASIC = "basic"  # Simple transfer without processing
    STANDARD = "standard"  # Regular summarization and organization
    DEEP = "deep"  # Advanced abstraction and integration
    COMPREHENSIVE = "comprehensive"  # Full semantic extraction and relationship building


@dataclass
class ConsolidationJob:
    """
    A job for memory consolidation.

    This class encapsulates the parameters and data for a memory consolidation
    operation, tracking the process from initiation to completion.
    """

    # Job identity
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Source information
    source_type: MemoryType = MemoryType.WORKING
    memory_ids: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Target information
    target_types: List[MemoryType] = field(default_factory=lambda: [MemoryType.EPISODIC])

    # Configuration
    strategy: ConsolidationStrategy = ConsolidationStrategy.SESSION_BASED
    level: ConsolidationLevel = ConsolidationLevel.STANDARD

    # Status tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"

    # Results
    consolidated_memory_ids: List[str] = field(default_factory=list)
    extracted_semantic_facts: List[Dict[str, Any]] = field(default_factory=list)
    relationships_created: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None

    def start(self) -> None:
        """Mark the job as started."""
        self.started_at = datetime.now(timezone.utc)
        self.status = "processing"

    def complete(self, results: Dict[str, Any]) -> None:
        """
        Mark the job as completed with results.

        Args:
            results: Results of the consolidation
        """
        self.completed_at = datetime.now(timezone.utc)
        self.status = "completed"

        if "memory_ids" in results:
            self.consolidated_memory_ids = results["memory_ids"]

        if "semantic_facts" in results:
            self.extracted_semantic_facts = results["semantic_facts"]

        if "relationships" in results:
            self.relationships_created = results["relationships"]

        if "processing_time" in results:
            self.processing_time = results["processing_time"]

    def fail(self, error_message: str) -> None:
        """
        Mark the job as failed.

        Args:
            error_message: Error message
        """
        self.completed_at = datetime.now(timezone.utc)
        self.status = "failed"
        self.error = error_message


class MemoryConsolidator:
    """
    Memory consolidation system for the AI assistant.

    This class provides methods for consolidating memories between different
    memory systems, implementing cognitive-inspired processes for summarization,
    abstraction, pattern recognition, and knowledge integration.
    """

    def __init__(self, container: Container):
        """
        Initialize the memory consolidator.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Load configuration
        self.config_loader = container.get(ConfigLoader)
        self.consolidation_config = self.config_loader.get("memory.consolidation", {})

        # Event system
        self.event_bus = container.get(EventBus)

        # Get memory systems
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)

        # Get storage systems
        self.vector_store = container.get(VectorMemoryStore)
        try:
            self.graph_store = container.get(MemoryGraphStore)
        except Exception:
            self.logger.warning(
                "MemoryGraphStore not available, relationship consolidation will be limited"
            )
            self.graph_store = None

        # Get model router for LLM operations
        try:
            self.model_router = container.get(ModelRouter)
        except Exception:
            self.logger.warning(
                "ModelRouter not available, advanced consolidation features will be limited"
            )
            self.model_router = None

        # NLP components
        try:
            self.entity_extractor = container.get(EntityExtractor)
        except Exception:
            self.logger.warning("EntityExtractor not available, entity extraction will be limited")
            self.entity_extractor = None

        try:
            self.intent_manager = container.get(IntentManager)
        except Exception:
            self.logger.warning("IntentManager not available, intent detection will be limited")
            self.intent_manager = None

        # Monitoring components
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.logger.warning("Monitoring components not available")
            self.metrics = None
            self.tracer = None

        # Configure settings
        self.default_level = ConsolidationLevel(
            self.consolidation_config.get("default_level", "standard")
        )
        self.default_strategy = ConsolidationStrategy(
            self.consolidation_config.get("default_strategy", "session_based")
        )
        self.auto_consolidation = self.consolidation_config.get("auto_consolidation", True)
        self.auto_semantic_extraction = self.consolidation_config.get(
            "auto_semantic_extraction", True
        )
        self.auto_relationship_building = self.consolidation_config.get(
            "auto_relationship_building", True
        )
        self.working_memory_threshold = self.consolidation_config.get(
            "working_memory_threshold", 100
        )
        self.temporal_schedule_minutes = self.consolidation_config.get(
            "temporal_schedule_minutes", 60
        )
        self.max_job_age_days = self.consolidation_config.get("max_job_age_days", 7)
        self.max_concurrent_jobs = self.consolidation_config.get("max_concurrent_jobs", 3)

        # Initialize job tracking
        self._active_jobs: Dict[str, ConsolidationJob] = {}
        self._completed_jobs: Dict[str, ConsolidationJob] = {}
        self._pending_jobs: List[ConsolidationJob] = []
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)

        # Scheduled tasks
        self._scheduled_tasks = []

        # Register metrics
        if self.metrics:
            self.metrics.register_counter("memory_consolidation_jobs_total")
            self.metrics.register_counter("memory_consolidation_jobs_completed")
            self.metrics.register_counter("memory_consolidation_jobs_failed")
            self.metrics.register_counter("memory_consolidation_items_processed")
            self.metrics.register_histogram("memory_consolidation_processing_time_seconds")
            self.metrics.register_gauge("memory_consolidation_active_jobs")

        self.logger.info("MemoryConsolidator initialized")

    async def initialize(self) -> None:
        """Initialize the consolidator and start background tasks."""
        # Set up event handlers
        await self._register_event_handlers()

        # Start temporal consolidation if enabled
        if self.auto_consolidation and self.default_strategy == ConsolidationStrategy.TEMPORAL:
            schedule_minutes = self.temporal_schedule_minutes
            self.logger.info(f"Setting up temporal consolidation every {schedule_minutes} minutes")

            # Schedule periodic task
            task = asyncio.create_task(self._run_temporal_consolidation(schedule_minutes))
            self._scheduled_tasks.append(task)

        self.logger.info("MemoryConsolidator background tasks started")

    async def shutdown(self) -> None:
        """Shut down the consolidator and clean up resources."""
        # Cancel all scheduled tasks
        for task in self._scheduled_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)

        # Complete any active jobs
        for job_id, job in self._active_jobs.items():
            if job.status == "processing":
                job.fail("Shutdown requested")
                self._completed_jobs[job_id] = job

        self._active_jobs.clear()
        self.logger.info("MemoryConsolidator shutdown completed")

    async def consolidate_session(
        self,
        session_id: str,
        level: Optional[ConsolidationLevel] = None,
        include_semantic: bool = True,
    ) -> str:
        """
        Consolidate memories from a completed session.

        Args:
            session_id: Session identifier
            level: Consolidation processing level
            include_semantic: Whether to extract semantic memories

        Returns:
            Job identifier
        """
        # Create consolidation job
        job = ConsolidationJob(
            source_type=MemoryType.WORKING,
            session_id=session_id,
            target_types=[MemoryType.EPISODIC]
            + ([MemoryType.SEMANTIC] if include_semantic else []),
            strategy=ConsolidationStrategy.SESSION_BASED,
            level=level or self.default_level,
        )

        # Schedule job
        return await self._schedule_job(job)

    async def consolidate_memories(
        self,
        memory_ids: List[str],
        source_type: MemoryType = MemoryType.WORKING,
        target_types: Optional[List[MemoryType]] = None,
        level: Optional[ConsolidationLevel] = None,
    ) -> str:
        """
        Consolidate specific memories.

        Args:
            memory_ids: List of memory identifiers
            source_type: Source memory type
            target_types: Target memory types
            level: Consolidation processing level

        Returns:
            Job identifier
        """
        # Create consolidation job
        job = ConsolidationJob(
            source_type=source_type,
            memory_ids=memory_ids,
            target_types=target_types or [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            strategy=ConsolidationStrategy.PRIORITY_BASED,
            level=level or self.default_level,
        )

        # Schedule job
        return await self._schedule_job(job)

    async def extract_semantic_memory(
        self, source_memory_ids: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract semantic memories from source memories.

        Args:
            source_memory_ids: Source memory identifiers
            context: Optional context information

        Returns:
            List of extracted semantic facts
        """
        if not self.model_router:
            self.logger.warning("Cannot extract semantic memory without ModelRouter")
            return []

        start_time = time.time()

        try:
            # Collect source memories
            source_memories = []
            for memory_id in source_memory_ids:
                # Try to retrieve from each memory system
                memory = await self.working_memory.retrieve(memory_id)
                if not memory:
                    memory = await self.episodic_memory.retrieve(memory_id)

                if memory:
                    source_memories.append(memory)

            if not source_memories:
                self.logger.warning(f"No source memories found for semantic extraction")
                return []

            # Extract semantic facts
            semantic_facts = await self._extract_semantic_facts(source_memories, context)

            # Store facts in semantic memory
            stored_facts = []
            for fact in semantic_facts:
                try:
                    # Prepare storage parameters
                    fact_content = fact["content"]
                    confidence = fact.get("confidence", 0.8)
                    domain = fact.get("domain", "general")
                    concepts = fact.get("concepts", [])
                    entities = fact.get("entities", [])

                    # Store in semantic memory
                    memory_id = await self.semantic_memory.store(
                        fact_content,
                        concepts=concepts,
                        domain=domain,
                        entities=entities,
                        confidence=confidence,
                    )

                    # Add memory ID to fact
                    fact["memory_id"] = memory_id
                    stored_facts.append(fact)

                except Exception as e:
                    self.logger.error(f"Failed to store semantic fact: {str(e)}")

            # Emit event for semantic memory extraction
            await self.event_bus.emit(
                SemanticMemoryExtracted(
                    source_count=len(source_memories),
                    fact_count=len(stored_facts),
                    extraction_time=time.time() - start_time,
                )
            )

            # Update metrics
            if self.metrics:
                self.metrics.increment("memory_semantic_facts_extracted", len(stored_facts))
                self.metrics.record(
                    "memory_semantic_extraction_time_seconds", time.time() - start_time
                )

            return stored_facts

        except Exception as e:
            self.logger.error(f"Failed to extract semantic memory: {str(e)}")
            traceback.print_exc()
            return []

    async def build_relationships(
        self, memory_ids: List[str], context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Build relationships between memories in the memory graph.

        Args:
            memory_ids: Memory identifiers
            context: Optional context information

        Returns:
            Number of relationships created
        """
        if not self.graph_store:
            self.logger.warning("Cannot build relationships without MemoryGraphStore")
            return 0

        try:
            # Collect memories
            memories = []
            for memory_id in memory_ids:
                # Try to retrieve from each memory system
                memory = None
                for memory_system in [
                    self.working_memory,
                    self.episodic_memory,
                    self.semantic_memory,
                ]:
                    memory = await memory_system.retrieve(memory_id)
                    if memory:
                        break

                if memory:
                    memories.append(memory)

            if not memories:
                self.logger.warning(f"No memories found for relationship building")
                return 0

            # Create relationship graph
            return await self._build_memory_relationships(memories, context)

        except Exception as e:
            self.logger.error(f"Failed to build relationships: {str(e)}")
            return 0

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a consolidation job.

        Args:
            job_id: Job identifier

        Returns:
            Job status or None if not found
        """
        # Check active jobs
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            return self._job_to_dict(job)

        # Check completed jobs
        if job_id in self._completed_jobs:
            job = self._completed_jobs[job_id]
            return self._job_to_dict(job)

        # Check pending jobs
        for job in self._pending_jobs:
            if job.job_id == job_id:
                return self._job_to_dict(job)

        return None

    async def get_active_jobs(self) -> List[Dict[str, Any]]:
        """
        Get a list of active consolidation jobs.

        Returns:
            List of active jobs
        """
        return [self._job_to_dict(job) for job in self._active_jobs.values()]

    async def get_completed_jobs(
        self, limit: int = 10, include_failed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get a list of completed consolidation jobs.

        Args:
            limit: Maximum number of jobs to return
            include_failed: Whether to include failed jobs

        Returns:
            List of completed jobs
        """
        # Filter and sort jobs
        jobs = list(self._completed_jobs.values())

        if not include_failed:
            jobs = [job for job in jobs if job.status == "completed"]

        # Sort by completion time (most recent first)
        jobs.sort(key=lambda j: j.completed_at or j.created_at, reverse=True)

        # Convert to dictionaries
        return [self._job_to_dict(job) for job in jobs[:limit]]

    async def cleanup_old_jobs(self, max_age_days: Optional[int] = None) -> int:
        """
        Clean up old completed jobs.

        Args:
            max_age_days: Maximum age in days (default from config)

        Returns:
            Number of jobs cleaned up
        """
        max_age = max_age_days or self.max_job_age_days
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age)

        # Find old jobs
        old_job_ids = [
            job_id
            for job_id, job in self._completed_jobs.items()
            if job.completed_at and job.completed_at < cutoff_time
        ]

        # Remove old jobs
        for job_id in old_job_ids:
            del self._completed_jobs[job_id]

        self.logger.info(f"Cleaned up {len(old_job_ids)} old consolidation jobs")
        return len(old_job_ids)

    async def _schedule_job(self, job: ConsolidationJob) -> str:
        """
        Schedule a consolidation job for execution.

        Args:
            job: Consolidation job

        Returns:
            Job identifier
        """
        # Add to pending queue
        self._pending_jobs.append(job)

        # Start processing in background
        asyncio.create_task(self._process_pending_jobs())

        # Update metrics
        if self.metrics:
            self.metrics.increment("memory_consolidation_jobs_total")
            self.metrics.gauge("memory_consolidation_pending_jobs", len(self._pending_jobs))

        return job.job_id

    async def _process_pending_jobs(self) -> None:
        """Process pending consolidation jobs."""
        # Process jobs while we have pending ones and available concurrency
        while self._pending_jobs and len(self._active_jobs) < self.max_concurrent_jobs:
            # Get next job
            job = self._pending_jobs.pop(0)

            # Start job
            self._active_jobs[job.job_id] = job

            # Execute job in background
            asyncio.create_task(self._execute_job(job))

            # Update metrics
            if self.metrics:
                self.metrics.gauge("memory_consolidation_active_jobs", len(self._active_jobs))
                self.metrics.gauge("memory_consolidation_pending_jobs", len(self._pending_jobs))

    async def _execute_job(self, job: ConsolidationJob) -> None:
        """
        Execute a consolidation job.

        Args:
            job: Consolidation job to execute
        """
        # Acquire semaphore to limit concurrency
        async with self._semaphore:
            job.start()
            start_time = time.time()

            # Emit consolidation started event
            await self.event_bus.emit(
                MemoryConsolidationStarted(
                    job_id=job.job_id,
                    strategy=job.strategy.value,
                    level=job.level.value,
                    source_type=job.source_type.value,
                    session_id=job.session_id,
                )
            )

            try:
                # Choose consolidation method based on strategy
                if job.strategy == ConsolidationStrategy.SESSION_BASED:
                    results = await self._consolidate_session_memories(job)
                elif job.strategy == ConsolidationStrategy.THRESHOLD_BASED:
                    results = await self._consolidate_threshold_memories(job)
                elif job.strategy == ConsolidationStrategy.PRIORITY_BASED:
                    results = await self._consolidate_priority_memories(job)
                else:  # Default/temporal
                    results = await self._consolidate_temporal_memories(job)

                # Add processing time
                results["processing_time"] = time.time() - start_time

                # Mark job as completed
                job.complete(results)

                # Move to completed jobs
                self._completed_jobs[job.job_id] = job
                del self._active_jobs[job.job_id]

                # Emit consolidation completed event
                await self.event_bus.emit(
                    MemoryConsolidationCompleted(
                        job_id=job.job_id,
                        processing_time=results["processing_time"],
                        memory_count=len(results.get("memory_ids", [])),
                        semantic_fact_count=len(results.get("semantic_facts", [])),
                        relationship_count=results.get("relationships", 0),
                    )
                )

                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_consolidation_jobs_completed")
                    self.metrics.increment(
                        "memory_consolidation_items_processed", len(results.get("memory_ids", []))
                    )
                    self.metrics.record(
                        "memory_consolidation_processing_time_seconds", results["processing_time"]
                    )
                    self.metrics.gauge("memory_consolidation_active_jobs", len(self._active_jobs))

                # Start processing more pending jobs
                asyncio.create_task(self._process_pending_jobs())

            except Exception as e:
                error_message = f"Consolidation job failed: {str(e)}"
                self.logger.error(error_message)
                traceback.print_exc()

                # Mark job as failed
                job.fail(error_message)

                # Move to completed jobs
                self._completed_jobs[job.job_id] = job
                del self._active_jobs[job.job_id]

                # Emit consolidation failed event
                await self.event_bus.emit(
                    MemoryConsolidationFailed(
                        job_id=job.job_id,
                        error=error_message,
                        processing_time=time.time() - start_time,
                    )
                )

                # Update metrics
                if self.metrics:
                    self.metrics.increment("memory_consolidation_jobs_failed")
                    self.metrics.gauge("memory_consolidation_active_jobs", len(self._active_jobs))

                # Start processing more pending jobs
                asyncio.create_task(self._process_pending_jobs())

    async def _consolidate_session_memories(self, job: ConsolidationJob) -> Dict[str, Any]:
        """
        Consolidate memories from a session.

        Args:
            job: Consolidation job

        Returns:
            Consolidation results
        """
        session_id = job.session_id
        if not session_id:
            raise ValueError("Session ID required for session-based consolidation")

        self.logger.info(f"Starting session-based consolidation for session {session_id}")

        # Get memories from working memory
        working_items = await self.working_memory.get_recent_items(session_id, limit=1000)

        # Skip if no memories to consolidate
        if not working_items:
            self.logger.info(f"No working memories found for session {session_id}")
            return {"memory_ids": [], "semantic_facts": [], "relationships": 0}

        # Group memories by context for better summarization
        context_groups = defaultdict(list)
        for item in working_items:
            context_id = item.context_id or "default"
            context_groups[context_id].append(item)

        # Consolidate each context group
        all_consolidated_ids = []
        all_semantic_facts = []
        total_relationships = 0

        for context_id, items in context_groups.items():
            # Skip empty groups
            if not items:
                continue

            self.logger.debug(f"Consolidating {len(items)} memories for context {context_id}")

            # Consolidate to episodic memory
            if MemoryType.EPISODIC in job.target_types:
                # Generate episodic memory
                episodic_ids = await self._consolidate_to_episodic(
                    items,
                    job.level,
                    context_id=context_id,
                    session_id=session_id,
                    user_id=job.user_id,
                )
                all_consolidated_ids.extend(episodic_ids)

                # Build relationships if graph store available
                if self.graph_store and self.auto_relationship_building:
                    # Get consolidated memories
                    episodic_memories = []
                    for memory_id in episodic_ids:
                        memory = await self.episodic_memory.retrieve(memory_id)
                        if memory:
                            episodic_memories.append(memory)

                    # Build relationships
                    context = {"session_id": session_id, "context_id": context_id}
                    relationships = await self._build_memory_relationships(
                        episodic_memories, context
                    )
                    total_relationships += relationships

            # Extract semantic memories if needed
            if MemoryType.SEMANTIC in job.target_types and self.auto_semantic_extraction:
                # Extract semantic memories from working memory
                semantic_facts = await self._extract_semantic_facts(
                    items, context={"session_id": session_id, "context_id": context_id}
                )

                # Store semantic facts
                for fact in semantic_facts:
                    try:
                        # Prepare storage parameters
                        fact_content = fact["content"]
                        confidence = fact.get("confidence", 0.8)
                        domain = fact.get("domain", "general")
                        concepts = fact.get("concepts", [])
                        entities = fact.get("entities", [])

                        # Store in semantic memory
                        memory_id = await self.semantic_memory.store(
                            fact_content,
                            concepts=concepts,
                            domain=domain,
                            entities=entities,
                            confidence=confidence,
                        )

                        # Add memory ID to fact
                        fact["memory_id"] = memory_id
                        all_semantic_facts.append(fact)

                    except Exception as e:
                        self.logger.error(f"Failed to store semantic fact: {str(e)}")

        # Clean up working memory
        await self.working_memory.cleanup_session(session_id)

        self.logger.info(
            f"Session consolidation completed: {len(all_consolidated_ids)} episodic memories, "
            f"{len(all_semantic_facts)} semantic facts, {total_relationships} relationships"
        )

        return {
            "memory_ids": all_consolidated_ids,
            "semantic_facts": all_semantic_facts,
            "relationships": total_relationships,
        }

    async def _consolidate_threshold_memories(self, job: ConsolidationJob) -> Dict[str, Any]:
        """
        Consolidate memories based on threshold (capacity).

        Args:
            job: Consolidation job

        Returns:
            Consolidation results
        """
        session_id = job.session_id
        if not session_id:
            raise ValueError("Session ID required for threshold-based consolidation")

        self.logger.info(f"Starting threshold-based consolidation for session {session_id}")

        # Get working memory stats
        stats = await self.working_memory.get_stats()

        # Check if we need to consolidate
        if stats.get("total_items", 0) < self.working_memory_threshold:
            self.logger.info(
                f"Working memory below threshold ({stats.get('total_items', 0)} < "
                f"{self.working_memory_threshold}), skipping consolidation"
            )
            return {"memory_ids": [], "semantic_facts": [], "relationships": 0}

        # Get oldest items to consolidate (50% of threshold)
        batch_size = self.working_memory_threshold // 2

        # Get memories from working memory by priority (lowest priority first)
        working_items = []
        for priority_level in range(stats.get("priority_levels", 3)):
            items_by_session = stats.get("items_by_session", {})
            session_items = items_by_session.get(session_id, 0)

            if session_items > 0:
                # Get items for this session and priority level
                session_query = f"session:{session_id}"
                session_items = await self.working_memory.search(session_query)

                # Sort by creation time (oldest first)
                session_items.sort(key=lambda x: x.metadata.created_at)

                # Add to working items (up to batch size)
                working_items.extend(session_items[: batch_size - len(working_items)])

                if len(working_items) >= batch_size:
                    break

        # Skip if no memories to consolidate
        if not working_items:
            self.logger.info(f"No working memories found to consolidate for session {session_id}")
            return {"memory_ids": [], "semantic_facts": [], "relationships": 0}

        # Group memories by context for better summarization
        context_groups = defaultdict(list)
        for item in working_items:
            context_id = item.context_id or "default"
            context_groups[context_id].append(item)

        # Consolidate each context group
        all_consolidated_ids = []
        all_semantic_facts = []
        total_relationships = 0

        for context_id, items in context_groups.items():
            # Skip empty groups
            if not items:
                continue

            self.logger.debug(f"Consolidating {len(items)} memories for context {context_id}")

            # Consolidate to episodic memory
            if MemoryType.EPISODIC in job.target_types:
                # Generate episodic memory
                episodic_ids = await self._consolidate_to_episodic(
                    items,
                    job.level,
                    context_id=context_id,
                    session_id=session_id,
                    user_id=job.user_id,
                )
                all_consolidated_ids.extend(episodic_ids)

                # Build relationships if graph store available
                if self.graph_store and self.auto_relationship_building:
                    # Get consolidated memories
                    episodic_memories = []
                    for memory_id in episodic_ids:
                        memory = await self.episodic_memory.retrieve(memory_id)
                        if memory:
                            episodic_memories.append(memory)

                    # Build relationships
                    context = {"session_id": session_id, "context_id": context_id}
                    relationships = await self._build_memory_relationships(
                        episodic_memories, context
                    )
                    total_relationships += relationships

            # Extract semantic memories if needed
            if MemoryType.SEMANTIC in job.target_types and self.auto_semantic_extraction:
                semantic_facts = await self._extract_semantic_facts(
                    items, context={"session_id": session_id, "context_id": context_id}
                )
                all_semantic_facts.extend(semantic_facts)

        # Get memory IDs of processed items
        processed_ids = [item.memory_id for item in working_items]

        # Clean up consolidated memories from working memory
        for memory_id in processed_ids:
            await self.working_memory.delete_item(memory_id)

        self.logger.info(
            f"Threshold consolidation completed: {len(all_consolidated_ids)} episodic memories, "
            f"{len(all_semantic_facts)} semantic facts, {total_relationships} relationships"
        )

        return {
            "memory_ids": all_consolidated_ids,
            "semantic_facts": all_semantic_facts,
            "relationships": total_relationships,
        }

    async def _consolidate_priority_memories(self, job: ConsolidationJob) -> Dict[str, Any]:
        """
        Consolidate specific memories by priority.

        Args:
            job: Consolidation job

        Returns:
            Consolidation results
        """
        memory_ids = job.memory_ids
        if not memory_ids:
            raise ValueError("Memory IDs required for priority-based consolidation")

        self.logger.info(f"Starting priority-based consolidation for {len(memory_ids)} memories")

        # Retrieve memories from source
        source_memories = []
        for memory_id in memory_ids:
            memory = None

            # Try to retrieve from source memory type
            if job.source_type == MemoryType.WORKING:
                memory = await self.working_memory.retrieve(memory_id)
            elif job.source_type == MemoryType.EPISODIC:
                memory = await self.episodic_memory.retrieve(memory_id)
            elif job.source_type == MemoryType.SEMANTIC:
                memory = await self.semantic_memory.retrieve(memory_id)

            if memory:
                source_memories.append(memory)

        # Skip if no memories to consolidate
        if not source_memories:
            self.logger.info(f"No source memories found to consolidate")
            return {"memory_ids": [], "semantic_facts": [], "relationships": 0}

        # Group memories by context for better consolidation
        context_groups = defaultdict(list)
        for item in source_memories:
            context_id = item.context_id or "default"
            context_groups[context_id].append(item)

        # Consolidate each context group
        all_consolidated_ids = []
        all_semantic_facts = []
        total_relationships = 0

        for context_id, items in context_groups.items():
            # Skip empty groups
            if not items:
                continue

            self.logger.debug(f"Consolidating {len(items)} memories for context {context_id}")

            # Extract session and user IDs from first memory
            session_id = items[0].session_id
            user_id = items[0].owner_id

            # Consolidate to episodic memory
            if MemoryType.EPISODIC in job.target_types:
                # Generate episodic memory
                episodic_ids = await self._consolidate_to_episodic(
                    items, job.level, context_id=context_id, session_id=session_id, user_id=user_id
                )
                all_consolidated_ids.extend(episodic_ids)

                # Build relationships if graph store available
                if self.graph_store and self.auto_relationship_building:
                    # Get consolidated memories
                    episodic_memories = []
                    for memory_id in episodic_ids:
                        memory = await self.episodic_memory.retrieve(memory_id)
                        if memory:
                            episodic_memories.append(memory)

                    # Build relationships
                    context = {"session_id": session_id, "context_id": context_id}
                    relationships = await self._build_memory_relationships(
                        episodic_memories, context
                    )
                    total_relationships += relationships

            # Extract semantic memories if needed
            if MemoryType.SEMANTIC in job.target_types and self.auto_semantic_extraction:
                semantic_facts = await self._extract_semantic_facts(
                    items, context={"session_id": session_id, "context_id": context_id}
                )
                all_semantic_facts.extend(semantic_facts)

        self.logger.info(
            f"Priority consolidation completed: {len(all_consolidated_ids)} episodic memories, "
            f"{len(all_semantic_facts)} semantic facts, {total_relationships} relationships"
        )

        return {
            "memory_ids": all_consolidated_ids,
            "semantic_facts": all_semantic_facts,
            "relationships": total_relationships,
        }

    async def _consolidate_temporal_memories(self, job: ConsolidationJob) -> Dict[str, Any]:
        """
        Consolidate memories based on temporal scheduling.

        Args:
            job: Consolidation job

        Returns:
            Consolidation results
        """
        self.logger.info("Starting temporal consolidation")

        # For temporal consolidation, we look at all active sessions
        # Get recent sessions with activity
        active_sessions = []

        # This is a simplified approach - in a real system, you would
        # query the session manager or other components to get active sessions

        # For now, inspect working memory stats to find active sessions
        stats = await self.working_memory.get_stats()
        active_sessions = list(stats.get("items_by_session", {}).keys())

        if not active_sessions:
            self.logger.info("No active sessions found for temporal consolidation")
            return {"memory_ids": [], "semantic_facts": [], "relationships": 0}

        # Consolidate each session
        all_consolidated_ids = []
        all_semantic_facts = []
        total_relationships = 0

        for session_id in active_sessions:
            self.logger.debug(f"Temporal consolidation for session {session_id}")

            # Get working memory items for this session
            working_items = await self.working_memory.get_recent_items(session_id, limit=100)

            # Skip if no items
            if not working_items:
                continue

            # Group by context
            context_groups = defaultdict(list)
            for item in working_items:
                context_id = item.context_id or "default"
                context_groups[context_id].append(item)

            # Process each context group
            for context_id, items in context_groups.items():
                # Skip if too few items
                if len(items) < 3:  # Require at least 3 items to consolidate
                    continue

                # Extract user ID from first memory
                user_id = items[0].owner_id

                # Consolidate to episodic memory
                if MemoryType.EPISODIC in job.target_types:
                    episodic_ids = await self._consolidate_to_episodic(
                        items,
                        job.level,
                        context_id=context_id,
                        session_id=session_id,
                        user_id=user_id,
                    )
                    all_consolidated_ids.extend(episodic_ids)

                    # Build relationships
                    if self.graph_store and self.auto_relationship_building:
                        episodic_memories = []
                        for memory_id in episodic_ids:
                            memory = await self.episodic_memory.retrieve(memory_id)
                            if memory:
                                episodic_memories.append(memory)

                        context = {"session_id": session_id, "context_id": context_id}
                        relationships = await self._build_memory_relationships(
                            episodic_memories, context
                        )
                        total_relationships += relationships

                # Extract semantic memories if needed
                if MemoryType.SEMANTIC in job.target_types and self.auto_semantic_extraction:
                    semantic_facts = await self._extract_semantic_facts(
                        items, context={"session_id": session_id, "context_id": context_id}
                    )
                    all_semantic_facts.extend(semantic_facts)

                # Remove consolidated items from working memory
                for item in items:
                    await self.working_memory.delete_item(item.memory_id)

        self.logger.info(
            f"Temporal consolidation completed: {len(all_consolidated_ids)} episodic memories, "
            f"{len(all_semantic_facts)} semantic facts, {total_relationships} relationships"
        )

        return {
            "memory_ids": all_consolidated_ids,
            "semantic_facts": all_semantic_facts,
            "relationships": total_relationships,
        }

    async def _consolidate_to_episodic(
        self,
        source_memories: List[MemoryItem],
        level: ConsolidationLevel,
        context_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Consolidate source memories into episodic memory.

        Args:
            source_memories: Source memory items
            level: Consolidation processing level
            context_id: Context identifier
            session_id: Session identifier
            user_id: User identifier

        Returns:
            List of created episodic memory IDs
        """
        if not source_memories:
            return []

        consolidated_ids = []

        # Basic consolidation simply transfers memories without processing
        if level == ConsolidationLevel.BASIC:
            for memory in source_memories:
                # Create episodic memory from working memory
                episodic_id = await self.episodic_memory.store(
                    data=memory.content,
                    user_id=user_id or memory.owner_id,
                    session_id=session_id or memory.session_id,
                    context_id=context_id or memory.context_id,
                    tags=memory.metadata.tags if memory.metadata else set(),
                )
                consolidated_ids.append(episodic_id)

        # Standard and higher consolidation levels perform summarization
        else:
            # Sort memories by time
            source_memories.sort(
                key=lambda x: x.metadata.created_at if x.metadata else datetime.now(timezone.utc)
            )

            # Group by similar time windows (5-minute intervals)
            time_windows = defaultdict(list)
            for memory in source_memories:
                created_time = (
                    memory.metadata.created_at if memory.metadata else datetime.now(timezone.utc)
                )
                # Create a 5-minute window key
                window_key = created_time.strftime("%Y-%m-%d %H:%M")
                window_key = window_key[:-1] + "0"  # Round to nearest 5 minutes
                time_windows[window_key].append(memory)

            # Process each time window
            for window_key, window_memories in time_windows.items():
                # Skip single-memory windows for efficiency
                if len(window_memories) == 1 and level != ConsolidationLevel.COMPREHENSIVE:
                    # Just transfer the memory
                    memory = window_memories[0]
                    episodic_id = await self.episodic_memory.store(
                        data=memory.content,
                        user_id=user_id or memory.owner_id,
                        session_id=session_id or memory.session_id,
                        context_id=context_id or memory.context_id,
                        tags=memory.metadata.tags if memory.metadata else set(),
                    )
                    consolidated_ids.append(episodic_id)
                    continue

                # Generate summary for this time window
                summary = await self._generate_memory_summary(window_memories, level)

                # Extract metadata
                all_tags = set()
                importance = 0.0
                emotion = None

                for memory in window_memories:
                    if memory.metadata:
                        # Merge tags
                        if memory.metadata.tags:
                            all_tags.update(memory.metadata.tags)

                        # Use maximum importance
                        importance = max(importance, memory.metadata.importance)

                        # Extract emotion if available
                        if (
                            memory.metadata.custom_metadata
                            and "emotion" in memory.metadata.custom_metadata
                        ):
                            emotion = memory.metadata.custom_metadata["emotion"]

                # Normalize importance
                importance = min(
                    1.0, importance * 1.1
                )  # Slightly boost importance for consolidated memories

                # Store consolidated episodic memory
                episodic_id = await self.episodic_memory.store(
                    data=summary,
                    user_id=user_id or window_memories[0].owner_id,
                    session_id=session_id or window_memories[0].session_id,
                    context_id=context_id or window_memories[0].context_id,
                    tags=all_tags,
                    importance=importance,
                    emotion=emotion,
                )
                consolidated_ids.append(episodic_id)

        return consolidated_ids

    async def _generate_memory_summary(
        self, memories: List[MemoryItem], level: ConsolidationLevel
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a summary of memories for consolidation.

        Args:
            memories: Source memory items
            level: Consolidation processing level

        Returns:
            Summary content (dictionary or string)
        """
        # For basic consolidation or without LLM, create a simple summary
        if level == ConsolidationLevel.BASIC or not self.model_router:
            # Simple dictionary summary
            memory_contents = []
            for memory in memories:
                if isinstance(memory.content, dict):
                    memory_contents.append(memory.content)
                else:
                    memory_contents.append({"content": str(memory.content)})

            return {
                "summary": f"Consolidated memory of {len(memories)} items",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "items": memory_contents,
            }

        # Use LLM for more advanced summarization
        try:
            # Prepare memory content for LLM
            memory_texts = []
            for i, memory in enumerate(memories, 1):
                if isinstance(memory.content, dict):
                    # Format dictionary content
                    if "text" in memory.content:
                        memory_texts.append(f"{i}. {memory.content['text']}")
                    elif "content" in memory.content:
                        memory_texts.append(f"{i}. {memory.content['content']}")
                    else:
                        # Stringify dictionary
                        content_str = json.dumps(memory.content)
                        memory_texts.append(f"{i}. {content_str}")
                elif isinstance(memory.content, str):
                    memory_texts.append(f"{i}. {memory.content}")
                else:
                    memory_texts.append(f"{i}. {str(memory.content)}")

            memory_content = "\n".join(memory_texts)

            # Different prompts based on consolidation level
            if level == ConsolidationLevel.STANDARD:
                prompt = (
                    "Summarize the following memory items into a coherent episodic memory. "
                    "Focus on key information, interactions, and outcomes. "
                    "Memory items:\n\n"
                    f"{memory_content}\n\n"
                    "Summary:"
                )

                # Generate summary
                summary = await self.model_router.generate_text(prompt)

                return {
                    "summary": summary,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "memory_count": len(memories),
                }

            elif level == ConsolidationLevel.DEEP:
                prompt = (
                    "Create a detailed consolidated memory from the following items. "
                    "Extract key information, identify patterns, and highlight important insights. "
                    "Include emotional content if present. Structure your response as a JSON object with: "
                    "summary, key_points (list), entities (list), and emotional_tone fields.\n\n"
                    f"Memory items:\n{memory_content}\n\n"
                    "Consolidated memory (JSON format):"
                )

                # Generate detailed summary
                summary_text = await self.model_router.generate_text(prompt)

                # Parse JSON response (with fallback)
                try:
                    summary = json.loads(summary_text)
                except json.JSONDecodeError:
                    # Fallback to text format
                    summary = {
                        "summary": summary_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "memory_count": len(memories),
                    }

                return summary

            elif level == ConsolidationLevel.COMPREHENSIVE:
                prompt = (
                    "Create a comprehensive consolidated memory from the following items. "
                    "Extract key information, identify patterns, relationships, and important insights. "
                    "Include emotional content, entities, concepts, and actions. "
                    "Structure your response as a JSON object with: summary, key_points (list), "
                    "entities (list), concepts (list), actions (list), emotional_tone, and significance (1-10).\n\n"
                    f"Memory items:\n{memory_content}\n\n"
                    "Comprehensive consolidated memory (JSON format):"
                )

                # Generate comprehensive summary
                summary_text = await self.model_router.generate_text(prompt)

                # Parse JSON response (with fallback)
                try:
                    summary = json.loads(summary_text)
                except json.JSONDecodeError:
                    # Fallback to text format
                    summary = {
                        "summary": summary_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "memory_count": len(memories),
                    }

                # Add source count
                summary["memory_count"] = len(memories)
                summary["timestamp"] = datetime.now(timezone.utc).isoformat()

                return summary

            else:
                # Default to standard summary
                prompt = (
                    "Summarize the following memory items into a coherent episodic memory. "
                    "Focus on key information, interactions, and outcomes. "
                    "Memory items:\n\n"
                    f"{memory_content}\n\n"
                    "Summary:"
                )

                summary = await self.model_router.generate_text(prompt)

                return {
                    "summary": summary,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "memory_count": len(memories),
                }

        except Exception as e:
            self.logger.error(f"Failed to generate summary with LLM: {str(e)}")

            # Fallback to simple summary
            return {
                "summary": f"Consolidated memory of {len(memories)} items",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "items": [str(m.content) for m in memories],
            }

    async def _extract_semantic_facts(
        self, memories: List[MemoryItem], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract semantic facts from memories.

        Args:
            memories: Source memory items
            context: Optional context information

        Returns:
            List of semantic facts
        """
        if not memories:
            return []

        if not self.model_router:
            self.logger.warning("Cannot extract semantic facts without ModelRouter")
            return []

        try:
            # Prepare memory content for LLM
            memory_texts = []
            for i, memory in enumerate(memories, 1):
                if isinstance(memory.content, dict):
                    # Format dictionary content
                    if "text" in memory.content:
                        memory_texts.append(f"{i}. {memory.content['text']}")
                    elif "content" in memory.content:
                        memory_texts.append(f"{i}. {memory.content['content']}")
                    elif "summary" in memory.content:
                        memory_texts.append(f"{i}. {memory.content['summary']}")
                    else:
                        # Stringify dictionary
                        content_str = json.dumps(memory.content)
                        memory_texts.append(f"{i}. {content_str}")
                elif isinstance(memory.content, str):
                    memory_texts.append(f"{i}. {memory.content}")
                else:
                    memory_texts.append(f"{i}. {str(memory.content)}")

            memory_content = "\n".join(memory_texts)

            # Add context information
            context_str = ""
            if context:
                context_parts = []
                for key, value in context.items():
                    context_parts.append(f"{key}: {value}")
                context_str = "Context: " + ", ".join(context_parts) + "\n\n"

            # Create prompt for semantic extraction
            prompt = (
                f"{context_str}Extract factual knowledge from the following memories. "
                "Identify generalizable facts, concepts, and relationships. "
                "Focus on information that is likely to be useful in future interactions. "
                "Return a list of 1-5 semantic facts in JSON format, where each fact is an object with: "
                "content (the fact itself), confidence (0.0-1.0), domain (e.g., 'personal', 'general', 'domain-specific'), "
                "concepts (list of related concepts), and entities (list of entities mentioned).\n\n"
                f"Memories:\n{memory_content}\n\n"
                "Extracted facts (JSON array):"
            )

            # Generate facts
            facts_text = await self.model_router.generate_text(prompt)

            # Parse JSON response (with fallback)
            try:
                facts = json.loads(facts_text)
                if not isinstance(facts, list):
                    facts = [facts]
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse semantic facts JSON: {facts_text}")
                # Fallback to simple fact
                facts = [
                    {
                        "content": facts_text,
                        "confidence": 0.5,
                        "domain": "general",
                        "concepts": [],
                        "entities": [],
                    }
                ]

            # Validate and clean up facts
            valid_facts = []
            for fact in facts:
                if not isinstance(fact, dict) or "content" not in fact:
                    continue

                # Ensure all required fields
                if "confidence" not in fact:
                    fact["confidence"] = 0.7
                if "domain" not in fact:
                    fact["domain"] = "general"
                if "concepts" not in fact:
                    fact["concepts"] = []
                if "entities" not in fact:
                    fact["entities"] = []

                # Add to valid facts
                valid_facts.append(fact)

            return valid_facts

        except Exception as e:
            self.logger.error(f"Failed to extract semantic facts: {str(e)}")
            return []

    async def _build_memory_relationships(
        self, memories: List[MemoryItem], context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Build relationships between memories in the memory graph.

        Args:
            memories: Memory items
            context: Optional context information

        Returns:
            Number of relationships created
        """
        if not memories or not self.graph_store:
            return 0

        try:
            # Count created relationships
            relationship_count = 0

            # First, ensure all memories are in the graph
            memory_nodes = {}
            for memory in memories:
                # Check if memory already has a node
                node = await self.graph_store.get_node_by_memory_id(memory.memory_id)

                if not node:
                    # Create node for memory
                    properties = {
                        "created_at": (
                            memory.metadata.created_at.isoformat()
                            if memory.metadata
                            else datetime.now(timezone.utc).isoformat()
                        ),
                        "memory_type": memory.memory_type.value,
                        "importance": memory.metadata.importance if memory.metadata else 0.5,
                        "owner_id": memory.owner_id,
                        "session_id": memory.session_id,
                        "context_id": memory.context_id,
                    }

                    # Add content summary
                    if isinstance(memory.content, dict) and "summary" in memory.content:
                        properties["content_summary"] = memory.content["summary"]
                    elif isinstance(memory.content, str):
                        properties["content_summary"] = memory.content[:100]
                    else:
                        properties["content_summary"] = str(memory.content)[:100]

                    # Create node
                    labels = (
                        memory.metadata.tags if memory.metadata and memory.metadata.tags else set()
                    )
                    node_id = await self.graph_store.add_node(
                        node_type=GraphNodeType.MEMORY,
                        properties=properties,
                        labels=labels,
                        memory_id=memory.memory_id,
                    )

                    # Get created node
                    node = await self.graph_store.get_node(node_id)

                if node:
                    memory_nodes[memory.memory_id] = node

            # If we have a model router, use it to identify semantic relationships
            if self.model_router and len(memories) > 1:
                # Prepare memory content for LLM
                memory_texts = {}
                for memory in memories:
                    memory_id = memory.memory_id

                    if isinstance(memory.content, dict):
                        if "summary" in memory.content:
                            memory_texts[memory_id] = memory.content["summary"]
                        elif "content" in memory.content:
                            memory_texts[memory_id] = memory.content["content"]
                        elif "text" in memory.content:
                            memory_texts[memory_id] = memory.content["text"]
                        else:
                            memory_texts[memory_id] = json.dumps(memory.content)
                    elif isinstance(memory.content, str):
                        memory_texts[memory_id] = memory.content
                    else:
                        memory_texts[memory_id] = str(memory.content)

                # Format for LLM
                memory_descriptions = []
                for i, (memory_id, text) in enumerate(memory_texts.items(), 1):
                    memory_descriptions.append(f"Memory {i} (ID: {memory_id}): {text[:200]}...")

                memory_content = "\n".join(memory_descriptions)

                # Create prompt for relationship extraction
                relationship_prompt = (
                    "Identify relationships between the following memories. "
                    "Return a JSON array of relationship objects, where each object has: "
                    "source_id, target_id, relationship_type (one of: is_a, has_a, part_of, before, after, during, "
                    "located_at, causes, related_to, similar_to, opposite_of, instance_of, referenced_by, depends_on, "
                    "interacts_with), weight (0.0-1.0, relationship strength), and description (short explanation).\n\n"
                    f"Memories:\n{memory_content}\n\n"
                    "Relationships (JSON array):"
                )

                # Process with LLM
                relationships_response = await self.llm_provider.generate_text(
                    prompt=relationship_prompt, max_tokens=2048, temperature=0.1
                )

                if relationships_response and "text" in relationships_response:
                    try:
                        relationships = json.loads(relationships_response["text"].strip())
                        # Process relationships into graph edges
                        for rel in relationships:
                            source_id = rel.get("source_id")
                            target_id = rel.get("target_id")
                            relationship_type = rel.get("relationship_type", "related_to")
                            weight = rel.get("weight", 0.5)

                            if source_id in memory_nodes and target_id in memory_nodes:
                                source_node = memory_nodes[source_id]
                                target_node = memory_nodes[target_id]

                                await self.graph_store.add_relationship(
                                    source_node=source_node,
                                    target_node=target_node,
                                    relationship_type=relationship_type,
                                    weight=weight,
                                    properties={"description": rel.get("description", "")},
                                )
                                relationship_count += 1

                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse relationships JSON: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"Error processing relationships: {str(e)}")

            return relationship_count

        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}")
            return 0
