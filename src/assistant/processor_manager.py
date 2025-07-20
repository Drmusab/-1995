"""
ProcessorManager - Centralized Processor Lifecycle Management

This component addresses the critical architectural flaw identified in the integration analysis
where the core engine lacks centralized processor management.

Author: Integration Analysis Response
Created: 2025-01-29
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Any, Union, Protocol
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import time
from concurrent.futures import ThreadPoolExecutor

from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged, ProcessingStarted, ProcessingCompleted,
    ProcessingError, ModalityProcessingStarted, ModalityProcessingCompleted
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.observability.logging.config import get_logger

class ProcessorState(Enum):
    """Processor lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ProcessorType(Enum):
    """Types of processors in the system."""
    NATURAL_LANGUAGE = "natural_language"
    SPEECH = "speech"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

@dataclass
class ProcessorHealth:
    """Health information for a processor."""
    processor_id: str
    state: ProcessorState
    last_ping: float
    error_count: int
    success_count: int
    avg_processing_time: float
    memory_usage: float = 0.0
    cpu_usage: float = 0.0

class ProcessorInterface(Protocol):
    """Interface that all processors must implement."""
    
    @property
    def processor_id(self) -> str:
        """Unique identifier for the processor."""
        ...
    
    @property 
    def processor_type(self) -> ProcessorType:
        """Type of processor."""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the processor with configuration."""
        ...
    
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Process input data and return results."""
        ...
    
    async def health_check(self) -> ProcessorHealth:
        """Return health status of the processor."""
        ...
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the processor."""
        ...

class ProcessorProxy:
    """Proxy wrapper for processors to add monitoring and error handling."""
    
    def __init__(self, processor: ProcessorInterface, event_bus: EventBus, logger: logging.Logger):
        self.processor = processor
        self.event_bus = event_bus
        self.logger = logger
        self.state = ProcessorState.UNINITIALIZED
        self.error_count = 0
        self.success_count = 0
        self.processing_times = []
        self.last_health_check = 0.0
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize processor with monitoring."""
        try:
            self.state = ProcessorState.INITIALIZING
            await self.processor.initialize(config)
            self.state = ProcessorState.READY
            self.logger.info(f"Processor {self.processor.processor_id} initialized successfully")
        except Exception as e:
            self.state = ProcessorState.ERROR
            self.error_count += 1
            self.logger.error(f"Failed to initialize processor {self.processor.processor_id}: {e}")
            raise
    
    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """Process with timing and error monitoring."""
        if self.state != ProcessorState.READY:
            raise RuntimeError(f"Processor {self.processor.processor_id} not ready (state: {self.state})")
        
        start_time = time.time()
        self.state = ProcessorState.BUSY
        
        try:
            # Emit processing started event
            await self.event_bus.emit(ModalityProcessingStarted(
                processor_id=self.processor.processor_id,
                processor_type=self.processor.processor_type.value,
                input_size=len(str(input_data)) if input_data else 0
            ))
            
            result = await self.processor.process(input_data, context)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:  # Keep last 100 measurements
                self.processing_times.pop(0)
            
            self.success_count += 1
            self.state = ProcessorState.READY
            
            # Emit processing completed event
            await self.event_bus.emit(ModalityProcessingCompleted(
                processor_id=self.processor.processor_id,
                processor_type=self.processor.processor_type.value,
                processing_time=processing_time,
                success=True
            ))
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.state = ProcessorState.ERROR
            
            # Emit processing error event
            await self.event_bus.emit(ProcessingError(
                processor_id=self.processor.processor_id,
                error_type=type(e).__name__,
                error_message=str(e),
                processing_time=time.time() - start_time
            ))
            
            self.logger.error(f"Processing error in {self.processor.processor_id}: {e}")
            raise
        finally:
            if self.state == ProcessorState.BUSY:
                self.state = ProcessorState.READY
    
    async def health_check(self) -> ProcessorHealth:
        """Get health status with proxy metrics."""
        try:
            base_health = await self.processor.health_check()
            
            # Update with proxy metrics
            base_health.error_count = self.error_count
            base_health.success_count = self.success_count
            base_health.avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
            base_health.state = self.state
            base_health.last_ping = time.time()
            
            self.last_health_check = time.time()
            return base_health
            
        except Exception as e:
            self.logger.error(f"Health check failed for {self.processor.processor_id}: {e}")
            return ProcessorHealth(
                processor_id=self.processor.processor_id,
                state=ProcessorState.ERROR,
                last_ping=time.time(),
                error_count=self.error_count,
                success_count=self.success_count,
                avg_processing_time=0.0
            )

class ProcessorManager:
    """
    Centralized processor lifecycle management.
    
    This component addresses the architectural flaw where processors
    were not centrally managed in the core engine.
    """
    
    def __init__(self, config: ConfigLoader, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.logger = get_logger(__name__)
        
        self.processors: Dict[str, ProcessorProxy] = {}
        self.processor_configs: Dict[str, Dict[str, Any]] = {}
        self.health_check_interval = config.get("processor_manager.health_check_interval", 30)
        self.max_retries = config.get("processor_manager.max_retries", 3)
        
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_flag = False
        
    async def register_processor(
        self, 
        processor: ProcessorInterface, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a processor with the manager."""
        processor_id = processor.processor_id
        
        if processor_id in self.processors:
            raise ValueError(f"Processor {processor_id} already registered")
        
        # Wrap processor in proxy for monitoring
        proxy = ProcessorProxy(processor, self.event_bus, self.logger)
        self.processors[processor_id] = proxy
        
        # Store configuration
        if config:
            self.processor_configs[processor_id] = config
        
        # Initialize processor
        try:
            await proxy.initialize(config or {})
            self.logger.info(f"Registered and initialized processor: {processor_id}")
            
            # Emit health change event
            await self.event_bus.emit(ComponentHealthChanged(
                component="processor_manager",
                component_id=processor_id,
                health_status="healthy",
                details=f"Processor {processor_id} registered successfully"
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processor {processor_id}: {e}")
            del self.processors[processor_id]
            raise
    
    async def get_processor(self, processor_id: str) -> Optional[ProcessorProxy]:
        """Get a processor by ID."""
        return self.processors.get(processor_id)
    
    async def get_processors_by_type(self, processor_type: ProcessorType) -> List[ProcessorProxy]:
        """Get all processors of a specific type."""
        return [
            proxy for proxy in self.processors.values()
            if proxy.processor.processor_type == processor_type
        ]
    
    async def process_with_processor(
        self, 
        processor_id: str, 
        input_data: Any, 
        context: Optional[Dict] = None
    ) -> Any:
        """Process data with a specific processor."""
        processor = self.processors.get(processor_id)
        if not processor:
            raise ValueError(f"Processor {processor_id} not found")
        
        return await processor.process(input_data, context)
    
    async def process_with_type(
        self, 
        processor_type: ProcessorType, 
        input_data: Any, 
        context: Optional[Dict] = None,
        load_balance: bool = True
    ) -> Any:
        """Process data with any processor of the specified type."""
        processors = await self.get_processors_by_type(processor_type)
        
        if not processors:
            raise ValueError(f"No processors of type {processor_type.value} available")
        
        # Simple load balancing - use processor with lowest error rate
        if load_balance and len(processors) > 1:
            best_processor = min(
                processors, 
                key=lambda p: p.error_count / max(p.success_count, 1)
            )
        else:
            best_processor = processors[0]
        
        return await best_processor.process(input_data, context)
    
    async def get_all_health(self) -> Dict[str, ProcessorHealth]:
        """Get health status of all processors."""
        health_status = {}
        
        for processor_id, proxy in self.processors.items():
            try:
                health = await proxy.health_check()
                health_status[processor_id] = health
            except Exception as e:
                self.logger.error(f"Health check failed for {processor_id}: {e}")
                health_status[processor_id] = ProcessorHealth(
                    processor_id=processor_id,
                    state=ProcessorState.ERROR,
                    last_ping=time.time(),
                    error_count=proxy.error_count,
                    success_count=proxy.success_count,
                    avg_processing_time=0.0
                )
        
        return health_status
    
    async def restart_processor(self, processor_id: str) -> None:
        """Restart a specific processor."""
        if processor_id not in self.processors:
            raise ValueError(f"Processor {processor_id} not found")
        
        proxy = self.processors[processor_id]
        config = self.processor_configs.get(processor_id, {})
        
        try:
            # Shutdown existing processor
            await proxy.processor.shutdown()
            
            # Reinitialize
            await proxy.initialize(config)
            
            self.logger.info(f"Restarted processor: {processor_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to restart processor {processor_id}: {e}")
            raise
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._health_check_task:
            return  # Already running
        
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        self.logger.info("Started processor health monitoring")
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_flag:
            try:
                health_status = await self.get_all_health()
                
                # Check for unhealthy processors
                for processor_id, health in health_status.items():
                    if health.state == ProcessorState.ERROR:
                        # Attempt restart if error count is high
                        if health.error_count > self.max_retries:
                            self.logger.warning(f"Attempting to restart unhealthy processor: {processor_id}")
                            try:
                                await self.restart_processor(processor_id)
                            except Exception as e:
                                self.logger.error(f"Failed to restart processor {processor_id}: {e}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def shutdown(self) -> None:
        """Shutdown all processors and stop monitoring."""
        self._shutdown_flag = True
        
        # Stop health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all processors
        shutdown_tasks = []
        for processor_id, proxy in self.processors.items():
            shutdown_tasks.append(self._shutdown_processor(processor_id, proxy))
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.processors.clear()
        self.logger.info("ProcessorManager shutdown complete")
    
    async def _shutdown_processor(self, processor_id: str, proxy: ProcessorProxy) -> None:
        """Shutdown a single processor."""
        try:
            await proxy.processor.shutdown()
            self.logger.info(f"Shutdown processor: {processor_id}")
        except Exception as e:
            self.logger.error(f"Error shutting down processor {processor_id}: {e}")
    
    @asynccontextmanager
    async def managed_processor(self, processor: ProcessorInterface, config: Optional[Dict] = None):
        """Context manager for temporary processor registration."""
        await self.register_processor(processor, config)
        try:
            yield self.processors[processor.processor_id]
        finally:
            if processor.processor_id in self.processors:
                await self._shutdown_processor(processor.processor_id, self.processors[processor.processor_id])
                del self.processors[processor.processor_id]