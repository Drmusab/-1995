"""
Advanced GPU Profiling and Monitoring System
Author: Drmusab
Last Modified: 2025-07-20

This module provides comprehensive GPU resource monitoring, profiling, and optimization
for the AI assistant, supporting multiple GPU frameworks (PyTorch, TensorFlow, CUDA),
with real-time metrics collection, bottleneck detection, and seamless integration with
the core event and monitoring systems.
"""

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import asyncio
import psutil

# Optional GPU framework imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

# Core imports (with graceful fallbacks for broken dependencies)
try:
    from src.core.config.loader import ConfigLoader
    from src.core.dependency_injection import Container
    from src.core.events.event_bus import EventBus
    from src.observability.logging.config import get_logger
    from src.observability.monitoring.metrics import MetricsCollector
except ImportError as e:
    # Create minimal fallbacks for standalone testing
    class MockLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

        def error(self, msg):
            print(f"ERROR: {msg}")

        def warning(self, msg):
            print(f"WARNING: {msg}")

    def get_logger(name):
        return MockLogger()


class GPUFramework(Enum):
    """Supported GPU frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUDA = "cuda"


@dataclass
class GPUMetrics:
    """GPU performance metrics."""

    utilization_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    temperature_c: float = 0.0
    power_usage_w: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GPUProfileSession:
    """GPU profiling session data."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    framework: GPUFramework = GPUFramework.PYTORCH
    metrics_history: List[GPUMetrics] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    average_utilization: float = 0.0
    total_operations: int = 0


class SimpleGPUProfiler:
    """
    Simplified GPU profiler for the AI assistant.

    Provides basic GPU monitoring and profiling capabilities
    with fallback support when GPU libraries are unavailable.
    """

    def __init__(self, container: Optional[Any] = None):
        """Initialize the GPU profiler."""
        self.logger = get_logger(__name__)
        self.container = container

        # Configuration
        self.enable_profiling = True
        self.collection_interval = 1.0  # seconds
        self.max_session_history = 100

        # GPU detection
        self.gpu_available = self._detect_gpu()
        self.framework = self._determine_framework()

        # Session management
        self.active_sessions: Dict[str, GPUProfileSession] = {}
        self.session_history: List[GPUProfileSession] = []

        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        self.logger.info(f"GPU Profiler initialized (GPU available: {self.gpu_available})")

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        if TORCH_AVAILABLE:
            return torch.cuda.is_available()
        elif TF_AVAILABLE:
            return len(tf.config.list_physical_devices("GPU")) > 0
        else:
            # Basic detection fallback
            try:
                import subprocess

                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
                return result.returncode == 0
            except:
                return False

    def _determine_framework(self) -> GPUFramework:
        """Determine the best available GPU framework."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return GPUFramework.PYTORCH
        elif TF_AVAILABLE:
            return GPUFramework.TENSORFLOW
        else:
            return GPUFramework.CUDA

    async def initialize(self):
        """Initialize the GPU profiler."""
        if self.enable_profiling and self.gpu_available:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("GPU monitoring started")
        else:
            self.logger.info("GPU profiling disabled or GPU not available")

    async def start_profiling_session(
        self, session_id: str, framework: Optional[GPUFramework] = None
    ) -> GPUProfileSession:
        """Start a new profiling session."""
        if not self.gpu_available:
            self.logger.warning("GPU not available for profiling")
            return None

        session = GPUProfileSession(
            session_id=session_id,
            start_time=datetime.now(timezone.utc),
            framework=framework or self.framework,
        )

        self.active_sessions[session_id] = session
        self.logger.info(f"Started GPU profiling session: {session_id}")

        return session

    async def end_profiling_session(self, session_id: str) -> Optional[GPUProfileSession]:
        """End a profiling session."""
        session = self.active_sessions.pop(session_id, None)
        if not session:
            return None

        session.end_time = datetime.now(timezone.utc)

        # Calculate summary statistics
        if session.metrics_history:
            session.average_utilization = sum(
                m.utilization_percent for m in session.metrics_history
            ) / len(session.metrics_history)
            session.peak_memory_mb = max(m.memory_used_mb for m in session.metrics_history)

        # Store in history
        self.session_history.append(session)

        # Maintain history size
        if len(self.session_history) > self.max_session_history:
            self.session_history = self.session_history[-self.max_session_history :]

        self.logger.info(f"Ended GPU profiling session: {session_id}")
        return session

    def get_current_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        if not self.gpu_available:
            return GPUMetrics()

        try:
            if self.framework == GPUFramework.PYTORCH and TORCH_AVAILABLE:
                return self._get_pytorch_metrics()
            elif self.framework == GPUFramework.TENSORFLOW and TF_AVAILABLE:
                return self._get_tensorflow_metrics()
            else:
                return self._get_basic_metrics()
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
            return GPUMetrics()

    def _get_pytorch_metrics(self) -> GPUMetrics:
        """Get GPU metrics using PyTorch."""
        if not torch.cuda.is_available():
            return GPUMetrics()

        device = torch.cuda.current_device()

        # Memory statistics
        memory_stats = torch.cuda.memory_stats(device)
        memory_used = memory_stats.get("allocated_bytes.all.current", 0) / (1024**2)
        memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**2)

        # Utilization (basic approximation)
        utilization = min(100.0, (memory_used / memory_total) * 100.0) if memory_total > 0 else 0.0

        return GPUMetrics(
            utilization_percent=utilization,
            memory_used_mb=memory_used,
            memory_total_mb=memory_total,
            memory_percent=(memory_used / memory_total * 100.0) if memory_total > 0 else 0.0,
            temperature_c=0.0,  # Not available through PyTorch
            power_usage_w=0.0,  # Not available through PyTorch
        )

    def _get_tensorflow_metrics(self) -> GPUMetrics:
        """Get GPU metrics using TensorFlow."""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                return GPUMetrics()

            # Basic memory info (TensorFlow doesn't provide detailed metrics easily)
            return GPUMetrics(
                utilization_percent=0.0,  # Would need nvidia-ml-py for detailed stats
                memory_used_mb=0.0,
                memory_total_mb=0.0,
                memory_percent=0.0,
                temperature_c=0.0,
                power_usage_w=0.0,
            )
        except Exception:
            return GPUMetrics()

    def _get_basic_metrics(self) -> GPUMetrics:
        """Get basic GPU metrics as fallback."""
        # This would ideally use nvidia-ml-py or similar
        # For now, return empty metrics
        return GPUMetrics()

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                current_metrics = self.get_current_metrics()

                # Update active sessions
                for session in self.active_sessions.values():
                    session.metrics_history.append(current_metrics)
                    session.total_operations += 1

                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in GPU monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)

    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        return {
            "gpu_available": self.gpu_available,
            "framework": self.framework.value if self.framework else None,
            "active_sessions": len(self.active_sessions),
            "total_sessions": len(self.session_history),
            "pytorch_available": TORCH_AVAILABLE,
            "tensorflow_available": TF_AVAILABLE,
            "current_metrics": self.get_current_metrics().__dict__ if self.gpu_available else {},
        }

    @contextmanager
    def profile_context(self, session_id: str):
        """Context manager for profiling."""
        session = None
        try:
            session = (
                asyncio.create_task(self.start_profiling_session(session_id)).result()
                if asyncio.get_event_loop().is_running()
                else None
            )
            yield session
        finally:
            if session:
                try:
                    (
                        asyncio.create_task(self.end_profiling_session(session_id)).result()
                        if asyncio.get_event_loop().is_running()
                        else None
                    )
                except:
                    pass

    async def shutdown(self):
        """Shutdown the GPU profiler."""
        self.logger.info("Shutting down GPU profiler...")
        self._shutdown_event.set()

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # End all active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_profiling_session(session_id)

        self.logger.info("GPU profiler shutdown complete")


# Alias for backward compatibility
GPUProfiler = SimpleGPUProfiler
