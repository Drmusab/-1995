"""
Performance tests for validating optimization improvements.
Author: Assistant
Created: 2025-01-17

This module contains benchmark tests to measure the performance improvements
made to the AI Assistant system.
"""

import pytest
import time
import asyncio
import threading
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.performance
class TestImportPerformance:
    """Test import performance improvements."""
    
    def test_cli_import_time(self):
        """Test that CLI imports are faster with lazy loading."""
        start_time = time.time()
        
        # Import the CLI module
        from src import cli
        
        import_time = time.time() - start_time
        
        # Should import in less than 2 seconds (was much slower before)
        assert import_time < 2.0, f"CLI import took {import_time:.3f}s, should be < 2.0s"
        
        # Verify lazy loading is working
        assert cli._numpy is None, "NumPy should not be imported until needed"
        assert cli._rich_modules is None, "Rich modules should not be imported until needed"
    
    def test_events_import_time(self):
        """Test that context events import successfully."""
        start_time = time.time()
        
        from src.core.events.event_types import (
            ContextUpdated, ContextWindowChanged, ContextEntityDetected,
            ContextEntityRemoved, ContextRestored, ContextCleared
        )
        
        import_time = time.time() - start_time
        
        # Should import quickly
        assert import_time < 0.5, f"Events import took {import_time:.3f}s, should be < 0.5s"
        
        # Verify all context events are available
        assert ContextUpdated is not None
        assert ContextWindowChanged is not None
        assert ContextEntityDetected is not None


@pytest.mark.performance
class TestHealthCheckPerformance:
    """Test health check system performance improvements."""
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_timing_precision(self):
        """Test that monitoring loops maintain precise timing."""
        from src.core.health_check import HealthChecker
        
        # Create a mock health checker
        checker = HealthChecker()
        checker.is_running = True
        checker.logger = Mock()
        checker.event_bus = Mock()
        checker.event_bus.emit = Mock(return_value=asyncio.create_future())
        checker.event_bus.emit.return_value.set_result(None)
        
        # Mock the check method
        async def mock_check():
            await asyncio.sleep(0.01)  # Simulate some work
        
        checker._check_component_health = mock_check
        
        # Test timing precision
        start_time = time.time()
        
        # Run a few iterations
        iterations = 3
        interval = 0.1  # 100ms interval
        
        async def run_test():
            task = asyncio.create_task(checker._component_monitoring_loop("test", interval))
            await asyncio.sleep(interval * iterations + 0.05)  # Run for ~300ms
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        await run_test()
        
        elapsed = time.time() - start_time
        expected_time = interval * iterations
        
        # Should maintain timing precision within 50ms
        timing_error = abs(elapsed - expected_time)
        assert timing_error < 0.05, f"Timing error: {timing_error:.3f}s, should be < 0.05s"


@pytest.mark.performance
class TestCachePerformance:
    """Test cache system performance improvements."""
    
    def test_compression_performance(self):
        """Test that compression is faster with optimized settings."""
        from src.integrations.cache.local_cache import CacheCompressor, CompressionType
        
        # Create test data
        test_data = {"key": "value", "numbers": list(range(1000))}
        
        # Test LZ4 compression (should be fastest)
        compressor = CacheCompressor(CompressionType.LZ4)
        
        start_time = time.time()
        compressed = compressor.compress(test_data)
        compression_time = time.time() - start_time
        
        # Compression should be very fast with LZ4
        assert compression_time < 0.1, f"Compression took {compression_time:.3f}s, should be < 0.1s"
        
        # Test decompression
        start_time = time.time()
        decompressed = compressor.decompress(compressed)
        decompression_time = time.time() - start_time
        
        assert decompression_time < 0.1, f"Decompression took {decompression_time:.3f}s, should be < 0.1s"
        assert decompressed == test_data, "Decompressed data should match original"


@pytest.mark.performance
class TestContextManagerPerformance:
    """Test context manager performance improvements."""
    
    def test_entity_to_dict_performance(self):
        """Test that entity serialization is faster."""
        from src.memory.operations.context_manager import ContextEntity
        from datetime import datetime, timezone
        
        # Create test entity
        entity = ContextEntity(
            entity_id="test",
            name="Test Entity",
            entity_type="person",
            first_mentioned_at=datetime.now(timezone.utc),
            last_mentioned_at=datetime.now(timezone.utc),
            attributes={"key": "value"}
        )
        
        # Benchmark to_dict performance
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            entity.to_dict()
        
        elapsed = time.time() - start_time
        avg_time = elapsed / iterations
        
        # Should be very fast (< 1ms per call)
        assert avg_time < 0.001, f"to_dict() took {avg_time*1000:.3f}ms on average, should be < 1ms"
    
    def test_context_update_performance(self):
        """Test that context updates are efficient."""
        from src.memory.operations.context_manager import ContextEntity
        from datetime import datetime, timezone
        
        entity = ContextEntity(
            entity_id="test",
            name="Test Entity", 
            entity_type="person",
            first_mentioned_at=datetime.now(timezone.utc),
            last_mentioned_at=datetime.now(timezone.utc)
        )
        
        # Benchmark update_mention performance
        iterations = 1000
        start_time = time.time()
        
        for i in range(iterations):
            entity.update_mention({"iteration": i})
        
        elapsed = time.time() - start_time
        avg_time = elapsed / iterations
        
        # Should be very fast
        assert avg_time < 0.0001, f"update_mention() took {avg_time*1000:.3f}ms on average, should be < 0.1ms"


@pytest.mark.performance
class TestOverallPerformance:
    """Test overall system performance."""
    
    def test_startup_performance(self):
        """Test that system components start up quickly."""
        start_time = time.time()
        
        # Import key system components
        from src.core.events.event_types import BaseEvent
        from src.core.health_check import HealthStatus
        from src.integrations.cache.local_cache import CacheEntry
        
        import_time = time.time() - start_time
        
        # All core imports should complete quickly
        assert import_time < 1.0, f"Core imports took {import_time:.3f}s, should be < 1.0s"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance"])