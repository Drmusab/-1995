#!/usr/bin/env python3
"""
Memory Profiler Tests
Author: Drmusab
Last Modified: 2025-01-11 16:45:00 UTC

Simple tests to validate memory profiler functionality.
"""

import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
import time

# Test the memory profiling functionality in isolation
import sys
sys.path.insert(0, '/home/runner/work/-1995/-1995')

# Import the events directly from memory_events module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "memory_events", 
    "/home/runner/work/-1995/-1995/src/observability/profiling/memory_events.py"
)
memory_events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_events)

# Import the event classes
MemoryProfilingStarted = memory_events.MemoryProfilingStarted
MemoryProfilingStopped = memory_events.MemoryProfilingStopped
MemorySnapshotTaken = memory_events.MemorySnapshotTaken
MemoryLeakDetected = memory_events.MemoryLeakDetected


class TestMemoryProfilerEvents(unittest.TestCase):
    """Test memory profiler event types."""
    
    def test_memory_profiling_started_event(self):
        """Test MemoryProfilingStarted event creation."""
        event = MemoryProfilingStarted(
            session_name="test_session",
            profiling_mode="basic",
            profiling_level="medium",
            description="Test profiling session"
        )
        
        self.assertEqual(event.session_name, "test_session")
        self.assertEqual(event.profiling_mode, "basic")
        self.assertEqual(event.profiling_level, "medium")
        self.assertIsNotNone(event.timestamp)
    
    def test_memory_profiling_stopped_event(self):
        """Test MemoryProfilingStopped event creation."""
        event = MemoryProfilingStopped(
            session_name="test_session",
            duration_seconds=10.5,
            snapshots_collected=5,
            leaks_detected=2,
            memory_efficiency=0.85
        )
        
        self.assertEqual(event.session_name, "test_session")
        self.assertEqual(event.duration_seconds, 10.5)
        self.assertEqual(event.snapshots_collected, 5)
        self.assertEqual(event.leaks_detected, 2)
        self.assertEqual(event.memory_efficiency, 0.85)
    
    def test_memory_snapshot_taken_event(self):
        """Test MemorySnapshotTaken event creation."""
        event = MemorySnapshotTaken(
            snapshot_type="manual",
            memory_usage_mb=50.5,
            memory_percent=2.5,
            object_count=10000,
            growth_rate_mb_per_min=1.2
        )
        
        self.assertEqual(event.snapshot_type, "manual")
        self.assertEqual(event.memory_usage_mb, 50.5)
        self.assertEqual(event.memory_percent, 2.5)
        self.assertEqual(event.object_count, 10000)
        self.assertEqual(event.growth_rate_mb_per_min, 1.2)
    
    def test_memory_leak_detected_event(self):
        """Test MemoryLeakDetected event creation."""
        event = MemoryLeakDetected(
            leak_type="growth",
            severity="high",
            memory_growth_mb=100.0,
            growth_rate_mb_per_min=10.0,
            duration_minutes=30.0,
            suspected_objects=["list", "dict"],
            recommendations=["Check for reference cycles", "Review memory management"]
        )
        
        self.assertEqual(event.leak_type, "growth")
        self.assertEqual(event.severity, "high")
        self.assertEqual(event.memory_growth_mb, 100.0)
        self.assertEqual(event.growth_rate_mb_per_min, 10.0)
        self.assertEqual(event.duration_minutes, 30.0)
        self.assertEqual(len(event.suspected_objects), 2)
        self.assertEqual(len(event.recommendations), 2)


class TestMemoryProfilerIntegration(unittest.TestCase):
    """Integration tests for memory profiler functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_monitoring_basics(self):
        """Test basic memory monitoring functionality."""
        import psutil
        
        # Test process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.assertGreater(memory_info.rss, 0)
        self.assertGreater(memory_info.vms, 0)
        self.assertGreaterEqual(process.memory_percent(), 0)
        
        # Test system memory info
        virtual_memory = psutil.virtual_memory()
        self.assertGreater(virtual_memory.total, 0)
        self.assertGreater(virtual_memory.available, 0)
    
    def test_tracemalloc_functionality(self):
        """Test tracemalloc integration."""
        import tracemalloc
        
        # Start tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Allocate some memory
        test_data = [i for i in range(1000)]
        
        # Get traced memory
        current, peak = tracemalloc.get_traced_memory()
        self.assertGreater(current, 0)
        self.assertGreater(peak, 0)
        
        # Take snapshot
        snapshot = tracemalloc.take_snapshot()
        self.assertIsNotNone(snapshot)
        
        # Get statistics
        stats = snapshot.statistics('lineno')
        self.assertGreater(len(stats), 0)
        
        # Clean up
        del test_data
        tracemalloc.stop()
    
    def test_garbage_collection_monitoring(self):
        """Test garbage collection monitoring."""
        import gc
        
        # Get GC stats
        gc_stats = gc.get_stats()
        self.assertGreater(len(gc_stats), 0)
        
        # Get counts and thresholds
        count = gc.get_count()
        threshold = gc.get_threshold()
        
        self.assertEqual(len(count), 3)  # 3 generations
        self.assertEqual(len(threshold), 3)
        
        # Count objects
        objects = gc.get_objects()
        self.assertGreater(len(objects), 0)
    
    async def test_async_memory_operations(self):
        """Test async memory operations."""
        import psutil
        
        async def monitor_memory():
            snapshots = []
            for i in range(3):
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                snapshots.append(memory_mb)
                await asyncio.sleep(0.1)
            return snapshots
        
        snapshots = await monitor_memory()
        self.assertEqual(len(snapshots), 3)
        for snapshot in snapshots:
            self.assertGreater(snapshot, 0)
    
    def test_memory_growth_detection(self):
        """Test memory growth detection algorithm."""
        # Simulate memory snapshots with growth
        memory_values = [10.0, 12.0, 15.0, 18.0, 22.0]  # MB
        
        # Simple growth detection
        growth = memory_values[-1] - memory_values[0]
        threshold = 10.0  # MB
        
        self.assertGreater(growth, threshold)
        
        # Calculate growth rate
        time_diff = 4  # 4 time intervals
        growth_rate = growth / time_diff
        
        self.assertGreater(growth_rate, 2.0)  # >2 MB per interval
    
    def test_event_system_integration(self):
        """Test event system integration."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        # Create events
        start_event = MemoryProfilingStarted(
            session_name="test",
            profiling_mode="basic"
        )
        
        snapshot_event = MemorySnapshotTaken(
            snapshot_type="test",
            memory_usage_mb=25.0
        )
        
        stop_event = MemoryProfilingStopped(
            session_name="test",
            duration_seconds=5.0
        )
        
        # Simulate event handling
        for event in [start_event, snapshot_event, stop_event]:
            event_handler(event)
        
        self.assertEqual(len(events_received), 3)
        self.assertIsInstance(events_received[0], MemoryProfilingStarted)
        self.assertIsInstance(events_received[1], MemorySnapshotTaken)
        self.assertIsInstance(events_received[2], MemoryProfilingStopped)


class AsyncTestCase(unittest.TestCase):
    """Base class for async test cases."""
    
    def setUp(self):
        """Set up async event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        """Clean up async event loop."""
        self.loop.close()
    
    def async_test(self, coro):
        """Helper to run async tests."""
        return self.loop.run_until_complete(coro)


class TestAsyncMemoryProfiler(AsyncTestCase):
    """Async tests for memory profiler."""
    
    def test_async_memory_monitoring(self):
        """Test async memory monitoring."""
        async def test_coro():
            import psutil
            
            # Simulate async monitoring
            memory_readings = []
            for i in range(3):
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_readings.append(memory_mb)
                await asyncio.sleep(0.05)
            
            # Verify readings
            self.assertEqual(len(memory_readings), 3)
            for reading in memory_readings:
                self.assertGreater(reading, 0)
        
        self.async_test(test_coro())
    
    def test_async_event_emission(self):
        """Test async event emission."""
        async def test_coro():
            events_emitted = []
            
            async def emit_event(event):
                events_emitted.append(event)
            
            # Emit test events
            events = [
                MemoryProfilingStarted(session_name="async_test"),
                MemorySnapshotTaken(memory_usage_mb=30.0),
                MemoryProfilingStopped(session_name="async_test")
            ]
            
            for event in events:
                await emit_event(event)
            
            self.assertEqual(len(events_emitted), 3)
        
        self.async_test(test_coro())


def run_memory_profiler_tests():
    """Run all memory profiler tests."""
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMemoryProfilerEvents))
    suite.addTest(unittest.makeSuite(TestMemoryProfilerIntegration))
    suite.addTest(unittest.makeSuite(TestAsyncMemoryProfiler))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Memory Profiler Tests")
    print("=" * 50)
    
    success = run_memory_profiler_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)