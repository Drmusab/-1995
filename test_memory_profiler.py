#!/usr/bin/env python3
"""
Comprehensive test script for the Memory Profiler.

This script demonstrates all the major features of the memory profiler including:
- Basic memory monitoring
- Profiling sessions
- Component tracking
- Memory leak detection
- Optimization recommendations
- Decorator usage
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.observability.profiling.memory_profiler import (
    MemoryProfiler,
    MemoryProfilingMode,
    MemoryProfilingLevel,
    profile_memory,
    memory_profiling_context
)


class MockContainer:
    """Mock container for testing."""
    def get(self, cls, default=None):
        return default


@profile_memory("memory_intensive_component")
async def memory_intensive_function():
    """Simulate a memory-intensive function."""
    data = []
    for i in range(1000):
        data.append([0] * 1000)  # Allocate memory
        if i % 100 == 0:
            await asyncio.sleep(0.01)  # Small delay
    return len(data)


async def test_basic_functionality():
    """Test basic memory profiler functionality."""
    print("=" * 60)
    print("TESTING BASIC MEMORY PROFILER FUNCTIONALITY")
    print("=" * 60)
    
    container = MockContainer()
    profiler = MemoryProfiler(container)
    
    print(f"✓ Memory profiler initialized")
    print(f"  - Config mode: {profiler.config.mode.value}")
    print(f"  - Monitoring interval: {profiler.config.monitoring_interval}s")
    print(f"  - Memory warning threshold: {profiler.config.memory_warning_threshold_mb}MB")
    
    # Test current memory usage
    usage = profiler.get_current_memory_usage()
    print(f"✓ Current memory usage collected")
    print(f"  - Process RSS: {usage['process_memory']['rss'] / (1024*1024):.2f}MB")
    print(f"  - System used: {usage['system_memory']['used_percent']:.1f}%")
    
    # Test memory snapshot
    snapshot = await profiler.get_memory_snapshot()
    print(f"✓ Memory snapshot taken")
    print(f"  - Timestamp: {snapshot.timestamp}")
    print(f"  - Process RSS: {snapshot.process_rss / (1024*1024):.2f}MB")
    print(f"  - Python objects: {snapshot.python_objects_count}")
    
    return profiler


async def test_component_tracking(profiler):
    """Test component tracking functionality."""
    print("\n" + "=" * 60)
    print("TESTING COMPONENT TRACKING")
    print("=" * 60)
    
    # Register components
    components = ["session_manager", "workflow_engine", "model_handler"]
    for component in components:
        await profiler.register_component(component)
        print(f"✓ Registered component: {component}")
    
    print(f"✓ Total components tracked: {len(profiler.component_trackers)}")
    
    # Simulate component memory usage
    for component_name in components:
        tracker = profiler.component_trackers[component_name]
        tracker.record_allocation(f"alloc_{component_name}_1", 1024 * 1024)  # 1MB
        tracker.record_allocation(f"alloc_{component_name}_2", 2048 * 1024)  # 2MB
        
    print("✓ Simulated component memory allocations")
    
    # Get component memory trends
    for component_name in components:
        tracker = profiler.component_trackers[component_name]
        trend = tracker.get_memory_trend()
        print(f"  - {component_name}: {trend['current_usage'] / (1024*1024):.2f}MB")


async def test_profiling_session(profiler):
    """Test profiling session functionality."""
    print("\n" + "=" * 60)
    print("TESTING PROFILING SESSION")
    print("=" * 60)
    
    # Start profiling session
    session_id = await profiler.start_profiling(
        session_name="comprehensive_test_session",
        description="Testing all memory profiler features",
        mode=MemoryProfilingMode.DETAILED,
        components_to_monitor=["session_manager", "workflow_engine"]
    )
    
    print(f"✓ Started profiling session: {session_id[:8]}...")
    print(f"  - Mode: {profiler.current_session.config.mode.value}")
    print(f"  - Components monitored: {len(profiler.current_session.components_monitored)}")
    
    # Simulate memory-intensive operations
    print("✓ Running memory-intensive operations...")
    
    # Use the memory profiling decorator
    result = await memory_intensive_function()
    print(f"  - Memory-intensive function completed: {result} items")
    
    # Simulate more memory usage
    big_data = []
    for i in range(500):
        big_data.append([0] * 2000)  # More memory allocation
        if i % 100 == 0:
            await asyncio.sleep(0.1)
    
    # Let the profiler collect data
    await asyncio.sleep(3)
    
    # Stop profiling session
    session = await profiler.stop_profiling(session_id)
    
    print(f"✓ Stopped profiling session")
    print(f"  - Duration: {session.duration:.2f}s")
    print(f"  - Snapshots collected: {len(session.snapshots)}")
    print(f"  - Peak memory: {session.peak_memory_usage / (1024*1024):.2f}MB")
    print(f"  - Average memory: {session.average_memory_usage / (1024*1024):.2f}MB")
    print(f"  - Efficiency score: {session.memory_efficiency_score:.1f}%")
    print(f"  - Leaks detected: {len(session.leaks_detected)}")
    print(f"  - Recommendations: {len(session.recommendations)}")
    
    # Show recommendations
    if session.recommendations:
        print("✓ Optimization recommendations:")
        for i, rec in enumerate(session.recommendations, 1):
            print(f"  {i}. {rec.title} (Priority: {rec.priority})")
            print(f"     - Potential savings: {rec.potential_savings_mb:.1f}MB")
            print(f"     - {rec.description}")


async def test_context_manager(profiler):
    """Test context manager functionality."""
    print("\n" + "=" * 60)
    print("TESTING CONTEXT MANAGER")
    print("=" * 60)
    
    async with memory_profiling_context(
        profiler,
        session_name="context_manager_test",
        description="Testing context manager functionality",
        mode=MemoryProfilingMode.LEAK_DETECTION
    ) as session_id:
        print(f"✓ Context manager session started: {session_id[:8]}...")
        
        # Simulate some operations
        data = []
        for i in range(200):
            data.append([i] * 500)
            if i % 50 == 0:
                await asyncio.sleep(0.05)
        
        print(f"✓ Completed operations within context")
    
    # Session should be automatically stopped
    session = profiler.get_session(session_id)
    print(f"✓ Context manager session completed")
    print(f"  - Final status: {session.status.value}")
    print(f"  - Duration: {session.duration:.2f}s")


async def test_memory_reporting(profiler):
    """Test memory reporting functionality."""
    print("\n" + "=" * 60)
    print("TESTING MEMORY REPORTING")
    print("=" * 60)
    
    # Generate comprehensive report
    report = await profiler.generate_memory_report()
    
    print(f"✓ Generated memory report")
    print(f"  - Report ID: {report['report_id'][:8]}...")
    print(f"  - Total sessions: {report['total_sessions']}")
    print(f"  - Peak memory: {report['summary']['peak_memory_mb']:.2f}MB")
    print(f"  - Total leaks detected: {report['summary']['total_leaks_detected']}")
    print(f"  - Total recommendations: {report['summary']['total_recommendations']}")
    print(f"  - Average efficiency: {report['summary']['average_efficiency_score']:.1f}%")
    
    # Show session summaries
    if report['sessions']:
        print("✓ Session summaries:")
        for session in report['sessions']:
            print(f"  - {session['name']}: {session['peak_memory_mb']:.2f}MB peak, "
                  f"{session['efficiency_score']:.1f}% efficiency")


async def test_health_status(profiler):
    """Test health status functionality."""
    print("\n" + "=" * 60)
    print("TESTING HEALTH STATUS")
    print("=" * 60)
    
    health = profiler.get_health_status()
    
    print(f"✓ Health status retrieved")
    print(f"  - Profiler status: {health['profiler_status']}")
    print(f"  - Sessions count: {health['sessions_count']}")
    print(f"  - Components tracked: {health['components_tracked']}")
    print(f"  - Current memory: {health['current_memory_mb']:.2f}MB")
    print(f"  - Leaks detected: {health['leaks_detected']}")
    print(f"  - Recommendations generated: {health['recommendations_generated']}")


async def main():
    """Main test function."""
    print("Memory Profiler Comprehensive Test Suite")
    print("========================================")
    
    try:
        # Initialize profiler
        profiler = await test_basic_functionality()
        
        # Test component tracking
        await test_component_tracking(profiler)
        
        # Test profiling session
        await test_profiling_session(profiler)
        
        # Test context manager
        await test_context_manager(profiler)
        
        # Test memory reporting
        await test_memory_reporting(profiler)
        
        # Test health status
        await test_health_status(profiler)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 60)
        print(f"Total memory profiling sessions: {len(profiler.profiling_sessions)}")
        print(f"Total memory snapshots taken: {len(profiler.memory_snapshots)}")
        print(f"Output directory: {profiler.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)