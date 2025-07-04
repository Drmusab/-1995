"""
Basic tests for GPU Profiler functionality
Author: Drmusab

Simple tests to validate GPU profiler core functionality.
"""

import asyncio
import sys
import tempfile
import json
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observability.profiling.gpu_profiler import (
    GPUProfiler, GPUDevice, GPUVendor, GPUType, 
    ProfilingMode, ProfilingLevel, GPUProfilerConfig
)


def test_gpu_device_creation():
    """Test GPU device data structure creation."""
    print("Testing GPU device creation...")
    
    device = GPUDevice(
        index=0,
        name="Test GPU",
        vendor=GPUVendor.NVIDIA,
        gpu_type=GPUType.DISCRETE,
        total_memory_mb=8192.0,
        free_memory_mb=6144.0,
        used_memory_mb=2048.0,
        utilization_percent=75.0,
        temperature_celsius=65.0
    )
    
    assert device.index == 0
    assert device.name == "Test GPU"
    assert device.vendor == GPUVendor.NVIDIA
    assert device.total_memory_mb == 8192.0
    assert device.memory_utilization_percent == 0.0  # Not calculated yet
    
    print("✓ GPU device creation test passed")


def test_gpu_profiler_config():
    """Test GPU profiler configuration."""
    print("Testing GPU profiler configuration...")
    
    config = GPUProfilerConfig(
        mode=ProfilingMode.COMPREHENSIVE,
        level=ProfilingLevel.HIGH,
        monitoring_interval_seconds=0.5,
        temperature_warning_celsius=75.0
    )
    
    assert config.mode == ProfilingMode.COMPREHENSIVE
    assert config.level == ProfilingLevel.HIGH
    assert config.monitoring_interval_seconds == 0.5
    assert config.temperature_warning_celsius == 75.0
    assert config.enable_cuda_profiling == True  # Default
    
    print("✓ GPU profiler configuration test passed")


async def test_gpu_profiler_basic_functionality():
    """Test basic GPU profiler functionality."""
    print("Testing GPU profiler basic functionality...")
    
    # Test instantiation
    profiler = GPUProfiler()
    assert profiler is not None
    assert isinstance(profiler.devices, list)
    assert profiler.is_running == False
    assert profiler.current_session is None
    
    # Test start/stop
    await profiler.start()
    assert profiler.is_running == True
    
    await profiler.stop()
    assert profiler.is_running == False
    
    print("✓ GPU profiler basic functionality test passed")


async def test_profiling_sessions():
    """Test profiling session management."""
    print("Testing profiling session management...")
    
    profiler = GPUProfiler()
    await profiler.start()
    
    # Start session
    session_id = await profiler.start_profiling_session("test_session", "Test description")
    assert session_id is not None
    assert profiler.current_session is not None
    assert profiler.current_session.session_id == session_id
    assert profiler.current_session.name == "test_session"
    
    # Track some inferences
    await profiler.track_inference(100.0)
    await profiler.track_inference(150.0)
    await profiler.track_inference(80.0)
    
    assert len(profiler.inference_times) == 3
    
    # Take snapshot
    snapshot = await profiler.take_snapshot("test")
    assert snapshot is not None
    assert snapshot.inference_count == 3
    
    # Stop session
    stopped_id = await profiler.stop_profiling_session()
    assert stopped_id == session_id
    assert profiler.current_session is None
    
    # Check session was stored
    sessions = await profiler.get_profiling_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    
    await profiler.stop()
    
    print("✓ Profiling session management test passed")


async def test_health_check():
    """Test health check functionality."""
    print("Testing health check functionality...")
    
    profiler = GPUProfiler()
    
    health = await profiler.health_check()
    assert health is not None
    assert "status" in health
    assert "details" in health
    assert health["status"] in ["healthy", "degraded", "warning", "unhealthy"]
    
    print("✓ Health check test passed")


async def test_gpu_status():
    """Test GPU status reporting."""
    print("Testing GPU status reporting...")
    
    profiler = GPUProfiler()
    await profiler.start()
    
    status = await profiler.get_gpu_status()
    assert status is not None
    assert "profiler_running" in status
    assert "devices_count" in status
    assert "devices" in status
    assert "timestamp" in status
    assert status["profiler_running"] == True
    assert status["devices_count"] == len(profiler.devices)
    
    await profiler.stop()
    
    print("✓ GPU status reporting test passed")


async def test_optimization_suggestions():
    """Test optimization suggestion generation."""
    print("Testing optimization suggestion generation...")
    
    profiler = GPUProfiler()
    await profiler.start()
    
    session_id = await profiler.start_profiling_session("optimization_test")
    
    # Simulate slow inferences
    for _ in range(5):
        await profiler.track_inference(1200.0)  # 1.2 second inferences
    
    stopped_id = await profiler.stop_profiling_session()
    
    # Check for optimization suggestions
    session = profiler.profiling_sessions[stopped_id]
    assert len(session.optimization_suggestions) > 0
    
    # Should have performance optimization suggestion
    perf_suggestions = [s for s in session.optimization_suggestions if s["type"] == "performance"]
    assert len(perf_suggestions) > 0
    
    await profiler.stop()
    
    print("✓ Optimization suggestion generation test passed")


async def test_report_export():
    """Test session report export functionality."""
    print("Testing session report export...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create profiler with custom output directory
        profiler = GPUProfiler()
        profiler.config.output_dir = temp_dir
        profiler.output_dir = Path(temp_dir)
        
        await profiler.start()
        
        session_id = await profiler.start_profiling_session("export_test")
        
        # Add some data
        await profiler.track_inference(100.0)
        await profiler.take_snapshot("test")
        
        stopped_id = await profiler.stop_profiling_session()
        
        # Check if report was exported
        session = profiler.profiling_sessions[stopped_id]
        if session.report_file_path:
            assert session.report_file_path.exists()
            
            # Verify report content
            with open(session.report_file_path, 'r') as f:
                report_data = json.load(f)
            
            assert "session" in report_data
            assert "generated_at" in report_data
            assert "profiler_version" in report_data
            assert report_data["session"]["session_id"] == session_id
        
        await profiler.stop()
    
    print("✓ Session report export test passed")


async def run_all_tests():
    """Run all tests."""
    print("🧪 Running GPU Profiler Tests")
    print("=" * 40)
    
    try:
        # Synchronous tests
        test_gpu_device_creation()
        test_gpu_profiler_config()
        
        # Asynchronous tests
        await test_gpu_profiler_basic_functionality()
        await test_profiling_sessions()
        await test_health_check()
        await test_gpu_status()
        await test_optimization_suggestions()
        await test_report_export()
        
        print("\n" + "=" * 40)
        print("🎉 All GPU Profiler tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)