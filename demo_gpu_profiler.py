"""
GPU Profiler Demonstration Script
Author: Drmusab

This script demonstrates the comprehensive GPU profiler functionality,
including device discovery, profiling sessions, performance tracking,
and optimization suggestions.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.observability.profiling.gpu_profiler import GPUProfiler


async def demonstrate_gpu_profiler():
    """Demonstrate GPU profiler functionality."""
    print("🚀 GPU Profiler Demonstration")
    print("=" * 50)
    
    # Create profiler instance
    print("\n1. Initializing GPU Profiler...")
    profiler = GPUProfiler()
    
    print(f"   ✓ GPU Profiler initialized")
    print(f"   ✓ Integration mode: {profiler.integration_mode}")
    print(f"   ✓ CUDA available: {profiler.cuda_available}")
    print(f"   ✓ OpenCL available: {profiler.opencl_available}")
    print(f"   ✓ Discovered {len(profiler.devices)} GPU device(s)")
    
    if profiler.devices:
        print("\n   📊 GPU Devices:")
        for device in profiler.devices:
            print(f"      - Device {device.index}: {device.name} ({device.vendor.value})")
            print(f"        Memory: {device.total_memory_mb:.0f} MB total, {device.used_memory_mb:.0f} MB used")
            if device.temperature_celsius:
                print(f"        Temperature: {device.temperature_celsius}°C")
            if device.power_draw_watts:
                print(f"        Power: {device.power_draw_watts:.1f}W")
    else:
        print("   ⚠️  No GPU devices available (expected in sandboxed environment)")
    
    # Start profiler
    print("\n2. Starting GPU Profiler...")
    await profiler.start()
    print("   ✓ GPU profiler started successfully")
    
    # Start a profiling session
    print("\n3. Starting Profiling Session...")
    session_id = await profiler.start_profiling_session(
        "demo_session", 
        "Demonstration of GPU profiling capabilities"
    )
    print(f"   ✓ Started profiling session: {session_id}")
    
    # Simulate some AI inference operations
    print("\n4. Simulating AI Inference Operations...")
    inference_times = [150.0, 120.0, 180.0, 95.0, 130.0, 200.0, 110.0, 160.0]
    
    for i, inference_time in enumerate(inference_times, 1):
        await profiler.track_inference(inference_time)
        print(f"   📈 Tracked inference {i}: {inference_time}ms")
        
        # Take a snapshot every few inferences
        if i % 3 == 0:
            snapshot = await profiler.take_snapshot(f"inference_batch_{i//3}")
            print(f"   📸 Took snapshot: {snapshot.snapshot_id}")
        
        # Small delay to simulate real processing
        await asyncio.sleep(0.1)
    
    # Get current status
    print("\n5. Getting GPU Status...")
    status = await profiler.get_gpu_status()
    print(f"   ✓ Devices: {status['devices_count']}")
    print(f"   ✓ Profiler running: {status['profiler_running']}")
    print(f"   ✓ Current session: {status['current_session']}")
    print(f"   ✓ Total snapshots: {status['total_snapshots']}")
    
    # Health check
    print("\n6. Performing Health Check...")
    health = await profiler.health_check()
    print(f"   ✓ Health status: {health['status']}")
    print(f"   ✓ Details: {health['details']}")
    
    # Let the profiler run for a bit to collect more data
    print("\n7. Collecting monitoring data...")
    await asyncio.sleep(2)
    print("   ✓ Monitoring data collected")
    
    # Stop the profiling session
    print("\n8. Stopping Profiling Session...")
    stopped_session_id = await profiler.stop_profiling_session()
    print(f"   ✓ Stopped session: {stopped_session_id}")
    
    # Get session information
    print("\n9. Reviewing Session Results...")
    sessions = await profiler.get_profiling_sessions()
    if sessions:
        session = sessions[0]
        print(f"   ✓ Session duration: {session['duration_seconds']:.2f} seconds")
        print(f"   ✓ Snapshots collected: {session['snapshots_count']}")
        print(f"   ✓ Total inferences: {session['total_inferences']}")
        print(f"   ✓ Peak GPU utilization: {session['peak_gpu_utilization']:.1f}%")
        
        # Show optimization suggestions if any
        if stopped_session_id in profiler.profiling_sessions:
            session_obj = profiler.profiling_sessions[stopped_session_id]
            if session_obj.optimization_suggestions:
                print(f"\n   💡 Optimization Suggestions:")
                for suggestion in session_obj.optimization_suggestions:
                    print(f"      - {suggestion['type']}: {suggestion['message']}")
                    print(f"        Recommendation: {suggestion['recommendation']}")
            else:
                print(f"   ✓ No optimization suggestions (system running efficiently)")
    
    # Show exported files
    print("\n10. Exported Files...")
    if stopped_session_id in profiler.profiling_sessions:
        session_obj = profiler.profiling_sessions[stopped_session_id]
        if session_obj.report_file_path:
            print(f"    📄 Report: {session_obj.report_file_path}")
        else:
            print(f"    ⚠️  No report file generated")
    
    # Stop profiler
    print("\n11. Stopping GPU Profiler...")
    await profiler.stop()
    print("   ✓ GPU profiler stopped successfully")
    
    print("\n" + "=" * 50)
    print("🎉 GPU Profiler Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Device discovery and monitoring")
    print("✓ Profiling session management")
    print("✓ Performance tracking and metrics")
    print("✓ Real-time snapshots")
    print("✓ Health monitoring")
    print("✓ Optimization suggestions")
    print("✓ Report generation")
    print("✓ Graceful degradation")


async def demonstrate_advanced_features():
    """Demonstrate advanced GPU profiler features."""
    print("\n🔬 Advanced Features Demonstration")
    print("=" * 50)
    
    profiler = GPUProfiler()
    await profiler.start()
    
    # Test different profiling modes
    print("\n1. Testing Different Profiling Modes...")
    from src.observability.profiling.gpu_profiler import ProfilingMode, ProfilingLevel, GPUProfilerConfig
    
    # High-detail session
    config = GPUProfilerConfig(
        mode=ProfilingMode.COMPREHENSIVE,
        level=ProfilingLevel.MAXIMUM,
        monitoring_interval_seconds=0.5
    )
    
    session_id = await profiler.start_profiling_session(
        "advanced_session",
        "High-detail profiling session",
        config
    )
    print(f"   ✓ Started comprehensive profiling session")
    
    # Simulate various workload patterns
    print("\n2. Simulating Different Workload Patterns...")
    
    # Burst workload
    print("   📊 Burst workload simulation...")
    for _ in range(5):
        await profiler.track_inference(50.0)  # Fast inferences
    
    # Heavy workload
    print("   📊 Heavy workload simulation...")
    for _ in range(3):
        await profiler.track_inference(500.0)  # Slow inferences
    
    # Variable workload
    print("   📊 Variable workload simulation...")
    import random
    for _ in range(10):
        inference_time = random.uniform(80.0, 300.0)
        await profiler.track_inference(inference_time)
    
    await asyncio.sleep(1)  # Let monitoring collect data
    
    # Take final snapshot
    final_snapshot = await profiler.take_snapshot("final_advanced")
    print(f"   📸 Final snapshot taken: {final_snapshot.inference_count} inferences tracked")
    
    await profiler.stop_profiling_session()
    await profiler.stop()
    
    print("   ✓ Advanced features demonstration complete")


async def main():
    """Run the complete demonstration."""
    try:
        await demonstrate_gpu_profiler()
        await demonstrate_advanced_features()
        
        print("\n🏆 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())