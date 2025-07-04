#!/usr/bin/env python3
"""
GPU Profiler Usage Examples
This file demonstrates various ways to use the GPU profiler.
"""

import asyncio
from src.core.dependency_injection import Container
from src.observability.profiling import GPUProfiler
from src.observability.profiling.gpu_profiler import GPUProfilingLevel, GPUProfilingMode


async def example_basic_usage():
    """Basic GPU profiler usage example."""
    print("=== Basic GPU Profiler Usage ===")
    
    # Initialize container and profiler
    container = Container()
    gpu_profiler = GPUProfiler(container)
    
    # Check if GPU is available
    if not gpu_profiler.gpu_available:
        print("No GPU available, skipping GPU-intensive examples")
        return
    
    # Start basic profiling
    print("Starting GPU profiling...")
    result = await gpu_profiler.start_profiling()
    print(f"Profiling started: {result}")
    
    # Simulate some GPU work
    print("Simulating GPU work...")
    await asyncio.sleep(2)
    
    # Get current status
    status = await gpu_profiler.get_gpu_status()
    print(f"Current status: {status}")
    
    # Stop profiling
    print("Stopping GPU profiling...")
    result = await gpu_profiler.stop_profiling()
    print(f"Profiling stopped: {result}")


async def example_detailed_profiling():
    """Detailed GPU profiling with advanced features."""
    print("\n=== Detailed GPU Profiling ===")
    
    container = Container()
    gpu_profiler = GPUProfiler(container)
    
    if not gpu_profiler.gpu_available:
        print("No GPU available for detailed profiling")
        return
    
    # Start detailed profiling
    await gpu_profiler.start_profiling(
        level=GPUProfilingLevel.DETAILED,
        mode=GPUProfilingMode.PROFILING
    )
    
    # Profile specific kernel execution
    print("Profiling kernel execution...")
    with gpu_profiler.profile_kernel("example_kernel", device_id=0, component="example_component") as kernel_info:
        print(f"Kernel started: {kernel_info.kernel_name}")
        # Simulate kernel work
        await asyncio.sleep(1)
    print("Kernel profiling completed")
    
    # Get optimization suggestions
    suggestions = await gpu_profiler.get_optimization_suggestions()
    print(f"\nOptimization suggestions ({len(suggestions)}):")
    for suggestion in suggestions:
        print(f"  - {suggestion['severity']}: {suggestion['message']}")
        print(f"    Suggestion: {suggestion['suggestion']}")
    
    await gpu_profiler.stop_profiling()


async def example_component_tracking():
    """Example of component-specific GPU tracking."""
    print("\n=== Component-Specific GPU Tracking ===")
    
    container = Container()
    gpu_profiler = GPUProfiler(container)
    
    # Start monitoring
    if gpu_profiler.gpu_available:
        await gpu_profiler.start_profiling(level=GPUProfilingLevel.STANDARD)
    
    # Simulate component events (these would normally be emitted by the system)
    print("Simulating component events...")
    
    # Component 1: Image processing
    event1 = type('ProcessingStarted', (), {'component': 'image_processor', 'session_id': 'session_1'})()
    await gpu_profiler._handle_processing_started(event1)
    
    await asyncio.sleep(0.5)  # Simulate processing time
    
    event1_complete = type('ProcessingCompleted', (), {
        'component': 'image_processor', 
        'session_id': 'session_1',
        'processing_time': 0.5
    })()
    await gpu_profiler._handle_processing_completed(event1_complete)
    
    # Component 2: NLP processing
    event2 = type('ProcessingStarted', (), {'component': 'nlp_processor', 'session_id': 'session_2'})()
    await gpu_profiler._handle_processing_started(event2)
    
    await asyncio.sleep(0.3)  # Simulate processing time
    
    event2_complete = type('ProcessingCompleted', (), {
        'component': 'nlp_processor', 
        'session_id': 'session_2',
        'processing_time': 0.3
    })()
    await gpu_profiler._handle_processing_completed(event2_complete)
    
    # Get component usage statistics
    usage = gpu_profiler.get_component_gpu_usage()
    print(f"Component GPU usage: {usage}")
    
    # Get specific component usage
    image_usage = gpu_profiler.get_component_gpu_usage("image_processor")
    print(f"Image processor usage: {image_usage}")
    
    if gpu_profiler.gpu_available:
        await gpu_profiler.stop_profiling()


async def example_workflow_tracking():
    """Example of workflow-level GPU tracking."""
    print("\n=== Workflow-Level GPU Tracking ===")
    
    container = Container()
    gpu_profiler = GPUProfiler(container)
    
    if gpu_profiler.gpu_available:
        await gpu_profiler.start_profiling()
    
    # Simulate workflow events
    workflow_id = "workflow_123"
    
    # Workflow started
    workflow_start = type('WorkflowStarted', (), {'workflow_id': workflow_id})()
    await gpu_profiler._handle_workflow_started(workflow_start)
    
    # Workflow step 1
    step1_start = type('WorkflowStepStarted', (), {
        'workflow_id': workflow_id, 
        'step_id': 'data_preprocessing'
    })()
    await gpu_profiler._handle_workflow_step_started(step1_start)
    
    await asyncio.sleep(0.2)
    
    step1_complete = type('WorkflowStepCompleted', (), {
        'workflow_id': workflow_id, 
        'step_id': 'data_preprocessing',
        'step_duration': 0.2
    })()
    await gpu_profiler._handle_workflow_step_completed(step1_complete)
    
    # Workflow step 2
    step2_start = type('WorkflowStepStarted', (), {
        'workflow_id': workflow_id, 
        'step_id': 'model_inference'
    })()
    await gpu_profiler._handle_workflow_step_started(step2_start)
    
    await asyncio.sleep(0.4)
    
    step2_complete = type('WorkflowStepCompleted', (), {
        'workflow_id': workflow_id, 
        'step_id': 'model_inference',
        'step_duration': 0.4
    })()
    await gpu_profiler._handle_workflow_step_completed(step2_complete)
    
    # Workflow completed
    workflow_complete = type('WorkflowCompleted', (), {
        'workflow_id': workflow_id, 
        'execution_time': 0.6
    })()
    await gpu_profiler._handle_workflow_completed(workflow_complete)
    
    # Get workflow usage
    workflow_usage = gpu_profiler.get_component_gpu_usage(f"workflow_{workflow_id}")
    print(f"Workflow usage: {workflow_usage}")
    
    step1_usage = gpu_profiler.get_component_gpu_usage(f"step_{workflow_id}_data_preprocessing")
    print(f"Step 1 usage: {step1_usage}")
    
    step2_usage = gpu_profiler.get_component_gpu_usage(f"step_{workflow_id}_model_inference")
    print(f"Step 2 usage: {step2_usage}")
    
    if gpu_profiler.gpu_available:
        await gpu_profiler.stop_profiling()


async def example_health_monitoring():
    """Example of GPU profiler health monitoring."""
    print("\n=== GPU Profiler Health Monitoring ===")
    
    container = Container()
    gpu_profiler = GPUProfiler(container)
    
    # Get health status
    health = await gpu_profiler._health_check_callback()
    
    print("GPU Profiler Health Status:")
    print(f"  Status: {health['status']}")
    print(f"  GPU Available: {health['gpu_available']}")
    print(f"  GPU Count: {health['gpu_count']}")
    print(f"  Monitoring Active: {health['monitoring_active']}")
    print(f"  Samples Collected: {health['samples_collected']}")
    print(f"  Active Kernels: {health['active_kernels']}")
    print(f"  Detected Bottlenecks: {health['detected_bottlenecks']}")
    print(f"  Detected Leaks: {health['detected_leaks']}")
    
    if health['gpu_available'] and 'devices' in health:
        print("\nGPU Devices:")
        for device_id, device_info in health['devices'].items():
            print(f"  Device {device_id}:")
            print(f"    Name: {device_info['name']}")
            print(f"    Available: {device_info['available']}")
            print(f"    Memory: {device_info['memory_total_mb']} MB")


async def example_error_handling():
    """Example of GPU profiler error handling."""
    print("\n=== GPU Profiler Error Handling ===")
    
    container = Container()
    gpu_profiler = GPUProfiler(container)
    
    # Test error handling when GPU is not available
    if not gpu_profiler.gpu_available:
        print("Testing error handling without GPU...")
        
        try:
            await gpu_profiler.start_profiling()
            print("ERROR: Should have raised RuntimeError")
        except RuntimeError as e:
            print(f"✓ Correctly raised RuntimeError: {e}")
        
        # Test status methods still work
        status = await gpu_profiler.get_gpu_status()
        print(f"✓ Status without GPU: {status}")
        
        suggestions = await gpu_profiler.get_optimization_suggestions()
        print(f"✓ Suggestions without GPU: {len(suggestions)} found")
        
        usage = gpu_profiler.get_component_gpu_usage()
        print(f"✓ Component usage without GPU: {list(usage.keys())}")
    
    else:
        print("GPU available - testing normal operation...")
        
        # Test normal operation
        result = await gpu_profiler.start_profiling()
        print(f"✓ Started profiling: {result}")
        
        # Test stopping when already running
        result2 = await gpu_profiler.start_profiling()
        print(f"✓ Start when running: {result2}")
        
        result3 = await gpu_profiler.stop_profiling()
        print(f"✓ Stopped profiling: {result3}")
        
        # Test stopping when already stopped
        result4 = await gpu_profiler.stop_profiling()
        print(f"✓ Stop when stopped: {result4}")


async def main():
    """Run all GPU profiler examples."""
    print("GPU Profiler Usage Examples")
    print("=" * 50)
    
    try:
        await example_basic_usage()
        await example_detailed_profiling()
        await example_component_tracking()
        await example_workflow_tracking()
        await example_health_monitoring()
        await example_error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())