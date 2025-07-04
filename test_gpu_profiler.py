#!/usr/bin/env python3
"""
Test script for GPU Profiler
Tests the GPU profiler in both GPU-available and GPU-unavailable environments.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock dependencies that might not be available
class MockContainer:
    def __init__(self):
        self.services = {}
        
    def get(self, service_type, default=None):
        if service_type.__name__ in self.services:
            return self.services[service_type.__name__]
        
        # Create mock services
        if hasattr(service_type, '__name__'):
            service_name = service_type.__name__
            if service_name == 'ConfigLoader':
                return MockConfigLoader()
            elif service_name == 'EventBus':
                return MockEventBus()
            elif service_name == 'ErrorHandler':
                return MockErrorHandler()
            elif service_name == 'HealthCheck':
                return MockHealthCheck()
            elif service_name == 'MetricsCollector':
                return MockMetricsCollector()
            elif service_name == 'TraceManager':
                return MockTraceManager()
        
        return default

class MockConfigLoader:
    def get(self, key, default=None):
        return default

class MockEventBus:
    async def emit(self, event):
        print(f"Mock event emitted: {event.__class__.__name__}")
        
    def subscribe(self, event_type, handler):
        print(f"Mock subscription: {event_type} -> {handler.__name__}")

class MockErrorHandler:
    pass

class MockHealthCheck:
    def register_component(self, name, callback):
        print(f"Mock health check registered: {name}")

class MockMetricsCollector:
    def register_gauge(self, name):
        print(f"Mock gauge registered: {name}")
        
    def register_counter(self, name):
        print(f"Mock counter registered: {name}")
        
    def register_histogram(self, name):
        print(f"Mock histogram registered: {name}")
        
    def set(self, name, value, tags=None):
        pass
        
    def increment(self, name, tags=None):
        pass
        
    def record(self, name, value, tags=None):
        pass

class MockTraceManager:
    pass

async def test_gpu_profiler():
    """Test GPU profiler functionality."""
    print("Testing GPU Profiler...")
    
    try:
        # Import the GPU profiler directly
        from observability.profiling.gpu_profiler import GPUProfiler
        
        # Create mock container
        container = MockContainer()
        
        # Create GPU profiler
        print("Creating GPU profiler...")
        gpu_profiler = GPUProfiler(container)
        
        print(f"GPU profiler created successfully!")
        print(f"GPU available: {gpu_profiler.gpu_available}")
        print(f"GPU devices: {len(gpu_profiler.gpu_devices)}")
        print(f"Status: {gpu_profiler.status}")
        
        # Test getting GPU status
        print("\nTesting get_gpu_status()...")
        status = await gpu_profiler.get_gpu_status()
        print(f"GPU status: {status}")
        
        # Test getting optimization suggestions
        print("\nTesting get_optimization_suggestions()...")
        suggestions = await gpu_profiler.get_optimization_suggestions()
        print(f"Optimization suggestions: {len(suggestions)} found")
        
        # Test getting component GPU usage
        print("\nTesting get_component_gpu_usage()...")
        usage = gpu_profiler.get_component_gpu_usage()
        print(f"Component GPU usage: {usage}")
        
        # Test profiling start/stop if GPU is available
        if gpu_profiler.gpu_available:
            print("\nTesting profiling start/stop...")
            result = await gpu_profiler.start_profiling()
            print(f"Start profiling result: {result}")
            
            # Wait a bit
            await asyncio.sleep(1)
            
            result = await gpu_profiler.stop_profiling()
            print(f"Stop profiling result: {result}")
        else:
            print("\nSkipping profiling start/stop (no GPU available)")
            
            # Test that starting profiling without GPU raises an error
            try:
                await gpu_profiler.start_profiling()
                print("ERROR: Should have raised RuntimeError")
            except RuntimeError as e:
                print(f"Correctly raised RuntimeError: {e}")
        
        # Test health check
        print("\nTesting health check...")
        health = await gpu_profiler._health_check_callback()
        print(f"Health check: {health}")
        
        print("\nGPU profiler test completed successfully!")
        return True
        
    except Exception as e:
        print(f"GPU profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_kernel_profiling():
    """Test kernel profiling context manager."""
    print("\nTesting kernel profiling context manager...")
    
    try:
        from observability.profiling.gpu_profiler import GPUProfiler
        
        container = MockContainer()
        gpu_profiler = GPUProfiler(container)
        
        # Test the context manager
        print("Testing kernel profiling context manager...")
        with gpu_profiler.profile_kernel("test_kernel", device_id=0, component="test_component") as kernel_info:
            print(f"Kernel info: {kernel_info.kernel_name}")
            # Simulate some work
            await asyncio.sleep(0.1)
        
        print("Kernel profiling context manager test completed!")
        return True
        
    except Exception as e:
        print(f"Kernel profiling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("GPU Profiler Test Suite")
        print("=" * 60)
        
        # Test basic GPU profiler functionality
        test1_result = await test_gpu_profiler()
        
        # Test kernel profiling
        test2_result = await test_kernel_profiling()
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"Basic GPU Profiler: {'PASS' if test1_result else 'FAIL'}")
        print(f"Kernel Profiling: {'PASS' if test2_result else 'FAIL'}")
        
        if test1_result and test2_result:
            print("All tests PASSED!")
            sys.exit(0)
        else:
            print("Some tests FAILED!")
            sys.exit(1)
    
    asyncio.run(main())