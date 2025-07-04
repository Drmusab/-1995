#!/usr/bin/env python3
"""
Simple test script for GPU Profiler module only
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
        print(f"Event emitted: {event.__class__.__name__}")
        
    def subscribe(self, event_type, handler):
        print(f"Subscribed: {event_type}")

class MockErrorHandler:
    pass

class MockHealthCheck:
    def register_component(self, name, callback):
        print(f"Health check registered: {name}")

class MockMetricsCollector:
    def register_gauge(self, name):
        pass
        
    def register_counter(self, name):
        pass
        
    def register_histogram(self, name):
        pass
        
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
        # Import the GPU profiler module directly
        sys.path.insert(0, str(Path(__file__).parent / "src" / "observability" / "profiling"))
        from gpu_profiler import GPUProfiler
        
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
        print(f"GPU status available: {status.get('available', False)}")
        
        # Test getting optimization suggestions
        print("\nTesting get_optimization_suggestions()...")
        suggestions = await gpu_profiler.get_optimization_suggestions()
        print(f"Optimization suggestions: {len(suggestions)} found")
        
        # Test getting component GPU usage
        print("\nTesting get_component_gpu_usage()...")
        usage = gpu_profiler.get_component_gpu_usage()
        print(f"Component GPU usage keys: {list(usage.keys())}")
        
        # Test profiling start/stop if GPU is available
        if gpu_profiler.gpu_available:
            print("\nTesting profiling start/stop...")
            result = await gpu_profiler.start_profiling()
            print(f"Start profiling result: {result}")
            
            # Wait a bit
            await asyncio.sleep(0.5)
            
            result = await gpu_profiler.stop_profiling()
            print(f"Stop profiling result: {result}")
        else:
            print("\nSkipping profiling start/stop (no GPU available)")
            
            # Test that starting profiling without GPU raises an error
            try:
                await gpu_profiler.start_profiling()
                print("ERROR: Should have raised RuntimeError")
            except RuntimeError as e:
                print(f"Correctly raised RuntimeError: {str(e)[:100]}")
        
        # Test health check
        print("\nTesting health check...")
        health = await gpu_profiler._health_check_callback()
        print(f"Health check status: {health.get('status', 'unknown')}")
        
        # Test kernel profiling context manager
        print("\nTesting kernel profiling context manager...")
        with gpu_profiler.profile_kernel("test_kernel", device_id=0, component="test_component") as kernel_info:
            print(f"Kernel info created: {kernel_info.kernel_name}")
            # Simulate some work
            await asyncio.sleep(0.1)
        
        print("\nGPU profiler test completed successfully!")
        return True
        
    except Exception as e:
        print(f"GPU profiler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("=" * 60)
        print("GPU Profiler Direct Test")
        print("=" * 60)
        
        result = await test_gpu_profiler()
        
        print("\n" + "=" * 60)
        print(f"Test Result: {'PASS' if result else 'FAIL'}")
        
        sys.exit(0 if result else 1)
    
    asyncio.run(main())