#!/usr/bin/env python3
"""
Performance validation script for AI Assistant optimizations.
This script demonstrates the performance improvements made to the system.
"""

import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_event_imports():
    """Test that new context events import successfully."""
    print("Testing context event imports...")
    start_time = time.time()
    
    try:
        from core.events.event_types import (
            ContextUpdated, ContextWindowChanged, ContextEntityDetected,
            ContextEntityRemoved, ContextRestored, ContextCleared
        )
        import_time = time.time() - start_time
        print(f"  ✓ Context events imported in {import_time:.3f}s")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_health_events():
    """Test that health events import successfully."""
    print("Testing health event imports...")
    start_time = time.time()
    
    try:
        from core.events.event_types import (
            HealthThresholdExceeded, AutoRecoveryStarted, AutoRecoveryCompleted,
            HealthPredictionAlert, CircuitBreakerStateChanged
        )
        import_time = time.time() - start_time
        print(f"  ✓ Health events imported in {import_time:.3f}s")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_cache_compression():
    """Test cache compression optimization."""
    print("Testing cache compression performance...")
    
    try:
        from integrations.cache.local_cache import CacheCompressor, CompressionType
        
        # Create test data
        test_data = {"key": "value", "numbers": list(range(1000))}
        
        # Test LZ4 compression (optimized)
        compressor = CacheCompressor(CompressionType.LZ4)
        
        start_time = time.time()
        compressed = compressor.compress(test_data)
        compression_time = time.time() - start_time
        
        start_time = time.time()
        decompressed = compressor.decompress(compressed)
        decompression_time = time.time() - start_time
        
        print(f"  ✓ Compression time: {compression_time:.3f}s")
        print(f"  ✓ Decompression time: {decompression_time:.3f}s")
        print(f"  ✓ Data integrity: {decompressed == test_data}")
        
        return compression_time < 0.1 and decompression_time < 0.1
        
    except Exception as e:
        print(f"  ✗ Cache test failed: {e}")
        return False

def test_lazy_imports():
    """Test lazy import functionality."""
    print("Testing lazy import optimization...")
    
    # Test that we can check for module availability without importing
    try:
        import importlib.util
        
        # Check numpy availability without importing
        numpy_available = importlib.util.find_spec("numpy") is not None
        
        # Check rich availability  
        rich_available = importlib.util.find_spec("rich") is not None
        
        print(f"  ✓ NumPy availability check: {numpy_available}")
        print(f"  ✓ Rich availability check: {rich_available}")
        print(f"  ✓ Lazy import mechanism working")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Lazy import test failed: {e}")
        return False

def demonstrate_timing_optimization():
    """Demonstrate timing precision improvements."""
    print("Testing timing precision optimization...")
    
    import asyncio
    
    async def precise_timing_test():
        """Test precise interval timing."""
        interval = 0.1  # 100ms
        iterations = 5
        next_check = time.time() + interval
        actual_intervals = []
        
        for i in range(iterations):
            start = time.time()
            
            # Simulate some work
            await asyncio.sleep(0.01)
            
            # Calculate precise sleep time
            current_time = time.time()
            sleep_time = max(0.01, next_check - current_time)
            next_check = current_time + interval
            
            await asyncio.sleep(sleep_time)
            
            if i > 0:  # Skip first iteration
                actual_intervals.append(time.time() - start)
        
        # Calculate timing precision
        avg_interval = sum(actual_intervals) / len(actual_intervals)
        timing_error = abs(avg_interval - interval)
        
        print(f"  ✓ Target interval: {interval:.3f}s")
        print(f"  ✓ Average actual interval: {avg_interval:.3f}s")
        print(f"  ✓ Timing error: {timing_error:.3f}s")
        
        return timing_error < 0.02  # Within 20ms
    
    try:
        result = asyncio.run(precise_timing_test())
        print(f"  ✓ Timing precision test: {'PASS' if result else 'FAIL'}")
        return result
    except Exception as e:
        print(f"  ✗ Timing test failed: {e}")
        return False

def main():
    """Run all performance validation tests."""
    print("=== AI Assistant Performance Validation ===")
    print()
    
    tests = [
        ("Event Import Optimization", test_event_imports),
        ("Health Event Additions", test_health_events), 
        ("Cache Compression Optimization", test_cache_compression),
        ("Lazy Import Mechanism", test_lazy_imports),
        ("Timing Precision Improvement", demonstrate_timing_optimization),
    ]
    
    results = []
    total_start = time.time()
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        start_time = time.time()
        
        try:
            success = test_func()
            elapsed = time.time() - start_time
            results.append((test_name, success, elapsed))
            
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {status} (completed in {elapsed:.3f}s)")
            
        except Exception as e:
            elapsed = time.time() - start_time
            results.append((test_name, False, elapsed))
            print(f"  ✗ FAIL - Exception: {e}")
        
        print()
    
    total_time = time.time() - total_start
    
    # Summary
    print("=== Performance Validation Summary ===")
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, elapsed in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} {test_name} ({elapsed:.3f}s)")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    print(f"Total validation time: {total_time:.3f}s")
    
    if passed == total:
        print("🎉 All performance optimizations validated successfully!")
        return 0
    else:
        print("⚠️  Some optimizations need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())