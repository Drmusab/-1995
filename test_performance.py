#!/usr/bin/env python3
"""
Quick performance test script to validate optimizations
Author: Drmusab
Last Modified: 2025-08-10
"""

import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_import_performance():
    """Test import performance with lazy imports."""
    print("Testing import performance...")
    
    # Test lazy imports
    start_time = time.perf_counter()
    from src.core.lazy_imports import lazy_import
    
    # Create lazy imports
    numpy_lazy = lazy_import('numpy')
    json_lazy = lazy_import('json')
    asyncio_lazy = lazy_import('asyncio')
    
    lazy_import_time = time.perf_counter() - start_time
    print(f"Lazy imports created in: {lazy_import_time:.4f}s")
    
    # Test actual usage
    start_time = time.perf_counter()
    result = json_lazy.dumps({"test": "data"})
    usage_time = time.perf_counter() - start_time
    print(f"First lazy import usage: {usage_time:.4f}s")

def test_string_performance():
    """Test string building performance."""
    print("\nTesting string building performance...")
    
    from src.core.performance_utils import PerformantStringBuilder
    
    test_data = ["Hello", "World", "Test"] * 1000
    
    # Test old way (string concatenation)
    start_time = time.perf_counter()
    result_old = ""
    for s in test_data:
        result_old += s + " "
    old_time = time.perf_counter() - start_time
    
    # Test new way (string builder)
    start_time = time.perf_counter()
    builder = PerformantStringBuilder()
    for s in test_data:
        builder.append(s).append(" ")
    result_new = builder.build()
    new_time = time.perf_counter() - start_time
    
    improvement = ((old_time - new_time) / old_time) * 100
    print(f"String concatenation: {old_time:.4f}s")
    print(f"String builder: {new_time:.4f}s")
    print(f"Improvement: {improvement:.1f}%")
    
    # Verify results are equivalent
    assert result_old == result_new

def test_caching_performance():
    """Test caching performance."""
    print("\nTesting caching performance...")
    
    from src.core.enhanced_cache import get_memory_cache, memory_cache
    
    # Create an expensive function
    call_count = 0
    
    @memory_cache(ttl=60)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        time.sleep(0.001)  # Simulate work
        return x * x
    
    # Test with caching
    start_time = time.perf_counter()
    for i in range(100):
        result = expensive_function(i % 10)  # Many cache hits
    cached_time = time.perf_counter() - start_time
    
    print(f"Function calls made: {call_count}")
    print(f"Time with caching: {cached_time:.4f}s")
    
    # Clear cache and test without caching
    cache = get_memory_cache()
    cache.clear()
    call_count = 0
    
    def expensive_function_no_cache(x):
        nonlocal call_count
        call_count += 1
        time.sleep(0.001)
        return x * x
    
    start_time = time.perf_counter()
    for i in range(100):
        result = expensive_function_no_cache(i % 10)
    uncached_time = time.perf_counter() - start_time
    
    improvement = ((uncached_time - cached_time) / uncached_time) * 100
    print(f"Function calls made: {call_count}")
    print(f"Time without caching: {uncached_time:.4f}s")
    print(f"Improvement: {improvement:.1f}%")

def test_status_formatting():
    """Test status formatting performance."""
    print("\nTesting status formatting performance...")
    
    from src.core.performance_utils import get_status_formatter
    
    formatter = get_status_formatter()
    
    # Test old way
    start_time = time.perf_counter()
    messages = []
    for i in range(1000):
        msg = f"[yellow]System: Processing item {i}[/yellow]"
        messages.append(msg)
    old_time = time.perf_counter() - start_time
    
    # Test new way
    start_time = time.perf_counter()
    new_messages = []
    for i in range(1000):
        msg = formatter.format_system_message(f"Processing item {i}")
        new_messages.append(msg)
    new_time = time.perf_counter() - start_time
    
    improvement = ((old_time - new_time) / old_time) * 100
    print(f"Old formatting: {old_time:.4f}s")
    print(f"New formatting: {new_time:.4f}s")
    print(f"Improvement: {improvement:.1f}%")

def main():
    """Run all performance tests."""
    print("AI Assistant Performance Optimization Tests")
    print("=" * 50)
    
    test_import_performance()
    test_string_performance()
    test_caching_performance()
    test_status_formatting()
    
    print("\n" + "=" * 50)
    print("All performance tests completed successfully!")

if __name__ == "__main__":
    main()