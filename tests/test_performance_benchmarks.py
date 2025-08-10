"""
Performance benchmarking suite for AI Assistant optimizations
Author: Drmusab
Last Modified: 2025-08-10

This module provides comprehensive benchmarking to measure the impact
of performance optimizations and identify areas for improvement.
"""

import asyncio
import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from src.core.lazy_imports import lazy_import, clear_import_cache
from src.core.performance_utils import (
    PerformantStringBuilder, 
    EfficientListOperations,
    CompiledRegexCache,
    PerformanceTimer,
    get_status_formatter
)
from src.core.enhanced_cache import get_memory_cache, memory_cache
from src.core.connection_pool import ConnectionManager, ConnectionConfig


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    memory_peak_mb: float
    memory_current_mb: float
    throughput_per_sec: float


class PerformanceBenchmark:
    """
    Performance benchmarking utility for measuring optimization impact.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def measure_performance(self, name: str, iterations: int = 1):
        """Context manager for measuring performance of code blocks."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Start memory tracing
        tracemalloc.start()
        
        times = []
        start_overall = time.perf_counter()
        
        try:
            for i in range(iterations):
                start_iter = time.perf_counter()
                yield i
                end_iter = time.perf_counter()
                times.append(end_iter - start_iter)
            
            end_overall = time.perf_counter()
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate statistics
            total_time = end_overall - start_overall
            avg_time = total_time / iterations
            min_time = min(times) if times else 0
            max_time = max(times) if times else 0
            throughput = iterations / total_time if total_time > 0 else 0
            
            result = BenchmarkResult(
                name=name,
                iterations=iterations,
                total_time=total_time,
                avg_time=avg_time,
                min_time=min_time,
                max_time=max_time,
                memory_peak_mb=peak / 1024 / 1024,
                memory_current_mb=current / 1024 / 1024,
                throughput_per_sec=throughput
            )
            
            self.results.append(result)
            
        except Exception as e:
            tracemalloc.stop()
            raise
    
    def compare_implementations(self, name: str, old_impl: Callable, 
                              new_impl: Callable, test_data: Any, 
                              iterations: int = 1000) -> Dict[str, BenchmarkResult]:
        """Compare two implementations and return results."""
        results = {}
        
        # Benchmark old implementation
        with self.measure_performance(f"{name}_old", iterations):
            for i in range(iterations):
                old_impl(test_data)
        results["old"] = self.results[-1]
        
        # Benchmark new implementation  
        with self.measure_performance(f"{name}_new", iterations):
            for i in range(iterations):
                new_impl(test_data)
        results["new"] = self.results[-1]
        
        return results
    
    def print_results(self, compare_pairs: List[Tuple[str, str]] = None):
        """Print benchmark results with optional comparisons."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\nBenchmark: {result.name}")
            print(f"  Iterations: {result.iterations:,}")
            print(f"  Total Time: {result.total_time:.4f}s")
            print(f"  Average Time: {result.avg_time*1000:.2f}ms")
            print(f"  Min Time: {result.min_time*1000:.2f}ms")
            print(f"  Max Time: {result.max_time*1000:.2f}ms")
            print(f"  Memory Peak: {result.memory_peak_mb:.2f}MB")
            print(f"  Throughput: {result.throughput_per_sec:.1f} ops/sec")
        
        # Print comparisons
        if compare_pairs:
            print("\n" + "-"*50)
            print("PERFORMANCE COMPARISONS")
            print("-"*50)
            
            for old_name, new_name in compare_pairs:
                old_result = next((r for r in self.results if r.name == old_name), None)
                new_result = next((r for r in self.results if r.name == new_name), None)
                
                if old_result and new_result:
                    time_improvement = (old_result.avg_time - new_result.avg_time) / old_result.avg_time * 100
                    memory_improvement = (old_result.memory_peak_mb - new_result.memory_peak_mb) / old_result.memory_peak_mb * 100
                    throughput_improvement = (new_result.throughput_per_sec - old_result.throughput_per_sec) / old_result.throughput_per_sec * 100
                    
                    print(f"\n{old_name} vs {new_name}:")
                    print(f"  Time improvement: {time_improvement:+.1f}%")
                    print(f"  Memory improvement: {memory_improvement:+.1f}%")
                    print(f"  Throughput improvement: {throughput_improvement:+.1f}%")


class AIAssistantBenchmarks:
    """
    Specific benchmarks for AI Assistant optimizations.
    """
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
    
    def benchmark_string_operations(self):
        """Benchmark string building optimizations."""
        test_data = ["Hello", "World", "Test", "String"] * 100
        
        def old_string_concat(strings):
            result = ""
            for s in strings:
                result += s + " "
            return result
        
        def new_string_builder(strings):
            builder = PerformantStringBuilder()
            for s in strings:
                builder.append(s).append(" ")
            return builder.build()
        
        def join_method(strings):
            return " ".join(strings) + " "
        
        # Test each method
        with self.benchmark.measure_performance("string_concat_old", 1000):
            for i in range(1000):
                old_string_concat(test_data)
        
        with self.benchmark.measure_performance("string_builder_new", 1000):
            for i in range(1000):
                new_string_builder(test_data)
        
        with self.benchmark.measure_performance("string_join_method", 1000):
            for i in range(1000):
                join_method(test_data)
    
    def benchmark_list_operations(self):
        """Benchmark list operation optimizations."""
        test_data = list(range(1000))
        new_items = list(range(1000, 2000))
        
        def old_prepend_items(target, items):
            for item in reversed(items):
                target.insert(0, item)
            return target
        
        def new_prepend_items(target, items):
            return EfficientListOperations.efficient_prepend(target, items)
        
        # Test prepend operations
        with self.benchmark.measure_performance("list_prepend_old", 100):
            for i in range(100):
                old_prepend_items(test_data.copy(), new_items[:10])
        
        with self.benchmark.measure_performance("list_prepend_new", 100):
            for i in range(100):
                new_prepend_items(test_data.copy(), new_items[:10])
    
    def benchmark_regex_operations(self):
        """Benchmark regex caching optimizations."""
        test_pattern = r'\b\w+@\w+\.\w+\b'
        test_text = "Contact us at test@example.com or admin@test.org for support." * 100
        
        import re
        
        def old_regex_ops(pattern, text):
            matches = []
            for _ in range(10):
                matches.extend(re.findall(pattern, text))
            return matches
        
        def new_regex_ops(pattern, text):
            cache = CompiledRegexCache()
            matches = []
            for _ in range(10):
                matches.extend(cache.findall(pattern, text))
            return matches
        
        with self.benchmark.measure_performance("regex_old", 100):
            for i in range(100):
                old_regex_ops(test_pattern, test_text)
        
        with self.benchmark.measure_performance("regex_cached", 100):
            for i in range(100):
                new_regex_ops(test_pattern, test_text)
    
    def benchmark_caching_operations(self):
        """Benchmark caching system performance."""
        cache = get_memory_cache()
        
        # Generate test data
        test_keys = [f"key_{i}" for i in range(1000)]
        test_values = [f"value_{i}" * 100 for i in range(1000)]
        
        def cache_operations():
            # Set operations
            for key, value in zip(test_keys[:100], test_values[:100]):
                cache.set(key, value)
            
            # Get operations
            results = []
            for key in test_keys[:100]:
                result = cache.get(key)
                results.append(result)
            
            return results
        
        with self.benchmark.measure_performance("cache_operations", 10):
            for i in range(10):
                cache_operations()
                cache.clear()  # Clear for next iteration
    
    def benchmark_lazy_imports(self):
        """Benchmark lazy import performance."""
        # Clear cache before test
        clear_import_cache()
        
        def eager_imports():
            import json
            import time
            import os
            import sys
            import asyncio
            return [json, time, os, sys, asyncio]
        
        def lazy_imports():
            json_lazy = lazy_import('json')
            time_lazy = lazy_import('time')
            os_lazy = lazy_import('os')
            sys_lazy = lazy_import('sys')
            asyncio_lazy = lazy_import('asyncio')
            return [json_lazy, time_lazy, os_lazy, sys_lazy, asyncio_lazy]
        
        def lazy_imports_with_usage():
            json_lazy = lazy_import('json')
            time_lazy = lazy_import('time')
            # Use the imports to trigger actual loading
            json_lazy.dumps({"test": "data"})
            time_lazy.time()
            return [json_lazy, time_lazy]
        
        with self.benchmark.measure_performance("imports_eager", 100):
            for i in range(100):
                eager_imports()
        
        with self.benchmark.measure_performance("imports_lazy", 100):
            for i in range(100):
                lazy_imports()
        
        with self.benchmark.measure_performance("imports_lazy_used", 100):
            for i in range(100):
                lazy_imports_with_usage()
    
    async def benchmark_async_operations(self):
        """Benchmark async operation optimizations."""
        
        async def old_sequential_processing(items):
            results = []
            for item in items:
                # Simulate async work
                await asyncio.sleep(0.001)
                results.append(item * 2)
            return results
        
        async def new_batch_processing(items, batch_size=10):
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_tasks = [asyncio.create_task(self._async_work(item)) for item in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
            return results
        
        test_items = list(range(50))
        
        with self.benchmark.measure_performance("async_sequential", 5):
            for i in range(5):
                await old_sequential_processing(test_items)
        
        with self.benchmark.measure_performance("async_batched", 5):
            for i in range(5):
                await new_batch_processing(test_items)
    
    async def _async_work(self, item):
        """Simulate async work."""
        await asyncio.sleep(0.001)
        return item * 2
    
    def benchmark_status_formatting(self):
        """Benchmark status message formatting optimizations."""
        formatter = get_status_formatter()
        
        def old_string_formatting():
            messages = []
            for i in range(100):
                msg = f"[yellow]System: Processing item {i}[/yellow]"
                messages.append(msg)
            return messages
        
        def new_optimized_formatting():
            messages = []
            for i in range(100):
                msg = formatter.format_system_message(f"Processing item {i}")
                messages.append(msg)
            return messages
        
        with self.benchmark.measure_performance("formatting_old", 100):
            for i in range(100):
                old_string_formatting()
        
        with self.benchmark.measure_performance("formatting_new", 100):
            for i in range(100):
                new_optimized_formatting()
    
    def run_all_benchmarks(self) -> PerformanceBenchmark:
        """Run all benchmarks and return results."""
        print("Running AI Assistant Performance Benchmarks...")
        
        # Run synchronous benchmarks
        self.benchmark_string_operations()
        self.benchmark_list_operations()
        self.benchmark_regex_operations()
        self.benchmark_caching_operations()
        self.benchmark_lazy_imports()
        self.benchmark_status_formatting()
        
        return self.benchmark
    
    async def run_async_benchmarks(self) -> PerformanceBenchmark:
        """Run async benchmarks."""
        await self.benchmark_async_operations()
        return self.benchmark


def run_performance_tests():
    """Run comprehensive performance tests."""
    benchmarks = AIAssistantBenchmarks()
    
    # Run synchronous benchmarks
    sync_results = benchmarks.run_all_benchmarks()
    
    # Run async benchmarks
    async def run_async():
        await benchmarks.run_async_benchmarks()
        return benchmarks.benchmark
    
    async_results = asyncio.run(run_async())
    
    # Print results with comparisons
    comparisons = [
        ("string_concat_old", "string_builder_new"),
        ("string_concat_old", "string_join_method"),
        ("list_prepend_old", "list_prepend_new"),
        ("regex_old", "regex_cached"),
        ("imports_eager", "imports_lazy"),
        ("formatting_old", "formatting_new"),
        ("async_sequential", "async_batched"),
    ]
    
    sync_results.print_results(comparisons)
    
    return sync_results


# Pytest integration for automated testing
class TestPerformanceOptimizations:
    """Pytest test cases for performance optimizations."""
    
    def test_string_builder_performance(self, benchmark):
        """Test string builder performance with pytest-benchmark."""
        test_strings = ["test"] * 1000
        
        def old_concat():
            result = ""
            for s in test_strings:
                result += s
            return result
        
        def new_builder():
            builder = PerformantStringBuilder()
            for s in test_strings:
                builder.append(s)
            return builder.build()
        
        # Benchmark both approaches
        old_result = benchmark.pedantic(old_concat, iterations=10, rounds=3)
        new_result = benchmark.pedantic(new_builder, iterations=10, rounds=3)
        
        # Verify results are equivalent
        assert old_concat() == new_builder()
    
    def test_cache_performance(self, benchmark):
        """Test cache performance."""
        cache = get_memory_cache()
        
        @memory_cache(ttl=60)
        def expensive_operation(x):
            time.sleep(0.001)  # Simulate expensive operation
            return x * x
        
        def test_with_cache():
            results = []
            for i in range(100):
                result = expensive_operation(i % 10)  # Many cache hits
                results.append(result)
            return results
        
        def test_without_cache():
            results = []
            for i in range(100):
                time.sleep(0.001)
                result = (i % 10) * (i % 10)
                results.append(result)
            return results
        
        # Clear cache before test
        cache.clear()
        
        # Benchmark
        benchmark.pedantic(test_with_cache, iterations=1, rounds=1)
    
    @pytest.mark.asyncio
    async def test_async_performance(self, benchmark):
        """Test async operation performance."""
        
        async def async_operation():
            await asyncio.sleep(0.001)
            return "result"
        
        async def sequential_operations():
            results = []
            for _ in range(10):
                result = await async_operation()
                results.append(result)
            return results
        
        async def concurrent_operations():
            tasks = [async_operation() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            return results
        
        # Benchmark concurrent version
        result = await benchmark.pedantic(concurrent_operations, iterations=1, rounds=1)
        assert len(result) == 10


if __name__ == "__main__":
    # Run benchmarks when script is executed directly
    run_performance_tests()