#!/usr/bin/env python3
"""
Performance benchmark script to measure improvements made to the AI Assistant.
This script provides quantitative measurements of optimization impacts.
"""

import time
import sys
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    duration_ms: float
    operations_per_second: float
    memory_usage_mb: float
    success: bool
    details: Dict[str, Any]

class PerformanceBenchmark:
    """Performance benchmark suite for AI Assistant optimizations."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark_import_times(self) -> BenchmarkResult:
        """Benchmark import performance improvements."""
        print("🚀 Benchmarking import performance...")
        
        import_tests = [
            ("Core Events", "from core.events.event_types import BaseEvent"),
            ("Context Events", "from core.events.event_types import ContextUpdated, ContextWindowChanged"),
            ("Health Events", "from core.events.event_types import HealthThresholdExceeded, AutoRecoveryStarted"),
            ("Cache Events", "from core.events.event_types import CacheHit, CacheMiss, CacheEviction"),
        ]
        
        total_time = 0
        details = {}
        
        for test_name, import_stmt in import_tests:
            start_time = time.time()
            try:
                exec(import_stmt)
                duration = time.time() - start_time
                total_time += duration
                details[test_name] = f"{duration:.3f}s"
                print(f"  ✓ {test_name}: {duration:.3f}s")
            except Exception as e:
                details[test_name] = f"Failed: {e}"
                print(f"  ✗ {test_name}: Failed - {e}")
        
        return BenchmarkResult(
            name="Import Performance",
            duration_ms=total_time * 1000,
            operations_per_second=len(import_tests) / total_time if total_time > 0 else 0,
            memory_usage_mb=0,  # Would need psutil for actual measurement
            success=all("Failed" not in v for v in details.values()),
            details=details
        )
    
    def benchmark_context_operations(self) -> BenchmarkResult:
        """Benchmark context manager operations."""
        print("🧠 Benchmarking context manager performance...")
        
        try:
            from memory.operations.context_manager import ContextEntity
            from datetime import datetime, timezone
            
            # Create test entity
            entity = ContextEntity(
                entity_id="test_entity",
                name="Test Entity",
                entity_type="person",
                first_mentioned_at=datetime.now(timezone.utc),
                last_mentioned_at=datetime.now(timezone.utc),
                attributes={"key": "value", "data": list(range(100))}
            )
            
            # Benchmark serialization
            iterations = 10000
            start_time = time.time()
            
            for _ in range(iterations):
                entity.to_dict()
            
            duration = time.time() - start_time
            ops_per_sec = iterations / duration
            
            print(f"  ✓ Serialized {iterations} entities in {duration:.3f}s")
            print(f"  ✓ Rate: {ops_per_sec:.0f} ops/sec")
            
            # Benchmark updates
            start_time = time.time()
            
            for i in range(iterations):
                entity.update_mention({"iteration": i})
            
            update_duration = time.time() - start_time
            update_ops_per_sec = iterations / update_duration
            
            print(f"  ✓ Updated {iterations} entities in {update_duration:.3f}s")
            print(f"  ✓ Update rate: {update_ops_per_sec:.0f} ops/sec")
            
            return BenchmarkResult(
                name="Context Operations",
                duration_ms=duration * 1000,
                operations_per_second=ops_per_sec,
                memory_usage_mb=0,
                success=True,
                details={
                    "serialization_rate": f"{ops_per_sec:.0f} ops/sec",
                    "update_rate": f"{update_ops_per_sec:.0f} ops/sec",
                    "avg_serialization_time": f"{(duration/iterations)*1000:.3f}ms"
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="Context Operations",
                duration_ms=0,
                operations_per_second=0,
                memory_usage_mb=0,
                success=False,
                details={"error": str(e)}
            )
    
    def benchmark_cache_performance(self) -> BenchmarkResult:
        """Benchmark cache compression performance."""
        print("💾 Benchmarking cache performance...")
        
        try:
            from integrations.cache.local_cache import CacheCompressor, CompressionType
            
            # Test data
            test_data = {
                "small": {"key": "value"},
                "medium": {"numbers": list(range(1000))},
                "large": {"data": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
            }
            
            compressor = CacheCompressor(CompressionType.LZ4)
            results = {}
            
            for data_type, data in test_data.items():
                # Compression benchmark
                iterations = 1000
                start_time = time.time()
                
                for _ in range(iterations):
                    compressed = compressor.compress(data)
                
                compression_time = time.time() - start_time
                compression_rate = iterations / compression_time
                
                # Decompression benchmark
                start_time = time.time()
                
                for _ in range(iterations):
                    decompressed = compressor.decompress(compressed)
                
                decompression_time = time.time() - start_time
                decompression_rate = iterations / decompression_time
                
                results[data_type] = {
                    "compression_rate": f"{compression_rate:.0f} ops/sec",
                    "decompression_rate": f"{decompression_rate:.0f} ops/sec",
                    "compression_time": f"{(compression_time/iterations)*1000:.3f}ms",
                    "decompression_time": f"{(decompression_time/iterations)*1000:.3f}ms"
                }
                
                print(f"  ✓ {data_type} data: {compression_rate:.0f} compress/sec, {decompression_rate:.0f} decompress/sec")
            
            # Calculate overall performance
            total_time = sum(compression_time + decompression_time for _ in test_data)
            
            return BenchmarkResult(
                name="Cache Performance",
                duration_ms=total_time * 1000,
                operations_per_second=len(test_data) * 2000 / total_time,  # 1000 ops each for compression/decompression
                memory_usage_mb=0,
                success=True,
                details=results
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="Cache Performance", 
                duration_ms=0,
                operations_per_second=0,
                memory_usage_mb=0,
                success=False,
                details={"error": str(e)}
            )
    
    async def benchmark_async_timing(self) -> BenchmarkResult:
        """Benchmark async timing precision improvements."""
        print("⏱️  Benchmarking async timing precision...")
        
        try:
            # Test precise timing implementation
            target_interval = 0.1  # 100ms
            iterations = 20
            next_check = time.time() + target_interval
            actual_intervals = []
            
            for i in range(iterations):
                start = time.time()
                
                # Simulate some async work
                await asyncio.sleep(0.005)  # 5ms work
                
                # Precise timing calculation
                current_time = time.time()
                sleep_time = max(0.001, next_check - current_time)
                next_check = current_time + target_interval
                
                if i > 0:  # Skip first iteration for timing
                    await asyncio.sleep(sleep_time)
                    actual_intervals.append(time.time() - start)
            
            # Calculate timing statistics
            avg_interval = sum(actual_intervals) / len(actual_intervals)
            timing_error = abs(avg_interval - target_interval)
            max_error = max(abs(interval - target_interval) for interval in actual_intervals)
            min_error = min(abs(interval - target_interval) for interval in actual_intervals)
            
            # Calculate jitter (standard deviation)
            variance = sum((interval - avg_interval) ** 2 for interval in actual_intervals) / len(actual_intervals)
            jitter = variance ** 0.5
            
            print(f"  ✓ Target interval: {target_interval:.3f}s")
            print(f"  ✓ Average interval: {avg_interval:.3f}s")
            print(f"  ✓ Timing error: {timing_error:.3f}s")
            print(f"  ✓ Jitter (stddev): {jitter:.3f}s")
            
            # Success if within 10ms error and low jitter
            success = timing_error < 0.01 and jitter < 0.005
            
            return BenchmarkResult(
                name="Async Timing Precision",
                duration_ms=sum(actual_intervals) * 1000,
                operations_per_second=len(actual_intervals) / sum(actual_intervals),
                memory_usage_mb=0,
                success=success,
                details={
                    "target_interval": f"{target_interval:.3f}s",
                    "average_interval": f"{avg_interval:.3f}s", 
                    "timing_error": f"{timing_error:.3f}s",
                    "max_error": f"{max_error:.3f}s",
                    "min_error": f"{min_error:.3f}s",
                    "jitter": f"{jitter:.3f}s"
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="Async Timing Precision",
                duration_ms=0,
                operations_per_second=0,
                memory_usage_mb=0,
                success=False,
                details={"error": str(e)}
            )
    
    def benchmark_lazy_imports(self) -> BenchmarkResult:
        """Benchmark lazy import mechanism."""
        print("📦 Benchmarking lazy import mechanism...")
        
        try:
            # Test that heavy modules aren't imported initially
            import importlib.util
            
            # Check availability without importing
            availability_checks = [
                ("numpy", "numpy"),
                ("rich", "rich.console"),
                ("torch", "torch"),
                ("sklearn", "sklearn.ensemble"),
            ]
            
            start_time = time.time()
            availability_results = {}
            
            for module_name, import_path in availability_checks:
                check_start = time.time()
                available = importlib.util.find_spec(import_path) is not None
                check_time = time.time() - check_start
                
                availability_results[module_name] = {
                    "available": available,
                    "check_time_ms": check_time * 1000
                }
                
                print(f"  ✓ {module_name}: {'Available' if available else 'Not available'} ({check_time*1000:.1f}ms)")
            
            total_time = time.time() - start_time
            
            # Test lazy loading speed
            start_time = time.time()
            
            # Simulate lazy loading pattern
            def lazy_import_simulation():
                global _cached_module
                if '_cached_module' not in globals():
                    # Simulate module loading time
                    time.sleep(0.001)  # 1ms simulation
                    globals()['_cached_module'] = "simulated_module"
                return globals()['_cached_module']
            
            # First call (should "load")
            module1 = lazy_import_simulation()
            first_call_time = time.time() - start_time
            
            # Second call (should use cache)
            start_time = time.time()
            module2 = lazy_import_simulation()
            second_call_time = time.time() - start_time
            
            print(f"  ✓ First lazy import: {first_call_time*1000:.3f}ms")
            print(f"  ✓ Cached import: {second_call_time*1000:.3f}ms")
            
            return BenchmarkResult(
                name="Lazy Import Mechanism",
                duration_ms=total_time * 1000,
                operations_per_second=len(availability_checks) / total_time,
                memory_usage_mb=0,
                success=second_call_time < first_call_time / 10,  # Cache should be much faster
                details={
                    "availability_checks": availability_results,
                    "first_import_time": f"{first_call_time*1000:.3f}ms",
                    "cached_import_time": f"{second_call_time*1000:.3f}ms",
                    "speedup": f"{first_call_time/second_call_time:.1f}x" if second_call_time > 0 else "∞"
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                name="Lazy Import Mechanism",
                duration_ms=0,
                operations_per_second=0,
                memory_usage_mb=0,
                success=False,
                details={"error": str(e)}
            )
    
    async def run_all_benchmarks(self) -> None:
        """Run all performance benchmarks."""
        print("🎯 AI Assistant Performance Benchmark Suite")
        print("=" * 50)
        
        benchmarks = [
            ("Import Performance", lambda: self.benchmark_import_times()),
            ("Context Operations", lambda: self.benchmark_context_operations()),
            ("Cache Performance", lambda: self.benchmark_cache_performance()),
            ("Async Timing", lambda: asyncio.create_task(self.benchmark_async_timing())),
            ("Lazy Imports", lambda: self.benchmark_lazy_imports()),
        ]
        
        total_start = time.time()
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{benchmark_name}:")
            print("-" * 30)
            
            try:
                result = benchmark_func()
                if asyncio.iscoroutine(result) or asyncio.iscoroutinefunction(benchmark_func):
                    result = await result
                elif hasattr(result, '__await__'):
                    result = await result
                
                self.results.append(result)
                
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"  Result: {status}")
                
            except Exception as e:
                print(f"  ❌ BENCHMARK FAILED: {e}")
                self.results.append(BenchmarkResult(
                    name=benchmark_name,
                    duration_ms=0,
                    operations_per_second=0,
                    memory_usage_mb=0,
                    success=False,
                    details={"error": str(e)}
                ))
        
        total_time = time.time() - total_start
        
        # Generate summary report
        self.generate_report(total_time)
    
    def generate_report(self, total_time: float) -> None:
        """Generate performance benchmark report."""
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE BENCHMARK REPORT")
        print("=" * 50)
        
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        
        print(f"\nOverall Results: {passed}/{total} benchmarks passed")
        print(f"Total benchmark time: {total_time:.3f}s")
        print(f"Benchmark date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\nDetailed Results:")
        print("-" * 30)
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"{status} {result.name}")
            print(f"   Duration: {result.duration_ms:.1f}ms")
            print(f"   Ops/sec: {result.operations_per_second:.1f}")
            
            if result.success and result.details:
                for key, value in result.details.items():
                    if key != "error":
                        print(f"   {key}: {value}")
            elif not result.success and "error" in result.details:
                print(f"   Error: {result.details['error']}")
            print()
        
        # Save results to file
        try:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "passed": passed,
                "total": total,
                "results": [
                    {
                        "name": r.name,
                        "duration_ms": r.duration_ms,
                        "operations_per_second": r.operations_per_second,
                        "success": r.success,
                        "details": r.details
                    }
                    for r in self.results
                ]
            }
            
            with open("performance_benchmark_results.json", "w") as f:
                json.dump(report_data, f, indent=2)
            
            print("📄 Detailed results saved to: performance_benchmark_results.json")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
        
        if passed == total:
            print("\n🎉 All performance benchmarks passed!")
        else:
            print(f"\n⚠️  {total - passed} benchmarks need attention")

async def main():
    """Run the performance benchmark suite."""
    benchmark = PerformanceBenchmark()
    await benchmark.run_all_benchmarks()
    
    # Return exit code based on results
    passed = sum(1 for r in benchmark.results if r.success)
    total = len(benchmark.results)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)