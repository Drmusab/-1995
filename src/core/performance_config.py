"""
Performance monitoring configuration for AI Assistant
Author: Drmusab
Last Modified: 2025-08-01

This module provides performance monitoring configuration and utilities.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
from pathlib import Path


@dataclass
class PerformanceThresholds:
    """Performance thresholds for monitoring."""
    
    # Response time thresholds (seconds)
    response_time_warning: float = 2.0
    response_time_critical: float = 5.0
    
    # Memory usage thresholds (MB)
    memory_usage_warning: float = 512.0
    memory_usage_critical: float = 1024.0
    
    # CPU usage thresholds (percentage)
    cpu_usage_warning: float = 70.0
    cpu_usage_critical: float = 90.0
    
    # Cache hit rate thresholds (percentage)
    cache_hit_rate_warning: float = 80.0
    cache_hit_rate_critical: float = 60.0
    
    # Error rate thresholds (percentage)
    error_rate_warning: float = 1.0
    error_rate_critical: float = 5.0


@dataclass
class OptimizationSettings:
    """Settings for performance optimizations."""
    
    # Lazy loading settings
    enable_lazy_loading: bool = True
    lazy_load_threshold: int = 50  # Number of imports to trigger lazy loading
    
    # Caching settings
    enable_component_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    # Batch processing settings
    enable_batch_processing: bool = True
    batch_size: int = 100
    batch_timeout_ms: int = 100
    
    # String optimization settings
    enable_string_pooling: bool = True
    string_builder_initial_size: int = 1024
    
    # Loop optimization settings
    enable_loop_unrolling: bool = True
    max_nested_loop_depth: int = 3
    
    # Async optimization settings
    enable_async_batching: bool = True
    async_batch_size: int = 50
    concurrent_request_limit: int = 10


@dataclass
class PerformanceConfiguration:
    """Main performance configuration."""
    
    thresholds: PerformanceThresholds
    optimizations: OptimizationSettings
    monitoring_enabled: bool = True
    profiling_enabled: bool = False
    metrics_collection_interval: int = 60  # seconds
    performance_logging: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PerformanceConfiguration':
        """Create configuration from dictionary."""
        thresholds_dict = config_dict.get('thresholds', {})
        optimizations_dict = config_dict.get('optimizations', {})
        
        return cls(
            thresholds=PerformanceThresholds(**thresholds_dict),
            optimizations=OptimizationSettings(**optimizations_dict),
            monitoring_enabled=config_dict.get('monitoring_enabled', True),
            profiling_enabled=config_dict.get('profiling_enabled', False),
            metrics_collection_interval=config_dict.get('metrics_collection_interval', 60),
            performance_logging=config_dict.get('performance_logging', True)
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'PerformanceConfiguration':
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'thresholds': self.thresholds.__dict__,
            'optimizations': self.optimizations.__dict__,
            'monitoring_enabled': self.monitoring_enabled,
            'profiling_enabled': self.profiling_enabled,
            'metrics_collection_interval': self.metrics_collection_interval,
            'performance_logging': self.performance_logging
        }
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Default performance configuration
DEFAULT_PERFORMANCE_CONFIG = PerformanceConfiguration(
    thresholds=PerformanceThresholds(),
    optimizations=OptimizationSettings()
)


# Performance recommendations based on the analysis
PERFORMANCE_RECOMMENDATIONS = [
    {
        "issue": "Excessive imports",
        "description": "Files with >50 imports cause slow startup times",
        "solution": "Implement lazy imports for non-critical components",
        "files_affected": ["core_engine.py", "config_settings.py"],
        "priority": "HIGH",
        "estimated_improvement": "30-50% faster startup"
    },
    {
        "issue": "Nested loops",
        "description": "O(n³) complexity in graph traversal algorithms",
        "solution": "Use batch operations and pre-built lookup tables",
        "files_affected": ["memory_graph.py"],
        "priority": "HIGH", 
        "estimated_improvement": "60-80% faster graph operations"
    },
    {
        "issue": "String concatenation in loops",
        "description": "Inefficient string building in iterative processes",
        "solution": "Use StringIO or join() for string building",
        "files_affected": ["local_cache.py", "context_manager.py"],
        "priority": "MEDIUM",
        "estimated_improvement": "20-40% faster string operations"
    },
    {
        "issue": "Uncompiled regex patterns",
        "description": "Regex compilation overhead in repeated operations",
        "solution": "Pre-compile and cache regex patterns",
        "files_affected": ["multiple"],
        "priority": "MEDIUM",
        "estimated_improvement": "15-25% faster text processing"
    },
    {
        "issue": "Large monolithic files",
        "description": "Files >1500 lines are hard to maintain and slow to load",
        "solution": "Split into focused, smaller modules",
        "files_affected": ["context_manager.py", "commands.py", "cli.py"],
        "priority": "LOW",
        "estimated_improvement": "Better maintainability, 10-20% faster imports"
    }
]


def get_performance_recommendations() -> List[Dict[str, Any]]:
    """Get list of performance recommendations."""
    return PERFORMANCE_RECOMMENDATIONS


def generate_performance_report() -> str:
    """Generate a performance improvement report."""
    report = ["# AI Assistant Performance Improvement Report"]
    report.append(f"Generated with {len(PERFORMANCE_RECOMMENDATIONS)} recommendations")
    report.append("")
    
    for i, rec in enumerate(PERFORMANCE_RECOMMENDATIONS, 1):
        report.append(f"## {i}. {rec['issue']} ({rec['priority']} Priority)")
        report.append(f"**Description**: {rec['description']}")
        report.append(f"**Solution**: {rec['solution']}")
        report.append(f"**Files affected**: {', '.join(rec['files_affected'])}")
        report.append(f"**Estimated improvement**: {rec['estimated_improvement']}")
        report.append("")
    
    return "\n".join(report)