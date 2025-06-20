"""
Response compression middleware for the AI Assistant API.

This middleware provides intelligent compression for API responses,
integrating with the core system's event bus and configuration management.
"""

import gzip
import zlib
import brotli
from typing import Optional, List, Dict, Any, Union, Callable
from enum import Enum
import asyncio
import logging
from functools import wraps
import json
import time
from dataclasses import dataclass

from ..core.events.event_bus import EventBus
from ..core.events.event_types import CompressionEvent, APIEvent
from ..core.config.settings.base import get_settings
from ..observability.metrics import MetricsCollector
from ..observability.logging.config import get_logger


class CompressionType(Enum):
    """Supported compression algorithms."""
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "br"
    NONE = "identity"


class CompressionLevel(Enum):
    """Compression levels for different use cases."""
    FASTEST = 1
    BALANCED = 6
    BEST = 9


@dataclass
class CompressionStats:
    """Compression statistics for monitoring."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    algorithm: CompressionType
    level: int


class CompressionStrategy:
    """Base class for compression strategies."""
    
    def __init__(self, level: CompressionLevel = CompressionLevel.BALANCED):
        self.level = level.value
    
    def should_compress(self, content: bytes, content_type: str) -> bool:
        """Determine if content should be compressed."""
        # Don't compress already compressed formats
        non_compressible = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'video/mp4', 'video/mpeg', 'audio/mpeg', 'audio/ogg',
            'application/zip', 'application/gzip', 'application/x-rar'
        }
        
        # Don't compress very small responses
        if len(content) < 1024:
            return False
            
        # Don't compress already compressed content types
        if any(nc in content_type.lower() for nc in non_compressible):
            return False
            
        return True
    
    def compress(self, content: bytes) -> bytes:
        """Compress content using the strategy's algorithm."""
        raise NotImplementedError
    
    def get_encoding_header(self) -> str:
        """Get the Content-Encoding header value."""
        raise NotImplementedError


class GzipStrategy(CompressionStrategy):
    """GZIP compression strategy."""
    
    def compress(self, content: bytes) -> bytes:
        return gzip.compress(content, compresslevel=self.level)
    
    def get_encoding_header(self) -> str:
        return CompressionType.GZIP.value


class DeflateStrategy(CompressionStrategy):
    """Deflate compression strategy."""
    
    def compress(self, content: bytes) -> bytes:
        return zlib.compress(content, level=self.level)
    
    def get_encoding_header(self) -> str:
        return CompressionType.DEFLATE.value


class BrotliStrategy(CompressionStrategy):
    """Brotli compression strategy."""
    
    def __init__(self, level: CompressionLevel = CompressionLevel.BALANCED):
        # Brotli levels are 0-11, map our levels appropriately
        level_mapping = {
            CompressionLevel.FASTEST.value: 1,
            CompressionLevel.BALANCED.value: 6,
            CompressionLevel.BEST.value: 11
        }
        self.level = level_mapping.get(level.value, 6)
    
    def compress(self, content: bytes) -> bytes:
        return brotli.compress(content, quality=self.level)
    
    def get_encoding_header(self) -> str:
        return CompressionType.BROTLI.value


class AdaptiveCompressionStrategy:
    """Adaptive compression that chooses the best algorithm based on content."""
    
    def __init__(self, 
                 strategies: Optional[List[CompressionStrategy]] = None,
                 sample_size: int = 1024):
        self.strategies = strategies or [
            BrotliStrategy(CompressionLevel.BALANCED),
            GzipStrategy(CompressionLevel.BALANCED),
            DeflateStrategy(CompressionLevel.BALANCED)
        ]
        self.sample_size = sample_size
        self._performance_cache: Dict[str, CompressionStrategy] = {}
    
    def choose_best_strategy(self, content: bytes, content_type: str) -> CompressionStrategy:
        """Choose the best compression strategy for the given content."""
        # Use cache for similar content types
        cache_key = f"{content_type}_{len(content)//10000}"
        if cache_key in self._performance_cache:
            return self._performance_cache[cache_key]
        
        # For large content, test with a sample
        sample = content[:self.sample_size] if len(content) > self.sample_size else content
        
        best_strategy = None
        best_ratio = 0
        
        for strategy in self.strategies:
            if not strategy.should_compress(sample, content_type):
                continue
                
            try:
                start_time = time.time()
                compressed = strategy.compress(sample)
                compression_time = time.time() - start_time
                
                ratio = len(compressed) / len(sample)
                # Factor in compression speed for scoring
                score = (1 - ratio) - (compression_time * 0.1)
                
                if score > best_ratio:
                    best_ratio = score
                    best_strategy = strategy
                    
            except Exception:
                continue
        
        if best_strategy:
            self._performance_cache[cache_key] = best_strategy
            return best_strategy
        
        # Fallback to GZIP
        return GzipStrategy()


class CompressionMiddleware:
    """
    Compression middleware for API responses.
    
    Integrates with the AI Assistant's core system for configuration,
    events, and monitoring.
    """
    
    def __init__(self, 
                 event_bus: Optional[EventBus] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        self.event_bus = event_bus
        self.metrics = metrics_collector
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        
        # Configuration
        self.enabled = getattr(self.settings, 'COMPRESSION_ENABLED', True)
        self.min_size = getattr(self.settings, 'COMPRESSION_MIN_SIZE', 1024)
        self.compression_level = getattr(self.settings, 'COMPRESSION_LEVEL', 'balanced')
        self.adaptive_compression = getattr(self.settings, 'ADAPTIVE_COMPRESSION', True)
        
        # Initialize compression strategies
        self._init_strategies()
        
        # Performance tracking
        self.stats_buffer: List[CompressionStats] = []
        self.max_stats_buffer = 1000
        
    def _init_strategies(self):
        """Initialize compression strategies based on configuration."""
        level_map = {
            'fastest': CompressionLevel.FASTEST,
            'balanced': CompressionLevel.BALANCED,
            'best': CompressionLevel.BEST
        }
        level = level_map.get(self.compression_level, CompressionLevel.BALANCED)
        
        self.strategies = {
            CompressionType.GZIP: GzipStrategy(level),
            CompressionType.DEFLATE: DeflateStrategy(level),
            CompressionType.BROTLI: BrotliStrategy(level)
        }
        
        if self.adaptive_compression:
            self.adaptive_strategy = AdaptiveCompressionStrategy(
                list(self.strategies.values())
            )
    
    def parse_accept_encoding(self, accept_encoding: str) -> List[CompressionType]:
        """Parse Accept-Encoding header and return supported compression types."""
        if not accept_encoding:
            return []
        
        supported = []
        encodings = accept_encoding.lower().split(',')
        
        for encoding in encodings:
            encoding = encoding.strip().split(';')[0]  # Remove quality values
            
            if encoding == 'gzip':
                supported.append(CompressionType.GZIP)
            elif encoding == 'deflate':
                supported.append(CompressionType.DEFLATE)
            elif encoding == 'br':
                supported.append(CompressionType.BROTLI)
        
        return supported
    
    def should_compress_response(self, 
                                content: bytes, 
                                content_type: str, 
                                headers: Dict[str, str]) -> bool:
        """Determine if the response should be compressed."""
        if not self.enabled:
            return False
        
        # Check if already compressed
        if headers.get('Content-Encoding'):
            return False
        
        # Check minimum size
        if len(content) < self.min_size:
            return False
        
        # Check content type
        compressible_types = {
            'text/', 'application/json', 'application/xml',
            'application/javascript', 'application/x-javascript',
            'text/css', 'text/html', 'text/plain', 'text/xml'
        }
        
        if not any(content_type.startswith(ct) for ct in compressible_types):
            return False
        
        return True
    
    async def compress_response(self, 
                              content: bytes, 
                              content_type: str,
                              accept_encoding: str) -> tuple[bytes, str, CompressionStats]:
        """Compress response content and return compressed data with stats."""
        supported_encodings = self.parse_accept_encoding(accept_encoding)
        
        if not supported_encodings:
            raise ValueError("No supported compression encodings found")
        
        start_time = time.time()
        
        # Choose compression strategy
        if self.adaptive_compression:
            strategy = self.adaptive_strategy.choose_best_strategy(content, content_type)
            compression_type = None
            for ct, strat in self.strategies.items():
                if type(strat) == type(strategy):
                    compression_type = ct
                    break
        else:
            # Use first supported encoding
            compression_type = supported_encodings[0]
            strategy = self.strategies[compression_type]
        
        # Perform compression
        try:
            compressed_content = strategy.compress(content)
            compression_time = time.time() - start_time
            
            # Create stats
            stats = CompressionStats(
                original_size=len(content),
                compressed_size=len(compressed_content),
                compression_ratio=len(compressed_content) / len(content),
                compression_time=compression_time,
                algorithm=compression_type,
                level=strategy.level
            )
            
            encoding_header = strategy.get_encoding_header()
            
            return compressed_content, encoding_header, stats
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            raise
    
    def _record_stats(self, stats: CompressionStats):
        """Record compression statistics."""
        # Add to buffer
        self.stats_buffer.append(stats)
        if len(self.stats_buffer) > self.max_stats_buffer:
            self.stats_buffer.pop(0)
        
        # Send metrics
        if self.metrics:
            self.metrics.record_compression(
                algorithm=stats.algorithm.value,
                original_size=stats.original_size,
                compressed_size=stats.compressed_size,
                compression_ratio=stats.compression_ratio,
                compression_time=stats.compression_time
            )
        
        # Emit event
        if self.event_bus:
            event = CompressionEvent(
                algorithm=stats.algorithm.value,
                original_size=stats.original_size,
                compressed_size=stats.compressed_size,
                compression_ratio=stats.compression_ratio,
                compression_time=stats.compression_time
            )
            asyncio.create_task(self.event_bus.emit('compression.completed', event))
    
    async def process_response(self, 
                             content: Union[str, bytes, dict], 
                             content_type: str,
                             headers: Dict[str, str],
                             accept_encoding: str = '') -> tuple[bytes, Dict[str, str]]:
        """
        Process response for compression.
        
        Returns:
            Tuple of (processed_content, updated_headers)
        """
        # Convert content to bytes if needed
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        elif isinstance(content, dict):
            content_bytes = json.dumps(content).encode('utf-8')
        else:
            content_bytes = content
        
        # Check if compression should be applied
        if not self.should_compress_response(content_bytes, content_type, headers):
            return content_bytes, headers
        
        try:
            # Compress the content
            compressed_content, encoding_header, stats = await self.compress_response(
                content_bytes, content_type, accept_encoding
            )
            
            # Update headers
            updated_headers = headers.copy()
            updated_headers['Content-Encoding'] = encoding_header
            updated_headers['Content-Length'] = str(len(compressed_content))
            updated_headers['Vary'] = 'Accept-Encoding'
            
            # Record statistics
            self._record_stats(stats)
            
            self.logger.debug(
                f"Compressed response: {stats.original_size} -> {stats.compressed_size} bytes "
                f"({stats.compression_ratio:.2%}) using {stats.algorithm.value}"
            )
            
            return compressed_content, updated_headers
            
        except Exception as e:
            self.logger.error(f"Compression processing failed: {e}")
            # Return original content on failure
            return content_bytes, headers
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics summary."""
        if not self.stats_buffer:
            return {}
        
        total_original = sum(s.original_size for s in self.stats_buffer)
        total_compressed = sum(s.compressed_size for s in self.stats_buffer)
        avg_compression_time = sum(s.compression_time for s in self.stats_buffer) / len(self.stats_buffer)
        
        algorithm_counts = {}
        for stat in self.stats_buffer:
            algo = stat.algorithm.value
            algorithm_counts[algo] = algorithm_counts.get(algo, 0) + 1
        
        return {
            'total_requests': len(self.stats_buffer),
            'total_bytes_saved': total_original - total_compressed,
            'average_compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'average_compression_time': avg_compression_time,
            'algorithm_usage': algorithm_counts,
            'bytes_processed': {
                'original': total_original,
                'compressed': total_compressed
            }
        }


# Decorator for easy integration with route handlers
def compress_response(middleware: Optional[CompressionMiddleware] = None):
    """Decorator to apply compression to route handlers."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get the middleware instance
            compression_middleware = middleware or CompressionMiddleware()
            
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Extract response data
            if isinstance(result, tuple):
                content, status_code, headers = result
            else:
                content = result
                status_code = 200
                headers = {}
            
            # Get Accept-Encoding from request (this would need to be passed or extracted)
            accept_encoding = kwargs.get('accept_encoding', '')
            content_type = headers.get('Content-Type', 'application/json')
            
            # Process compression
            compressed_content, updated_headers = await compression_middleware.process_response(
                content, content_type, headers, accept_encoding
            )
            
            return compressed_content, status_code, updated_headers
        
        return wrapper
    return decorator


# Factory function for easy integration
def create_compression_middleware(event_bus: Optional[EventBus] = None,
                                metrics_collector: Optional[MetricsCollector] = None) -> CompressionMiddleware:
    """Factory function to create compression middleware with dependencies."""
    return CompressionMiddleware(event_bus=event_bus, metrics_collector=metrics_collector)


# Configuration validation
def validate_compression_config(config: Dict[str, Any]) -> bool:
    """Validate compression configuration."""
    required_keys = ['COMPRESSION_ENABLED', 'COMPRESSION_MIN_SIZE']
    
    for key in required_keys:
        if key not in config:
            return False
    
    if config.get('COMPRESSION_LEVEL') not in ['fastest', 'balanced', 'best']:
        return False
    
    return True


# Health check for compression middleware
async def compression_health_check(middleware: CompressionMiddleware) -> Dict[str, Any]:
    """Health check for compression middleware."""
    try:
        # Test compression with sample data
        test_data = b"Hello, World!" * 100
        compressed, encoding, stats = await middleware.compress_response(
            test_data, 'text/plain', 'gzip'
        )
        
        return {
            'status': 'healthy',
            'algorithms_available': list(middleware.strategies.keys()),
            'test_compression_ratio': stats.compression_ratio,
            'adaptive_compression': middleware.adaptive_compression
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }
