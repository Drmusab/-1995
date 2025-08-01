"""
Performance utilities for AI Assistant
Author: Drmusab
Last Modified: 2025-08-01

This module provides performance optimization utilities including efficient string
operations, list operations, and caching strategies.
"""

import re
import threading
import time
import weakref
from collections import defaultdict, deque
from functools import lru_cache, wraps
from io import StringIO
from typing import Any, Dict, Iterator, List, Optional, Union

import asyncio


class PerformantStringBuilder:
    """
    Efficient string builder to replace string concatenation in loops.
    Uses StringIO for better performance than repeated string concatenation.
    """

    def __init__(self, initial_capacity: int = 1024):
        self._buffer = StringIO()
        self._length = 0

    def append(self, text: str) -> "PerformantStringBuilder":
        """Append text to the builder."""
        self._buffer.write(text)
        self._length += len(text)
        return self

    def append_line(self, text: str = "") -> "PerformantStringBuilder":
        """Append text with a newline."""
        return self.append(text + "\n")

    def append_format(self, template: str, *args, **kwargs) -> "PerformantStringBuilder":
        """Append formatted text."""
        return self.append(template.format(*args, **kwargs))

    def clear(self) -> "PerformantStringBuilder":
        """Clear the buffer."""
        self._buffer = StringIO()
        self._length = 0
        return self

    def build(self) -> str:
        """Build the final string."""
        return self._buffer.getvalue()

    def __len__(self) -> int:
        return self._length

    def __str__(self) -> str:
        return self.build()


class EfficientListOperations:
    """
    Utilities for efficient list operations to replace inefficient patterns.
    """

    @staticmethod
    def batch_append(target_list: List[Any], items: List[Any], batch_size: int = 1000) -> None:
        """
        Efficiently append multiple items to a list in batches.
        More efficient than repeated append() calls.
        """
        if not items:
            return

        # Pre-extend the list to avoid multiple reallocations
        target_list.extend(items)

    @staticmethod
    def efficient_prepend(target_list: List[Any], items: List[Any]) -> List[Any]:
        """
        Efficiently prepend items to a list.
        Returns a new list instead of using insert(0, item) which is O(n).
        """
        return items + target_list

    @staticmethod
    def batch_insert_at_beginning(target_list: List[Any], items: List[Any]) -> List[Any]:
        """
        Efficiently insert multiple items at the beginning of a list.
        More efficient than multiple insert(0, item) calls.
        """
        return items + target_list

    @staticmethod
    def deduplicate_preserve_order(items: List[Any]) -> List[Any]:
        """
        Remove duplicates while preserving order.
        More efficient than converting to set for ordered deduplication.
        """
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


class CompiledRegexCache:
    """
    Cache for compiled regex patterns to avoid recompilation.
    """

    def __init__(self, max_size: int = 256):
        self._cache = {}
        self._max_size = max_size
        self._access_order = deque()

    def get_pattern(self, pattern: str, flags: int = 0) -> re.Pattern:
        """Get compiled regex pattern, compiling and caching if needed."""
        cache_key = (pattern, flags)

        if cache_key in self._cache:
            # Move to end of access order
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._cache[cache_key]

        # Compile new pattern
        compiled = re.compile(pattern, flags)

        # Add to cache, evicting oldest if needed
        if len(self._cache) >= self._max_size:
            oldest = self._access_order.popleft()
            del self._cache[oldest]

        self._cache[cache_key] = compiled
        self._access_order.append(cache_key)

        return compiled

    def search(self, pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
        """Search using cached compiled pattern."""
        compiled_pattern = self.get_pattern(pattern, flags)
        return compiled_pattern.search(text)

    def findall(self, pattern: str, text: str, flags: int = 0) -> List[str]:
        """Find all matches using cached compiled pattern."""
        compiled_pattern = self.get_pattern(pattern, flags)
        return compiled_pattern.findall(text)

    def sub(self, pattern: str, repl: str, text: str, flags: int = 0) -> str:
        """Substitute using cached compiled pattern."""
        compiled_pattern = self.get_pattern(pattern, flags)
        return compiled_pattern.sub(repl, text)


# Global regex cache instance
_regex_cache = CompiledRegexCache()


def efficient_dict_get(dictionary: Dict[Any, Any], key: Any, default: Any = None) -> Any:
    """
    More efficient dictionary access that avoids dict.keys() calls.
    Use 'key in dict' instead of 'key in dict.keys()'.
    """
    return dictionary.get(key, default)


def efficient_dict_membership(dictionary: Dict[Any, Any], key: Any) -> bool:
    """
    Efficient dictionary membership test.
    Use 'key in dict' instead of 'key in dict.keys()'.
    """
    return key in dictionary


class PerformanceTimer:
    """Context manager for timing operations (sync and async compatible)."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


class LRUCache:
    """
    Simple LRU cache implementation with thread safety.
    Alternative to functools.lru_cache for more control.
    """

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = deque()
        self.lock = threading.Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return default

    def put(self, key: Any, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.access_order.remove(key)
                self.access_order.append(key)
            else:
                # Add new
                if len(self.cache) >= self.maxsize:
                    # Evict least recently used
                    oldest = self.access_order.popleft()
                    del self.cache[oldest]

                self.cache[key] = value
                self.access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


def async_cache(maxsize: int = 128):
    """
    Async version of lru_cache decorator.
    """

    def decorator(func):
        cache = {}
        access_order = deque()
        lock = asyncio.Lock()

        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))

            async with lock:
                if key in cache:
                    # Move to end
                    access_order.remove(key)
                    access_order.append(key)
                    return cache[key]

                # Compute result
                result = await func(*args, **kwargs)

                # Add to cache
                if len(cache) >= maxsize:
                    oldest = access_order.popleft()
                    del cache[oldest]

                cache[key] = result
                access_order.append(key)

                return result

        return wrapper

    return decorator


class BatchProcessor:
    """
    Batch processor for handling operations in batches to improve performance.
    """

    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.batch = []
        self.last_process_time = time.time()

    def add_item(self, item: Any) -> Optional[List[Any]]:
        """
        Add item to batch. Returns batch if ready for processing.
        """
        self.batch.append(item)

        current_time = time.time()
        should_process = (
            len(self.batch) >= self.batch_size
            or (current_time - self.last_process_time) >= self.max_wait_time
        )

        if should_process:
            batch_to_process = self.batch[:]
            self.batch.clear()
            self.last_process_time = current_time
            return batch_to_process

        return None

    def flush(self) -> List[Any]:
        """Force processing of current batch."""
        batch_to_process = self.batch[:]
        self.batch.clear()
        self.last_process_time = time.time()
        return batch_to_process


# Utility functions for common performance patterns


def join_strings(strings: List[str], separator: str = "") -> str:
    """Efficient string joining - replacement for string concatenation in loops."""
    return separator.join(strings)


def format_string_list(template: str, items: List[Any]) -> List[str]:
    """Efficiently format a list of items using a template."""
    return [template.format(item) for item in items]


def chunk_list(items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """Split a list into chunks of specified size."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Efficiently flatten a nested list."""
    result = []
    for sublist in nested_list:
        result.extend(sublist)
    return result


# Export commonly used instances
regex_cache = _regex_cache
