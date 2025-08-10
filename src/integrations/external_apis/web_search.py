"""
Advanced Web Search Integration for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 01:50:02 UTC

This module provides comprehensive web search capabilities with multi-provider support,
intelligent result processing, and seamless integration with the core AI assistant system.
"""

import hashlib
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, TypeVar, Union
from urllib.parse import quote, urlencode

import aiohttp
import asyncio

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Assistant components
from src.assistant.core import (
    ComponentInterface,
    ComponentMetadata,
    ComponentPriority,
    EnhancedComponentManager,
)
from src.assistant.core import EnhancedSessionManager
from src.assistant.core import StepExecutor, WorkflowOrchestrator, WorkflowStep

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged,
    ComponentInitialized,
    ProcessingCompleted,
    ProcessingError,
    ProcessingStarted,
)
from src.core.health_check import HealthCheck

# Integration components
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager
from src.learning.feedback_processor import FeedbackProcessor

# Memory and learning
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class SearchProvider(Enum):
    """Supported search providers."""

    GOOGLE = "google"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    YANDEX = "yandex"
    BAIDU = "baidu"


class SearchType(Enum):
    """Types of searches supported."""

    WEB = "web"
    IMAGES = "images"
    VIDEOS = "videos"
    NEWS = "news"
    ACADEMIC = "academic"
    SHOPPING = "shopping"
    MAPS = "maps"


class ResultType(Enum):
    """Types of search results."""

    WEBPAGE = "webpage"
    IMAGE = "image"
    VIDEO = "video"
    NEWS_ARTICLE = "news_article"
    ACADEMIC_PAPER = "academic_paper"
    PRODUCT = "product"
    MAP_LOCATION = "map_location"
    SNIPPET = "snippet"


class SearchQuality(Enum):
    """Search quality levels."""

    FAST = "fast"  # Quick results, basic processing
    BALANCED = "balanced"  # Good balance of speed and quality
    COMPREHENSIVE = "comprehensive"  # Detailed processing, slower


@dataclass
class SearchQuery:
    """Structured search query with metadata."""

    query: str
    search_type: SearchType = SearchType.WEB
    provider: Optional[SearchProvider] = None
    max_results: int = 10
    language: str = "en"
    region: str = "us"
    safe_search: bool = True
    time_range: Optional[str] = None  # hour, day, week, month, year
    site_filter: Optional[str] = None  # site:example.com
    file_type: Optional[str] = None  # filetype:pdf
    quality: SearchQuality = SearchQuality.BALANCED

    # Context and personalization
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Processing hints
    extract_content: bool = True
    summarize_results: bool = False
    filter_duplicates: bool = True
    rank_by_relevance: bool = True

    # Metadata
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchResult:
    """Individual search result with rich metadata."""

    title: str
    url: str
    snippet: str
    result_type: ResultType = ResultType.WEBPAGE

    # Content and metadata
    content: Optional[str] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)

    # Media specific
    image_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    video_duration: Optional[int] = None

    # Quality and relevance
    relevance_score: float = 0.0
    quality_score: float = 0.0
    authority_score: float = 0.0
    freshness_score: float = 0.0

    # Source information
    domain: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Technical metadata
    content_type: Optional[str] = None
    content_length: Optional[int] = None
    language: Optional[str] = None
    encoding: Optional[str] = None

    # Metadata
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    indexed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cached: bool = False
    provider: Optional[SearchProvider] = None


@dataclass
class SearchResponse:
    """Complete search response with analytics."""

    query: SearchQuery
    results: List[SearchResult] = field(default_factory=list)

    # Response metadata
    total_results: int = 0
    results_returned: int = 0
    search_time: float = 0.0
    providers_used: List[SearchProvider] = field(default_factory=list)

    # Quality metrics
    average_relevance: float = 0.0
    result_diversity: float = 0.0
    coverage_score: float = 0.0

    # Processing information
    cached_results: int = 0
    processed_results: int = 0
    filtered_results: int = 0

    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    partial_results: bool = False

    # Metadata
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl: int = 3600  # Cache TTL in seconds


class SearchError(Exception):
    """Custom exception for search operations."""

    def __init__(
        self,
        message: str,
        provider: Optional[SearchProvider] = None,
        query: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.query = query
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class SearchProviderInterface(ABC):
    """Abstract interface for search providers."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform search and return results."""
        pass

    @abstractmethod
    def get_provider_name(self) -> SearchProvider:
        """Get the provider identifier."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health status."""
        pass

    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        pass


class GoogleSearchProvider(SearchProviderInterface):
    """Google Custom Search API provider."""

    def __init__(self, config: Dict[str, Any], session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.api_key = config.get("api_key")
        self.cx = config.get("cx")  # Custom Search Engine ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.logger = get_logger(f"{__name__}.GoogleSearchProvider")

        # Rate limiting
        self.requests_made = defaultdict(int)
        self.last_reset = datetime.now(timezone.utc)
        self.rate_limit = config.get("rate_limit", 100)  # per day

        if not self.api_key or not self.cx:
            raise SearchError("Google Search requires api_key and cx configuration")

    def get_provider_name(self) -> SearchProvider:
        return SearchProvider.GOOGLE

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform Google search."""
        try:
            # Check rate limits
            await self._check_rate_limit()

            # Build search parameters
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": self._optimize_query(query.query),
                "num": min(query.max_results, 10),  # Google API limit
                "lr": f"lang_{query.language}",
                "gl": query.region,
                "safe": "high" if query.safe_search else "off",
            }

            # Add search type specific parameters
            if query.search_type == SearchType.IMAGES:
                params["searchType"] = "image"
            elif query.time_range:
                params["dateRestrict"] = query.time_range

            if query.site_filter:
                params["q"] += f" {query.site_filter}"

            if query.file_type:
                params["q"] += f" filetype:{query.file_type}"

            # Make API request
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_google_results(data, query)
                elif response.status == 429:
                    raise SearchError("Google Search rate limit exceeded", SearchProvider.GOOGLE)
                else:
                    raise SearchError(
                        f"Google Search API error: {response.status}", SearchProvider.GOOGLE
                    )

        except aiohttp.ClientError as e:
            raise SearchError(f"Google Search request failed: {str(e)}", SearchProvider.GOOGLE)

    async def _check_rate_limit(self) -> None:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)

        # Reset counter daily
        if (now - self.last_reset).days >= 1:
            self.requests_made.clear()
            self.last_reset = now

        today = now.date()
        if self.requests_made[today] >= self.rate_limit:
            raise SearchError("Daily rate limit exceeded for Google Search", SearchProvider.GOOGLE)

        self.requests_made[today] += 1

    def _optimize_query(self, query: str) -> str:
        """Optimize query for Google Search."""
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", query.strip())

        # Escape quotes
        query = query.replace('"', '\\"')

        return query

    async def _parse_google_results(
        self, data: Dict[str, Any], query: SearchQuery
    ) -> List[SearchResult]:
        """Parse Google API response into SearchResult objects."""
        results = []

        for item in data.get("items", []):
            try:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    result_type=(
                        ResultType.IMAGE
                        if query.search_type == SearchType.IMAGES
                        else ResultType.WEBPAGE
                    ),
                    provider=SearchProvider.GOOGLE,
                )

                # Add image-specific data
                if query.search_type == SearchType.IMAGES and "image" in item:
                    result.image_url = item["image"].get("contextLink")
                    result.thumbnail_url = item["image"].get("thumbnailLink")

                # Extract domain
                if result.url:
                    from urllib.parse import urlparse

                    result.domain = urlparse(result.url).netloc

                # Calculate initial relevance score
                result.relevance_score = self._calculate_relevance(result, query)

                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to parse Google result: {str(e)}")
                continue

        return results

    def _calculate_relevance(self, result: SearchResult, query: SearchQuery) -> float:
        """Calculate relevance score for a result."""
        score = 0.0
        query_lower = query.query.lower()

        # Title relevance (40%)
        title_lower = result.title.lower()
        title_score = sum(1 for word in query_lower.split() if word in title_lower)
        score += (title_score / len(query_lower.split())) * 0.4

        # Snippet relevance (30%)
        snippet_lower = result.snippet.lower()
        snippet_score = sum(1 for word in query_lower.split() if word in snippet_lower)
        score += (snippet_score / len(query_lower.split())) * 0.3

        # Domain authority (20%)
        known_domains = {"wikipedia.org": 0.9, "gov": 0.85, "edu": 0.8}
        for domain, auth_score in known_domains.items():
            if domain in (result.domain or ""):
                score += auth_score * 0.2
                break
        else:
            score += 0.1  # Default domain score

        # URL relevance (10%)
        url_lower = result.url.lower()
        url_score = sum(1 for word in query_lower.split() if word in url_lower)
        score += (url_score / len(query_lower.split())) * 0.1

        return min(score, 1.0)

    async def health_check(self) -> Dict[str, Any]:
        """Check Google Search API health."""
        try:
            # Make a simple test query
            params = {"key": self.api_key, "cx": self.cx, "q": "test", "num": 1}

            async with self.session.get(self.base_url, params=params) as response:
                return {
                    "status": "healthy" if response.status == 200 else "unhealthy",
                    "response_time": response.headers.get("x-response-time-ms", "unknown"),
                    "quota_used": self.requests_made.get(datetime.now(timezone.utc).date(), 0),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        today = datetime.now(timezone.utc).date()
        used = self.requests_made.get(today, 0)

        return {
            "provider": "google",
            "limit": self.rate_limit,
            "used": used,
            "remaining": max(0, self.rate_limit - used),
            "reset_time": (self.last_reset + timedelta(days=1)).isoformat(),
        }


class BingSearchProvider(SearchProviderInterface):
    """Bing Web Search API provider."""

    def __init__(self, config: Dict[str, Any], session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.api_key = config.get("api_key")
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        self.logger = get_logger(f"{__name__}.BingSearchProvider")

        # Rate limiting (Bing has different limits)
        self.requests_made = defaultdict(int)
        self.last_reset = datetime.now(timezone.utc)
        self.rate_limit = config.get("rate_limit", 3000)  # per month

        if not self.api_key:
            raise SearchError("Bing Search requires api_key configuration")

    def get_provider_name(self) -> SearchProvider:
        return SearchProvider.BING

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform Bing search."""
        try:
            # Check rate limits
            await self._check_rate_limit()

            # Build headers
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key,
                "User-Agent": "AI-Assistant-WebSearch/1.0",
            }

            # Build search parameters
            params = {
                "q": query.query,
                "count": min(query.max_results, 50),  # Bing API limit
                "mkt": f"{query.language}-{query.region.upper()}",
                "safeSearch": "Strict" if query.safe_search else "Off",
                "responseFilter": "Webpages",
            }

            if query.time_range:
                params["freshness"] = self._convert_time_range(query.time_range)

            # Make API request
            async with self.session.get(self.base_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._parse_bing_results(data, query)
                elif response.status == 429:
                    raise SearchError("Bing Search rate limit exceeded", SearchProvider.BING)
                else:
                    raise SearchError(
                        f"Bing Search API error: {response.status}", SearchProvider.BING
                    )

        except aiohttp.ClientError as e:
            raise SearchError(f"Bing Search request failed: {str(e)}", SearchProvider.BING)

    async def _check_rate_limit(self) -> None:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)

        # Reset counter monthly
        if (now - self.last_reset).days >= 30:
            self.requests_made.clear()
            self.last_reset = now

        month = now.replace(day=1)
        if self.requests_made[month] >= self.rate_limit:
            raise SearchError("Monthly rate limit exceeded for Bing Search", SearchProvider.BING)

        self.requests_made[month] += 1

    def _convert_time_range(self, time_range: str) -> str:
        """Convert time range to Bing format."""
        mapping = {
            "hour": "Day",  # Bing doesn't support hour, use day
            "day": "Day",
            "week": "Week",
            "month": "Month",
            "year": "Year",
        }
        return mapping.get(time_range, "Day")

    async def _parse_bing_results(
        self, data: Dict[str, Any], query: SearchQuery
    ) -> List[SearchResult]:
        """Parse Bing API response into SearchResult objects."""
        results = []

        webpages = data.get("webPages", {})
        for item in webpages.get("value", []):
            try:
                result = SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    result_type=ResultType.WEBPAGE,
                    provider=SearchProvider.BING,
                )

                # Parse date if available
                if "dateLastCrawled" in item:
                    try:
                        result.last_updated = datetime.fromisoformat(
                            item["dateLastCrawled"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                # Extract domain
                if result.url:
                    from urllib.parse import urlparse

                    result.domain = urlparse(result.url).netloc

                # Calculate relevance score
                result.relevance_score = self._calculate_relevance(result, query)

                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to parse Bing result: {str(e)}")
                continue

        return results

    def _calculate_relevance(self, result: SearchResult, query: SearchQuery) -> float:
        """Calculate relevance score for a result."""
        # Similar to Google implementation but adapted for Bing
        score = 0.0
        query_lower = query.query.lower()

        # Title relevance
        title_score = sum(1 for word in query_lower.split() if word in result.title.lower())
        score += (title_score / len(query_lower.split())) * 0.5

        # Snippet relevance
        snippet_score = sum(1 for word in query_lower.split() if word in result.snippet.lower())
        score += (snippet_score / len(query_lower.split())) * 0.3

        # Domain authority and freshness
        score += 0.2  # Base score

        return min(score, 1.0)

    async def health_check(self) -> Dict[str, Any]:
        """Check Bing Search API health."""
        try:
            headers = {"Ocp-Apim-Subscription-Key": self.api_key}
            params = {"q": "test", "count": 1}

            async with self.session.get(self.base_url, headers=headers, params=params) as response:
                return {
                    "status": "healthy" if response.status == 200 else "unhealthy",
                    "response_time": response.headers.get("bingapis-traceid", "unknown"),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        month = datetime.now(timezone.utc).replace(day=1)
        used = self.requests_made.get(month, 0)

        return {
            "provider": "bing",
            "limit": self.rate_limit,
            "used": used,
            "remaining": max(0, self.rate_limit - used),
            "reset_time": (month + timedelta(days=32)).replace(day=1).isoformat(),
        }


class DuckDuckGoSearchProvider(SearchProviderInterface):
    """DuckDuckGo search provider (no API key required)."""

    def __init__(self, config: Dict[str, Any], session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.base_url = "https://html.duckduckgo.com/html"
        self.logger = get_logger(f"{__name__}.DuckDuckGoSearchProvider")

        # Rate limiting (conservative for courtesy)
        self.last_request = 0
        self.min_interval = config.get("min_interval", 1.0)  # seconds between requests

    def get_provider_name(self) -> SearchProvider:
        return SearchProvider.DUCKDUCKGO

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform DuckDuckGo search."""
        try:
            # Rate limiting
            now = time.time()
            time_since_last = now - self.last_request
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)

            self.last_request = time.time()

            # Build search parameters
            params = {
                "q": query.query,
                "l": query.language,
                "safe": "strict" if query.safe_search else "off",
                "df": self._convert_time_range(query.time_range) if query.time_range else "",
            }

            headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Assistant-WebSearch/1.0)"}

            # Make request
            async with self.session.get(self.base_url, params=params, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    return await self._parse_duckduckgo_html(html, query)
                else:
                    raise SearchError(
                        f"DuckDuckGo search failed: {response.status}", SearchProvider.DUCKDUCKGO
                    )

        except aiohttp.ClientError as e:
            raise SearchError(f"DuckDuckGo request failed: {str(e)}", SearchProvider.DUCKDUCKGO)

    def _convert_time_range(self, time_range: str) -> str:
        """Convert time range to DuckDuckGo format."""
        mapping = {"day": "d", "week": "w", "month": "m", "year": "y"}
        return mapping.get(time_range, "")

    async def _parse_duckduckgo_html(self, html: str, query: SearchQuery) -> List[SearchResult]:
        """Parse DuckDuckGo HTML response."""
        if not HAS_BS4:
            raise SearchError("BeautifulSoup4 required for DuckDuckGo parsing")

        results = []
        soup = BeautifulSoup(html, "lxml")

        # Find result elements
        result_elements = soup.find_all("div", class_="result")

        for element in result_elements[: query.max_results]:
            try:
                # Extract title and URL
                title_link = element.find("a", class_="result__a")
                if not title_link:
                    continue

                title = title_link.get_text(strip=True)
                url = title_link.get("href", "")

                # Extract snippet
                snippet_elem = element.find("a", class_="result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                result = SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    result_type=ResultType.WEBPAGE,
                    provider=SearchProvider.DUCKDUCKGO,
                )

                # Extract domain
                if result.url:
                    from urllib.parse import urlparse

                    result.domain = urlparse(result.url).netloc

                # Calculate relevance
                result.relevance_score = self._calculate_relevance(result, query)

                results.append(result)

            except Exception as e:
                self.logger.warning(f"Failed to parse DuckDuckGo result: {str(e)}")
                continue

        return results

    def _calculate_relevance(self, result: SearchResult, query: SearchQuery) -> float:
        """Calculate relevance score."""
        score = 0.0
        query_lower = query.query.lower()

        # Basic relevance calculation
        title_score = sum(1 for word in query_lower.split() if word in result.title.lower())
        score += (title_score / len(query_lower.split())) * 0.6

        snippet_score = sum(1 for word in query_lower.split() if word in result.snippet.lower())
        score += (snippet_score / len(query_lower.split())) * 0.4

        return min(score, 1.0)

    async def health_check(self) -> Dict[str, Any]:
        """Check DuckDuckGo availability."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Assistant-WebSearch/1.0)"}
            async with self.session.get("https://duckduckgo.com", headers=headers) as response:
                return {
                    "status": "healthy" if response.status == 200 else "unhealthy",
                    "response_time": response.headers.get("x-response-time", "unknown"),
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit info."""
        return {
            "provider": "duckduckgo",
            "limit": "unlimited",
            "used": "unknown",
            "remaining": "unlimited",
            "min_interval": self.min_interval,
        }


class ContentExtractor:
    """Extracts and processes content from web pages."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.logger = get_logger(f"{__name__}.ContentExtractor")

    async def extract_content(self, url: str, max_length: int = 5000) -> Optional[str]:
        """Extract main content from a web page."""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Assistant-WebSearch/1.0)"}

            async with self.session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()

                    if HAS_BS4:
                        return self._extract_with_bs4(html, max_length)
                    else:
                        return self._extract_simple(html, max_length)

        except Exception as e:
            self.logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return None

    def _extract_with_bs4(self, html: str, max_length: int) -> str:
        """Extract content using BeautifulSoup."""
        soup = BeautifulSoup(html, "lxml")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Try to find main content areas
        content_selectors = ["article", "main", ".content", "#content", ".post", ".entry"]

        content_text = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_text = " ".join(elem.get_text(strip=True) for elem in elements)
                break

        # Fallback to body
        if not content_text:
            body = soup.find("body")
            if body:
                content_text = body.get_text(strip=True)

        # Clean up text
        content_text = re.sub(r"\s+", " ", content_text)

        return content_text[:max_length] if content_text else ""

    def _extract_simple(self, html: str, max_length: int) -> str:
        """Simple text extraction without BeautifulSoup."""
        # Remove scripts and styles
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", html)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_length]


class ResultRanker:
    """Ranks and scores search results."""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.ResultRanker")

    async def rank_results(
        self, results: List[SearchResult], query: SearchQuery
    ) -> List[SearchResult]:
        """Rank results by relevance and quality."""
        try:
            # Calculate composite scores
            for result in results:
                result.quality_score = self._calculate_quality_score(result)
                result.authority_score = self._calculate_authority_score(result)
                result.freshness_score = self._calculate_freshness_score(result)

                # Composite score
                composite_score = (
                    result.relevance_score * 0.4
                    + result.quality_score * 0.3
                    + result.authority_score * 0.2
                    + result.freshness_score * 0.1
                )
                result.relevance_score = composite_score

            # Sort by composite score
            results.sort(key=lambda r: r.relevance_score, reverse=True)

            return results

        except Exception as e:
            self.logger.error(f"Error ranking results: {str(e)}")
            return results

    def _calculate_quality_score(self, result: SearchResult) -> float:
        """Calculate content quality score."""
        score = 0.5  # Base score

        # Title quality
        if result.title and len(result.title) > 10:
            score += 0.2

        # Snippet quality
        if result.snippet and len(result.snippet) > 50:
            score += 0.2

        # URL quality (shorter, cleaner URLs are better)
        if result.url:
            if len(result.url) < 100:
                score += 0.1
            if not any(char in result.url for char in ["?", "&", "="]):
                score += 0.1

        return min(score, 1.0)

    def _calculate_authority_score(self, result: SearchResult) -> float:
        """Calculate domain authority score."""
        if not result.domain:
            return 0.3

        # Known high-authority domains
        authority_domains = {
            "wikipedia.org": 0.95,
            "gov": 0.9,
            "edu": 0.85,
            "stackoverflow.com": 0.8,
            "github.com": 0.8,
            "medium.com": 0.7,
            "reddit.com": 0.6,
        }

        domain_lower = result.domain.lower()

        # Check for exact matches
        for domain, score in authority_domains.items():
            if domain in domain_lower:
                return score

        # Check for TLD-based scoring
        if domain_lower.endswith(".gov"):
            return 0.9
        elif domain_lower.endswith(".edu"):
            return 0.85
        elif domain_lower.endswith(".org"):
            return 0.7
        elif domain_lower.endswith(".com"):
            return 0.5

        return 0.3

    def _calculate_freshness_score(self, result: SearchResult) -> float:
        """Calculate content freshness score."""
        if not result.last_updated and not result.publish_date:
            return 0.5  # Unknown freshness

        reference_date = result.last_updated or result.publish_date
        if not reference_date:
            return 0.5

        now = datetime.now(timezone.utc)
        age_days = (now - reference_date).days

        # Fresher content gets higher scores
        if age_days <= 1:
            return 1.0
        elif age_days <= 7:
            return 0.9
        elif age_days <= 30:
            return 0.7
        elif age_days <= 365:
            return 0.5
        else:
            return 0.2


class WebSearchStepExecutor(StepExecutor):
    """Workflow step executor for web search operations."""

    def __init__(self, web_search_manager: "EnhancedWebSearchManager"):
        self.web_search_manager = web_search_manager
        self.logger = get_logger(f"{__name__}.WebSearchStepExecutor")

    def can_execute(self, step: WorkflowStep) -> bool:
        """Check if this executor can handle web search steps."""
        return (
            step.component_name == "web_search"
            or step.skill_name == "web_search"
            or "search" in step.step_id.lower()
        )

    async def execute(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a web search workflow step."""
        try:
            # Extract search query from step parameters or context
            query_text = (
                step.parameters.get("query")
                or context.get("search_query")
                or context.get("user_input")
                or context.get("query")
            )

            if not query_text:
                raise WorkflowError("No search query provided for web search step")

            # Build search query
            search_query = SearchQuery(
                query=query_text,
                search_type=SearchType(step.parameters.get("search_type", "web")),
                max_results=step.parameters.get("max_results", 10),
                quality=SearchQuality(step.parameters.get("quality", "balanced")),
                user_id=context.get("user_id"),
                session_id=context.get("session_id"),
                context=context,
            )

            # Perform search
            response = await self.web_search_manager.search(search_query)

            # Return results for next workflow steps
            return {
                "search_results": [asdict(result) for result in response.results],
                "total_results": response.total_results,
                "search_time": response.search_time,
                "response_id": response.response_id,
            }

        except Exception as e:
            self.logger.error(f"Web search step execution failed: {str(e)}")
            raise WorkflowError(f"Web search step failed: {str(e)}", step_id=step.step_id)


class EnhancedWebSearchManager(ComponentInterface):
    """
    Advanced Web Search Manager for the AI Assistant.

    This manager provides comprehensive web search capabilities including:
    - Multi-provider search support (Google, Bing, DuckDuckGo, etc.)
    - Intelligent result processing and ranking
    - Caching and rate limiting
    - Integration with core AI assistant components
    - Real-time search streaming
    - Content extraction and summarization
    - Search analytics and learning
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced web search manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Optional integrations
        try:
            self.cache_strategy = container.get(CacheStrategy)
        except Exception:
            self.cache_strategy = None

        try:
            self.database = container.get(DatabaseManager)
        except Exception:
            self.database = None

        try:
            self.memory_manager = container.get(MemoryManager)
            self.context_manager = container.get(ContextManager)
        except Exception:
            self.memory_manager = None
            self.context_manager = None

        try:
            self.feedback_processor = container.get(FeedbackProcessor)
        except Exception:
            self.feedback_processor = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # HTTP session for requests
        self.session: Optional[aiohttp.ClientSession] = None

        # Search providers
        self.providers: Dict[SearchProvider, SearchProviderInterface] = {}
        self.provider_configs = self.config.get("web_search.providers", {})

        # Content processing
        self.content_extractor: Optional[ContentExtractor] = None
        self.result_ranker = ResultRanker()

        # Configuration
        self.default_provider = SearchProvider(
            self.config.get("web_search.default_provider", "duckduckgo")
        )
        self.max_results = self.config.get("web_search.max_results", 10)
        self.timeout_seconds = self.config.get("web_search.timeout_seconds", 30)
        self.enable_caching = self.config.get("web_search.enable_caching", True)
        self.cache_ttl = self.config.get("web_search.cache_ttl", 3600)

        # Performance tracking
        self.search_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.provider_performance: Dict[SearchProvider, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # State management
        self.active_searches: Dict[str, SearchQuery] = {}
        self.search_history: deque = deque(maxlen=1000)

        # Setup monitoring
        self._setup_monitoring()

        self.logger.info("EnhancedWebSearchManager initialized")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register web search metrics
            self.metrics.register_counter("web_search_requests_total")
            self.metrics.register_counter("web_search_requests_successful")
            self.metrics.register_counter("web_search_requests_failed")
            self.metrics.register_histogram("web_search_duration_seconds")
            self.metrics.register_histogram("web_search_results_count")
            self.metrics.register_counter("web_search_cache_hits")
            self.metrics.register_counter("web_search_cache_misses")
            self.metrics.register_gauge("web_search_active_requests")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the web search manager."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize content extractor
            self.content_extractor = ContentExtractor(self.session)

            # Initialize search providers
            await self._initialize_providers()

            # Register health check
            self.health_check.register_component("web_search_manager", self._health_check_callback)

            # Register as component
            component_manager = self.container.get(EnhancedComponentManager)
            component_manager.register_component(
                "web_search", type(self), ComponentPriority.NORMAL, []
            )

            # Register workflow step executor
            workflow_orchestrator = self.container.get(WorkflowOrchestrator)
            step_executor = WebSearchStepExecutor(self)
            workflow_orchestrator.step_executors.append(step_executor)

            self.logger.info("Web search manager initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize web search manager: {str(e)}")
            raise SearchError(f"Initialization failed: {str(e)}")

    async def _initialize_providers(self) -> None:
        """Initialize search providers based on configuration."""
        try:
            # Google Search
            if "google" in self.provider_configs:
                try:
                    provider = GoogleSearchProvider(self.provider_configs["google"], self.session)
                    self.providers[SearchProvider.GOOGLE] = provider
                    self.logger.info("Initialized Google Search provider")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Google Search: {str(e)}")

            # Bing Search
            if "bing" in self.provider_configs:
                try:
                    provider = BingSearchProvider(self.provider_configs["bing"], self.session)
                    self.providers[SearchProvider.BING] = provider
                    self.logger.info("Initialized Bing Search provider")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Bing Search: {str(e)}")

            # DuckDuckGo (no API key required)
            try:
                provider = DuckDuckGoSearchProvider(
                    self.provider_configs.get("duckduckgo", {}), self.session
                )
                self.providers[SearchProvider.DUCKDUCKGO] = provider
                self.logger.info("Initialized DuckDuckGo Search provider")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DuckDuckGo Search: {str(e)}")

            if not self.providers:
                raise SearchError("No search providers could be initialized")

        except Exception as e:
            self.logger.error(f"Provider initialization failed: {str(e)}")
            raise

    @handle_exceptions
    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Perform a comprehensive web search.

        Args:
            query: Search query with parameters

        Returns:
            Search response with results and metadata
        """
        start_time = time.time()
        self.active_searches[query.query_id] = query

        try:
            with self.tracer.trace("web_search") as span:
                span.set_attributes(
                    {
                        "query": query.query,
                        "search_type": query.search_type.value,
                        "provider": query.provider.value if query.provider else "auto",
                        "max_results": query.max_results,
                        "user_id": query.user_id or "anonymous",
                    }
                )

                # Emit search started event
                await self.event_bus.emit(
                    ProcessingStarted(
                        session_id=query.session_id or "",
                        request_id=query.query_id,
                        component="web_search",
                        input_data={"query": query.query},
                    )
                )

                # Check cache first
                cached_response = await self._check_cache(query)
                if cached_response:
                    self.metrics.increment("web_search_cache_hits")
                    return cached_response

                self.metrics.increment("web_search_cache_misses")

                # Select provider(s)
                providers_to_use = await self._select_providers(query)

                # Perform search with providers
                all_results = []
                providers_used = []
                errors = []

                for provider_enum in providers_to_use:
                    if provider_enum not in self.providers:
                        continue

                    provider = self.providers[provider_enum]

                    try:
                        provider_start = time.time()
                        results = await provider.search(query)
                        provider_time = time.time() - provider_start

                        # Track provider performance
                        self.provider_performance[provider_enum].append(provider_time)

                        all_results.extend(results)
                        providers_used.append(provider_enum)

                        self.logger.debug(
                            f"Provider {provider_enum.value} returned {len(results)} results "
                            f"in {provider_time:.2f}s"
                        )

                    except Exception as e:
                        error_msg = f"Provider {provider_enum.value} failed: {str(e)}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)
                        continue

                if not all_results and errors:
                    raise SearchError(f"All providers failed: {'; '.join(errors)}")

                # Process and rank results
                processed_results = await self._process_results(all_results, query)

                # Create response
                search_time = time.time() - start_time
                response = SearchResponse(
                    query=query,
                    results=processed_results[: query.max_results],
                    total_results=len(all_results),
                    results_returned=len(processed_results[: query.max_results]),
                    search_time=search_time,
                    providers_used=providers_used,
                    errors=errors,
                    ttl=self.cache_ttl,
                )

                # Calculate quality metrics
                response.average_relevance = (
                    sum(r.relevance_score for r in response.results) / len(response.results)
                    if response.results
                    else 0.0
                )

                # Cache the response
                await self._cache_response(query, response)

                # Store in memory for learning
                await self._store_search_memory(query, response)

                # Update metrics
                self.metrics.increment("web_search_requests_total")
                self.metrics.increment("web_search_requests_successful")
                self.metrics.record("web_search_duration_seconds", search_time)
                self.metrics.record("web_search_results_count", len(response.results))
                self.metrics.set("web_search_active_requests", len(self.active_searches))

                # Emit completion event
                await self.event_bus.emit(
                    ProcessingCompleted(
                        session_id=query.session_id or "",
                        request_id=query.query_id,
                        component="web_search",
                        output_data={
                            "results_count": len(response.results),
                            "search_time": search_time,
                            "providers_used": [p.value for p in providers_used],
                        },
                        processing_time=search_time,
                        success=True,
                    )
                )

                # Add to search history
                self.search_history.append(response)

                self.logger.info(
                    f"Search completed: '{query.query}' -> {len(response.results)} results "
                    f"in {search_time:.2f}s using {len(providers_used)} providers"
                )

                return response

        except Exception as e:
            # Handle search failure
            search_time = time.time() - start_time

            self.metrics.increment("web_search_requests_failed")

            await self.event_bus.emit(
                ProcessingError(
                    session_id=query.session_id or "",
                    request_id=query.query_id,
                    component="web_search",
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            )

            self.logger.error(f"Search failed for '{query.query}': {str(e)}")
            raise SearchError(f"Search failed: {str(e)}", query=query.query)

        finally:
            self.active_searches.pop(query.query_id, None)

    async def _select_providers(self, query: SearchQuery) -> List[SearchProvider]:
        """Select appropriate providers for the search query."""
        if query.provider and query.provider in self.providers:
            return [query.provider]

        # Auto-select based on search type and provider health
        available_providers = []

        for provider_enum, provider in self.providers.items():
            try:
                health = await provider.health_check()
                if health.get("status") == "healthy":
                    available_providers.append(provider_enum)
            except Exception:
                continue

        if not available_providers:
            # Fallback to default provider if available
            if self.default_provider in self.providers:
                return [self.default_provider]
            else:
                # Use any available provider
                return list(self.providers.keys())[:1]

        # For high-quality searches, use multiple providers
        if query.quality == SearchQuality.COMPREHENSIVE and len(available_providers) > 1:
            return available_providers[:2]  # Use top 2 providers

        # Prefer providers based on historical performance
        provider_scores = {}
        for provider in available_providers:
            performance_history = self.provider_performance.get(provider, [])
            if performance_history:
                avg_time = sum(performance_history) / len(performance_history)
                provider_scores[provider] = 1.0 / (avg_time + 0.1)  # Lower time = higher score
            else:
                provider_scores[provider] = 0.5  # Unknown performance

        # Sort by score and return best provider(s)
        sorted_providers = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
        return [provider for provider, score in sorted_providers[:1]]

    async def _check_cache(self, query: SearchQuery) -> Optional[SearchResponse]:
        """Check if search results are cached."""
        if not self.enable_caching or not self.cache_strategy:
            return None

        try:
            # Create cache key from query
            cache_key = self._create_cache_key(query)

            # Try to get cached response
            cached_data = await self.cache_strategy.get(cache_key)
            if cached_data:
                # Deserialize and return
                response_data = json.loads(cached_data)
                return self._deserialize_search_response(response_data)

        except Exception as e:
            self.logger.warning(f"Cache check failed: {str(e)}")

        return None

    async def _cache_response(self, query: SearchQuery, response: SearchResponse) -> None:
        """Cache search response."""
        if not self.enable_caching or not self.cache_strategy:
            return

        try:
            # Create cache key
            cache_key = self._create_cache_key(query)

            # Serialize response
            response_data = self._serialize_search_response(response)
            serialized = json.dumps(response_data, default=str)

            # Store in cache
            await self.cache_strategy.set(cache_key, serialized, ttl=response.ttl)

        except Exception as e:
            self.logger.warning(f"Failed to cache response: {str(e)}")

    def _create_cache_key(self, query: SearchQuery) -> str:
        """Create cache key for search query."""
        # Include relevant query parameters in cache key
        key_data = {
            "query": query.query.lower().strip(),
            "search_type": query.search_type.value,
            "language": query.language,
            "region": query.region,
            "safe_search": query.safe_search,
            "max_results": query.max_results,
        }

        # Create hash of key data
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"web_search:{key_hash}"

    async def _process_results(
        self, results: List[SearchResult], query: SearchQuery
    ) -> List[SearchResult]:
        """Process and enhance search results."""
        try:
            # Remove duplicates
            if query.filter_duplicates:
                results = self._remove_duplicates(results)

            # Extract content if requested and quality allows
            if query.extract_content and query.quality in [
                SearchQuality.BALANCED,
                SearchQuality.COMPREHENSIVE,
            ]:
                await self._extract_content_for_results(
                    results[:5]
                )  # Limit to top 5 for performance

            # Rank results
            if query.rank_by_relevance:
                results = await self.result_ranker.rank_results(results, query)

            return results

        except Exception as e:
            self.logger.warning(f"Result processing failed: {str(e)}")
            return results

    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        unique_results = []

        for result in results:
            # Check URL duplicates
            if result.url in seen_urls:
                continue

            # Check title similarity (simple approach)
            title_words = set(result.title.lower().split())
            is_similar = False

            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words.intersection(seen_words)) / max(len(title_words), 1) > 0.8:
                    is_similar = True
                    break

            if is_similar:
                continue

            seen_urls.add(result.url)
            seen_titles.add(result.title.lower())
            unique_results.append(result)

        return unique_results

    async def _extract_content_for_results(self, results: List[SearchResult]) -> None:
        """Extract content for search results."""
        if not self.content_extractor:
            return

        # Extract content concurrently (limited)
        semaphore = asyncio.Semaphore(3)  # Limit concurrent extractions

        async def extract_for_result(result: SearchResult):
            async with semaphore:
                content = await self.content_extractor.extract_content(result.url)
                if content:
                    result.content = content

                    # Update quality score based on content
                    if len(content) > 500:
                        result.quality_score = min(result.quality_score + 0.2, 1.0)

        tasks = [extract_for_result(result) for result in results]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _store_search_memory(self, query: SearchQuery, response: SearchResponse) -> None:
        """Store search in memory for learning and context."""
        if not self.memory_manager or not query.session_id:
            return

        try:
            # Create memory entry
            memory_data = {
                "type": "web_search",
                "query": query.query,
                "results_count": len(response.results),
                "search_time": response.search_time,
                "providers_used": [p.value for p in response.providers_used],
                "average_relevance": response.average_relevance,
                "user_id": query.user_id,
                "session_id": query.session_id,
                "timestamp": query.timestamp.isoformat(),
            }

            # Store in episodic memory
            await self.memory_manager.store_episodic_memory(
                event_type="web_search", data=memory_data, session_id=query.session_id
            )

        except Exception as e:
            self.logger.warning(f"Failed to store search memory: {str(e)}")

    def _serialize_search_response(self, response: SearchResponse) -> Dict[str, Any]:
        """Serialize search response for caching."""
        return {
            "query": asdict(response.query),
            "results": [asdict(result) for result in response.results],
            "total_results": response.total_results,
            "results_returned": response.results_returned,
            "search_time": response.search_time,
            "providers_used": [p.value for p in response.providers_used],
            "average_relevance": response.average_relevance,
            "result_diversity": response.result_diversity,
            "coverage_score": response.coverage_score,
            "cached_results": response.cached_results,
            "processed_results": response.processed_results,
            "filtered_results": response.filtered_results,
            "errors": response.errors,
            "warnings": response.warnings,
            "partial_results": response.partial_results,
            "response_id": response.response_id,
            "timestamp": response.timestamp.isoformat(),
            "ttl": response.ttl,
        }

    def _deserialize_search_response(self, data: Dict[str, Any]) -> SearchResponse:
        """Deserialize search response from cache."""
        # Reconstruct query
        query_data = data["query"]
        query_data["search_type"] = SearchType(query_data["search_type"])
        query_data["quality"] = SearchQuality(query_data["quality"])
        query_data["timestamp"] = datetime.fromisoformat(query_data["timestamp"])
        query = SearchQuery(**query_data)

        results = []
        for result_data in data["results"]:
            result_data["result_type"] = ResultType(result_data["result_type"])
            if result_data.get("provider"):
                result_data["provider"] = SearchProvider(result_data["provider"])

            results.append(SearchResult(**result_data))

        return SearchResponse(query=query, results=results)
