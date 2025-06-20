"""
Advanced Weather API Integration for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 01:59:04 UTC

This module provides comprehensive weather data integration for the AI assistant,
supporting multiple weather providers, intelligent caching, real-time updates,
and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
import json
import aiohttp
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
from urllib.parse import urljoin, quote
import math

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComponentHealthChanged, ErrorOccurred, SystemStateChanged,
    ProcessingStarted, ProcessingCompleted, ExternalAPICallStarted,
    ExternalAPICallCompleted, DataCacheHit, DataCacheMiss, DataUpdated
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.sanitization import InputSanitizer

# Assistant components
from src.assistant.component_manager import ComponentManager, ComponentMetadata, ComponentPriority
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.session_manager import SessionManager
from src.assistant.interaction_handler import InteractionHandler

# Integration components
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.storage.database import DatabaseManager

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class WeatherProvider(Enum):
    """Supported weather data providers."""
    OPENWEATHERMAP = "openweathermap"
    WEATHERAPI = "weatherapi"
    ACCUWEATHER = "accuweather"
    DARKSKY = "darksky"
    NATIONAL_WEATHER_SERVICE = "nws"
    METEOSTAT = "meteostat"


class WeatherDataType(Enum):
    """Types of weather data."""
    CURRENT = "current"
    FORECAST_HOURLY = "forecast_hourly"
    FORECAST_DAILY = "forecast_daily"
    FORECAST_EXTENDED = "forecast_extended"
    HISTORICAL = "historical"
    ALERTS = "alerts"
    RADAR = "radar"
    SATELLITE = "satellite"
    AIR_QUALITY = "air_quality"
    UV_INDEX = "uv_index"


class WeatherUnits(Enum):
    """Weather measurement units."""
    METRIC = "metric"          # Celsius, km/h, hPa
    IMPERIAL = "imperial"      # Fahrenheit, mph, inHg
    KELVIN = "kelvin"         # Kelvin, m/s, hPa
    STANDARD = "standard"     # Kelvin, m/s, hPa


class AlertSeverity(Enum):
    """Weather alert severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


@dataclass
class WeatherLocation:
    """Represents a weather location."""
    latitude: float
    longitude: float
    name: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    timezone: Optional[str] = None
    elevation: Optional[float] = None
    
    def __post_init__(self):
        """Validate coordinates."""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")


@dataclass
class WeatherConditions:
    """Current weather conditions."""
    temperature: float
    feels_like: float
    humidity: float
    pressure: float
    visibility: Optional[float] = None
    
    # Wind data
    wind_speed: float = 0.0
    wind_direction: Optional[float] = None
    wind_gust: Optional[float] = None
    
    # Precipitation
    precipitation: float = 0.0
    precipitation_probability: float = 0.0
    
    # Sky conditions
    cloud_cover: float = 0.0
    condition: str = "clear"
    condition_code: Optional[str] = None
    
    # Additional data
    uv_index: Optional[float] = None
    dew_point: Optional[float] = None
    air_quality_index: Optional[int] = None
    
    # Metadata
    observation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    units: WeatherUnits = WeatherUnits.METRIC


@dataclass
class WeatherForecastPeriod:
    """Weather forecast for a specific time period."""
    start_time: datetime
    end_time: datetime
    conditions: WeatherConditions
    
    # Forecast-specific data
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    precipitation_total: float = 0.0
    precipitation_type: Optional[str] = None
    
    # Confidence metrics
    confidence: float = 1.0
    forecast_accuracy: Optional[float] = None


@dataclass
class WeatherAlert:
    """Weather alert or warning."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    
    # Time information
    start_time: datetime
    end_time: Optional[datetime] = None
    issued_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Geographic information
    areas: List[str] = field(default_factory=list)
    coordinates: Optional[List[WeatherLocation]] = None
    
    # Classification
    event_type: Optional[str] = None
    urgency: Optional[str] = None
    certainty: Optional[str] = None
    
    # Source information
    issuer: Optional[str] = None
    source_url: Optional[str] = None


@dataclass
class WeatherData:
    """Comprehensive weather data container."""
    location: WeatherLocation
    provider: WeatherProvider
    data_type: WeatherDataType
    
    # Weather information
    current_conditions: Optional[WeatherConditions] = None
    forecast_periods: List[WeatherForecastPeriod] = field(default_factory=list)
    alerts: List[WeatherAlert] = field(default_factory=list)
    
    # Quality metrics
    data_quality: float = 1.0
    freshness_score: float = 1.0
    completeness_score: float = 1.0
    
    # Metadata
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    cache_key: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class WeatherRequest:
    """Weather data request configuration."""
    location: WeatherLocation
    data_types: Set[WeatherDataType] = field(default_factory=lambda: {WeatherDataType.CURRENT})
    units: WeatherUnits = WeatherUnits.METRIC
    
    # Request parameters
    forecast_days: int = 5
    forecast_hours: int = 24
    include_alerts: bool = True
    include_air_quality: bool = False
    language: str = "en"
    
    # Quality requirements
    max_age_minutes: int = 60
    min_accuracy: float = 0.8
    preferred_providers: List[WeatherProvider] = field(default_factory=list)
    
    # Context information
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 1  # 1-5, higher is more urgent


class WeatherError(Exception):
    """Custom exception for weather API operations."""
    
    def __init__(self, message: str, provider: Optional[WeatherProvider] = None,
                 error_code: Optional[str] = None, location: Optional[WeatherLocation] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code
        self.location = location
        self.timestamp = datetime.now(timezone.utc)


class WeatherProviderBase(ABC):
    """Abstract base class for weather data providers."""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.config = config
        self.logger = get_logger(f"weather_provider_{self.__class__.__name__.lower()}")
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 1000)  # Requests per hour
        self.request_count = 0
        self.rate_window_start = datetime.now(timezone.utc)
        
        # Quality metrics
        self.success_rate = 1.0
        self.average_response_time = 0.0
        self.error_count = 0
    
    @abstractmethod
    async def get_current_weather(self, location: WeatherLocation, 
                                units: WeatherUnits = WeatherUnits.METRIC) -> WeatherConditions:
        """Get current weather conditions."""
        pass
    
    @abstractmethod
    async def get_forecast(self, location: WeatherLocation, 
                          data_type: WeatherDataType,
                          units: WeatherUnits = WeatherUnits.METRIC,
                          **kwargs) -> List[WeatherForecastPeriod]:
        """Get weather forecast."""
        pass
    
    @abstractmethod
    async def get_alerts(self, location: WeatherLocation) -> List[WeatherAlert]:
        """Get weather alerts."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AI-Assistant-Weather/1.0'}
        )
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit allows request."""
        current_time = datetime.now(timezone.utc)
        time_diff = (current_time - self.rate_window_start).total_seconds()
        
        # Reset window if an hour has passed
        if time_diff >= 3600:
            self.request_count = 0
            self.rate_window_start = current_time
        
        return self.request_count < self.rate_limit
    
    def record_request(self, success: bool, response_time: float) -> None:
        """Record request metrics."""
        self.request_count += 1
        
        if success:
            self.success_rate = (self.success_rate * 0.9) + (1.0 * 0.1)
        else:
            self.success_rate = (self.success_rate * 0.9) + (0.0 * 0.1)
            self.error_count += 1
        
        self.average_response_time = (self.average_response_time * 0.9) + (response_time * 0.1)


class OpenWeatherMapProvider(WeatherProviderBase):
    """OpenWeatherMap API provider."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    async def get_current_weather(self, location: WeatherLocation, 
                                units: WeatherUnits = WeatherUnits.METRIC) -> WeatherConditions:
        """Get current weather from OpenWeatherMap."""
        if not self.check_rate_limit():
            raise WeatherError("Rate limit exceeded", WeatherProvider.OPENWEATHERMAP)
        
        start_time = time.time()
        
        try:
            url = f"{self.BASE_URL}/weather"
            params = {
                'lat': location.latitude,
                'lon': location.longitude,
                'appid': self.api_key,
                'units': self._convert_units(units)
            }
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                conditions = self._parse_current_weather(data, units)
                
                response_time = time.time() - start_time
                self.record_request(True, response_time)
                
                return conditions
                
        except Exception as e:
            response_time = time.time() - start_time
            self.record_request(False, response_time)
            raise WeatherError(f"OpenWeatherMap API error: {str(e)}", 
                             WeatherProvider.OPENWEATHERMAP) from e
    
    async def get_forecast(self, location: WeatherLocation, 
                          data_type: WeatherDataType,
                          units: WeatherUnits = WeatherUnits.METRIC,
                          **kwargs) -> List[WeatherForecastPeriod]:
        """Get weather forecast from OpenWeatherMap."""
        if not self.check_rate_limit():
            raise WeatherError("Rate limit exceeded", WeatherProvider.OPENWEATHERMAP)
        
        start_time = time.time()
        
        try:
            if data_type == WeatherDataType.FORECAST_HOURLY:
                url = f"{self.BASE_URL}/forecast"
            else:
                url = f"{self.BASE_URL}/forecast/daily"
            
            params = {
                'lat': location.latitude,
                'lon': location.longitude,
                'appid': self.api_key,
                'units': self._convert_units(units)
            }
            
            # Add specific parameters
            if 'cnt' in kwargs:
                params['cnt'] = kwargs['cnt']
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                forecast = self._parse_forecast(data, data_type, units)
                
                response_time = time.time() - start_time
                self.record_request(True, response_time)
                
                return forecast
                
        except Exception as e:
            response_time = time.time() - start_time
            self.record_request(False, response_time)
            raise WeatherError(f"OpenWeatherMap forecast error: {str(e)}", 
                             WeatherProvider.OPENWEATHERMAP) from e
    
    async def get_alerts(self, location: WeatherLocation) -> List[WeatherAlert]:
        """Get weather alerts from OpenWeatherMap One Call API."""
        if not self.check_rate_limit():
            raise WeatherError("Rate limit exceeded", WeatherProvider.OPENWEATHERMAP)
        
        start_time = time.time()
        
        try:
            url = f"{self.BASE_URL.replace('data/2.5', 'data/3.0')}/onecall"
            params = {
                'lat': location.latitude,
                'lon': location.longitude,
                'appid': self.api_key,
                'exclude': 'minutely,hourly,daily,current'
            }
            
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                alerts = self._parse_alerts(data)
                
                response_time = time.time() - start_time
                self.record_request(True, response_time)
                
                return alerts
                
        except Exception as e:
            response_time = time.time() - start_time
            self.record_request(False, response_time)
            raise WeatherError(f"OpenWeatherMap alerts error: {str(e)}", 
                             WeatherProvider.OPENWEATHERMAP) from e
    
    def _convert_units(self, units: WeatherUnits) -> str:
        """Convert units enum to OpenWeatherMap format."""
        mapping = {
            WeatherUnits.METRIC: "metric",
            WeatherUnits.IMPERIAL: "imperial",
            WeatherUnits.KELVIN: "standard",
            WeatherUnits.STANDARD: "standard"
        }
        return mapping.get(units, "metric")
    
    def _parse_current_weather(self, data: Dict[str, Any], units: WeatherUnits) -> WeatherConditions:
        """Parse current weather data from OpenWeatherMap."""
        main = data.get('main', {})
        wind = data.get('wind', {})
        weather = data.get('weather', [{}])[0]
        
        return WeatherConditions(
            temperature=main.get('temp', 0.0),
            feels_like=main.get('feels_like', 0.0),
            humidity=main.get('humidity', 0.0),
            pressure=main.get('pressure', 0.0),
            visibility=data.get('visibility', 0.0) / 1000,  # Convert to km
            wind_speed=wind.get('speed', 0.0),
            wind_direction=wind.get('deg'),
            wind_gust=wind.get('gust'),
            cloud_cover=data.get('clouds', {}).get('all', 0.0),
            condition=weather.get('main', 'clear').lower(),
            condition_code=str(weather.get('id', '')),
            observation_time=datetime.fromtimestamp(data.get('dt', 0), timezone.utc),
            units=units
        )
    
    def _parse_forecast(self, data: Dict[str, Any], data_type: WeatherDataType, 
                       units: WeatherUnits) -> List[WeatherForecastPeriod]:
        """Parse forecast data from OpenWeatherMap."""
        forecast_list = data.get('list', [])
        periods = []
        
        for item in forecast_list:
            main = item.get('main', {})
            wind = item.get('wind', {})
            weather = item.get('weather', [{}])[0]
            
            start_time = datetime.fromtimestamp(item.get('dt', 0), timezone.utc)
            end_time = start_time + timedelta(hours=3)  # OWM 3-hour intervals
            
            conditions = WeatherConditions(
                temperature=main.get('temp', 0.0),
                feels_like=main.get('feels_like', 0.0),
                humidity=main.get('humidity', 0.0),
                pressure=main.get('pressure', 0.0),
                wind_speed=wind.get('speed', 0.0),
                wind_direction=wind.get('deg'),
                cloud_cover=item.get('clouds', {}).get('all', 0.0),
                condition=weather.get('main', 'clear').lower(),
                condition_code=str(weather.get('id', '')),
                precipitation_probability=item.get('pop', 0.0) * 100,
                observation_time=start_time,
                units=units
            )
            
            period = WeatherForecastPeriod(
                start_time=start_time,
                end_time=end_time,
                conditions=conditions,
                temperature_min=main.get('temp_min'),
                temperature_max=main.get('temp_max'),
                precipitation_total=item.get('rain', {}).get('3h', 0.0) + 
                                  item.get('snow', {}).get('3h', 0.0)
            )
            
            periods.append(period)
        
        return periods
    
    def _parse_alerts(self, data: Dict[str, Any]) -> List[WeatherAlert]:
        """Parse weather alerts from OpenWeatherMap."""
        alerts_data = data.get('alerts', [])
        alerts = []
        
        for alert_data in alerts_data:
            severity_map = {
                'minor': AlertSeverity.MINOR,
                'moderate': AlertSeverity.MODERATE,
                'severe': AlertSeverity.SEVERE,
                'extreme': AlertSeverity.EXTREME
            }
            
            alert = WeatherAlert(
                alert_id=str(hash(alert_data.get('event', '') + str(alert_data.get('start', 0)))),
                title=alert_data.get('event', 'Weather Alert'),
                description=alert_data.get('description', ''),
                severity=severity_map.get(alert_data.get('severity', 'unknown'), AlertSeverity.UNKNOWN),
                start_time=datetime.fromtimestamp(alert_data.get('start', 0), timezone.utc),
                end_time=datetime.fromtimestamp(alert_data.get('end', 0), timezone.utc) 
                         if alert_data.get('end') else None,
                event_type=alert_data.get('event'),
                issuer=alert_data.get('sender_name')
            )
            
            alerts.append(alert)
        
        return alerts


class WeatherCacheManager:
    """Intelligent weather data caching manager."""
    
    def __init__(self, redis_cache: Optional[RedisCache] = None, 
                 cache_strategy: Optional[CacheStrategy] = None):
        self.redis_cache = redis_cache
        self.cache_strategy = cache_strategy
        self.logger = get_logger(__name__)
        
        # Cache configuration
        self.default_ttl = {
            WeatherDataType.CURRENT: 600,  # 10 minutes
            WeatherDataType.FORECAST_HOURLY: 3600,  # 1 hour
            WeatherDataType.FORECAST_DAILY: 21600,  # 6 hours
            WeatherDataType.ALERTS: 300,  # 5 minutes
        }
        
        # Memory cache for hot data
        self.memory_cache: Dict[str, WeatherData] = {}
        self.cache_access_times: Dict[str, datetime] = {}
        self.max_memory_items = 1000
    
    def generate_cache_key(self, request: WeatherRequest, provider: WeatherProvider) -> str:
        """Generate a cache key for the weather request."""
        location_key = f"{request.location.latitude:.4f},{request.location.longitude:.4f}"
        data_types_key = ",".join(sorted([dt.value for dt in request.data_types]))
        
        key_components = [
            provider.value,
            location_key,
            data_types_key,
            request.units.value,
            str(request.forecast_days),
            str(request.forecast_hours)
        ]
        
        return hashlib.md5(":".join(key_components).encode()).hexdigest()
    
    async def get_cached_data(self, cache_key: str) -> Optional[WeatherData]:
        """Retrieve cached weather data."""
        try:
            # Check memory cache first
            if cache_key in self.memory_cache:
                weather_data = self.memory_cache[cache_key]
                
                # Check if data is still fresh
                if self._is_data_fresh(weather_data):
                    self.cache_access_times[cache_key] = datetime.now(timezone.utc)
                    return weather_data
                else:
                    # Remove stale data
                    del self.memory_cache[cache_key]
                    self.cache_access_times.pop(cache_key, None)
            
            # Check Redis cache
            if self.redis_cache:
                cached_data = await self.redis_cache.get(f"weather:{cache_key}")
                if cached_data:
                    weather_data = self._deserialize_weather_data(cached_data)
                    
                    if self._is_data_fresh(weather_data):
                        # Store in memory cache for faster access
                        self._store_in_memory_cache(cache_key, weather_data)
                        return weather_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache retrieval error: {str(e)}")
            return None
    
    async def store_cached_data(self, cache_key: str, weather_data: WeatherData) -> None:
        """Store weather data in cache."""
        try:
            # Determine TTL based on data type
            ttl = self.default_ttl.get(weather_data.data_type, 3600)
            
            # Set expiration time
            weather_data.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            weather_data.cache_key = cache_key
            
            # Store in memory cache
            self._store_in_memory_cache(cache_key, weather_data)
            
            # Store in Redis cache
            if self.redis_cache:
                serialized_data = self._serialize_weather_data(weather_data)
                await self.redis_cache.set(f"weather:{cache_key}", serialized_data, ttl)
            
        except Exception as e:
            self.logger.warning(f"Cache storage error: {str(e)}")
    
    def _store_in_memory_cache(self, cache_key: str, weather_data: WeatherData) -> None:
        """Store data in memory cache with LRU eviction."""
        # Evict old items if cache is full
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_old_items()
        
        self.memory_cache[cache_key] = weather_data
        self.cache_access_times[cache_key] = datetime.now(timezone.utc)
    
    def _evict_old_items(self) -> None:
        """Evict least recently used items from memory cache."""
        # Sort by access time and remove oldest 10%
        sorted_items = sorted(self.cache_access_times.items(), key=lambda x: x[1])
        items_to_remove = int(len(sorted_items) * 0.1)
        
        for cache_key, _ in sorted_items[:items_to_remove]:
            self.memory_cache.pop(cache_key, None)
            self.cache_access_times.pop(cache_key, None)
    
    def _is_data_fresh(self, weather_data: WeatherData) -> bool:
        """Check if weather data is still fresh."""
        if not weather_data.expires_at:
            return True
        
        return datetime.now(timezone.utc) < weather_data.expires_at
    
    def _serialize_weather_data(self, weather_data: WeatherData) -> str:
        """Serialize weather data for caching."""
        data_dict = asdict(weather_data)
        
        # Convert datetime objects to ISO strings
        for key, value in data_dict.items():
            if isinstance(value, datetime):
                data_dict[key] = value.isoformat()
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(v, datetime):
                                data_dict[key][i][k] = v.isoformat()
        
        return json.dumps(data_dict)
    
    def _deserialize_weather_data(self, cached_data: str) -> WeatherData:
        """Deserialize cached weather data."""
        data_dict = json.loads(cached_data)
        
        # Convert ISO strings back to datetime objects
        datetime_fields = ['retrieved_at', 'expires_at']
        for field in datetime_fields:
            if data_dict.get(field):
                data_dict[field] = datetime.fromisoformat(data_dict[field])
        
        # Handle nested datetime fields
        if data_dict.get('current_conditions'):
            obs_time = data_dict['current_conditions'].get('observation_time')
            if obs_time:
                data_dict['current_conditions']['observation_time'] = datetime.fromisoformat(obs_time)
        
        # Convert enums
        data_dict['provider'] = WeatherProvider(data_dict['provider'])
        data_dict['data_type'] = WeatherDataType(data_dict['data_type'])
        
        return WeatherData(**data_dict)


class EnhancedWeatherAPI:
    """
    Advanced Weather API Integration for the AI Assistant.
    
    This component provides comprehensive weather data integration with:
    - Multiple weather data providers with intelligent fallback
    - Advanced caching and data freshness management
    - Real-time weather updates and alerts
    - Location-based weather queries with geocoding
    - Integration with core assistant components
    - Performance monitoring and optimization
    - Weather-based insights and recommendations
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced weather API.
        
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
        self.input_sanitizer = container.get(InputSanitizer)
        
        # Assistant components
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.session_manager = container.get(SessionManager)
        self.interaction_handler = container.get(InteractionHandler)
        
        # Storage and caching
        try:
            self.redis_cache = container.get(RedisCache)
            self.cache_strategy = container.get(CacheStrategy)
            self.database = container.get(DatabaseManager)
        except Exception:
            self.redis_cache = None
            self.cache_strategy = None
            self.database = None
        
        # Memory systems
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Weather providers
        self.providers: Dict[WeatherProvider, WeatherProviderBase] = {}
        self.provider_priorities: List[WeatherProvider] = []
        
        # Cache management
        self.cache_manager = WeatherCacheManager(self.redis_cache, self.cache_strategy)
        
        # Request queue and processing
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.active_requests: Dict[str, WeatherRequest] = {}
        self.request_semaphore = asyncio.Semaphore(10)
        
        # Location cache for geocoding
        self.location_cache: Dict[str, WeatherLocation] = {}
        
        # Configuration
        self.default_units = WeatherUnits(self.config.get("weather.default_units", "metric"))
        self.cache_enabled = self.config.get("weather.cache_enabled", True)
        self.max_retries = self.config.get("weather.max_retries", 3)
        self.request_timeout = self.config.get("weather.request_timeout", 30.0)
        
        # Performance tracking
        self.request_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.provider_health: Dict[WeatherProvider, float] = {}
        
        # Setup monitoring and health checks
        self._setup_monitoring()
        self.health_check.register_component("weather_api", self._health_check_callback)
        
        self.logger.info("EnhancedWeatherAPI initialized successfully")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics collection."""
        try:
            # Register weather API metrics
            self.metrics.register_counter("weather_requests_total")
            self.metrics.register_counter("weather_requests_successful")
            self.metrics.register_counter("weather_requests_failed")
            self.metrics.register_histogram("weather_request_duration_seconds")
            self.metrics.register_counter("weather_cache_hits")
            self.metrics.register_counter("weather_cache_misses")
            self.metrics.register_gauge("weather_provider_health")
            self.metrics.register_counter("weather_api_errors_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the weather API system."""
        try:
            self.logger.info("Initializing weather API system...")
            
            # Initialize weather providers
            await self._initialize_providers()
            
            # Register component
            await self._register_as_component()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            asyncio.create_task(self._request_processor_loop())
            asyncio.create_task(self._health_monitor_loop())
            asyncio.create_task(self._cache_cleanup_loop())
            
            self.logger.info("Weather API system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize weather API: {str(e)}")
            raise WeatherError(f"Weather API initialization failed: {str(e)}")

    async def _initialize_providers(self) -> None:
        """Initialize weather data providers."""
        # OpenWeatherMap
        owm_config = self.config.get("weather.providers.openweathermap", {})
        if owm_config.get("enabled", False) and owm_config.get("api_key"):
            provider = OpenWeatherMapProvider(owm_config["api_key"], owm_config)
            await provider.initialize()
            self.providers[WeatherProvider.OPENWEATHERMAP] = provider
            self.provider_priorities.append(WeatherProvider.OPENWEATHERMAP)
            self.logger.info("Initialized OpenWeatherMap provider")
        
        # Add other providers here (WeatherAPI, AccuWeather, etc.)
        
        if not self.providers:
            raise WeatherError("No weather providers configured")
        
        self.logger.info(f"Initialized {len(self.providers)} weather providers")

    async def _register_as_component(self) -> None:
        """Register weather API as a system component."""
        try:
            metadata = ComponentMetadata(
                component_id="weather_api",
                component_type=type(self),
                priority=ComponentPriority.NORMAL,
                description="Advanced weather data integration service",
                config_section="weather",
                health_check_interval=60.0
            )
            
            self.component_manager.register_component(
                "weather_api",
                type(self),
                ComponentPriority.NORMAL,
                []
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to register as component: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)
        
        # Session events for location context
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

    @handle_exceptions
    async def get_weather(self, request: WeatherRequest) -> WeatherData:
        """
        Get weather data for a location.
        
        Args:
            request: Weather request configuration
            
        Returns:
            Comprehensive weather data
        """
        async with self.request_semaphore:
            start_time = time.time()
            
            try:
                with self.tracer.trace("weather_request") as span:
                    span.set_attributes({
                        "location_lat": request.location.latitude,
                        "location_lon": request.location.longitude,
                        "data_types": ",".join([dt.value for dt in request.data_types]),
                        "units": request.units.value,
                        "session_id": request.session_id or "anonymous"
                    })
                    
                    # Emit request started event
                    await self.event_bus.emit(ExternalAPICallStarted(
                        api_name="weather",
                        endpoint="get_weather",
                        request_id=request.request_id,
                        session_id=request.session_id
                    ))
                    
                    # Check cache first
                    weather_data = None
                    if self.cache_enabled:
                        weather_data = await self._get_cached_weather(request)
                        if weather_data:
                            await self.event_bus.emit(DataCacheHit(
                                cache_key=weather_data.cache_key,
                                data_type="weather",
                                session_id=request.session_id
                            ))
                    
                    # Fetch from providers if not cached
                    if not weather_data:
                        await self.event_bus.emit(DataCacheMiss(
                            cache_key=f"weather_{request.request_id}",
                            data_type="weather",
                            session_id=request.session_id
                        ))
                        
                        weather_data = await self._fetch_from_providers(request)
                        
                        # Cache the result
                        if self.cache_enabled and weather_data:
                            await self._cache_weather_data(request, weather_data)
                    
                    # Store in user memory if session provided
                    if request.session_id and weather_data:
                        await self._store_weather_memory(request, weather_data)
                    
                    # Update metrics
                    processing_time = time.time() - start_time
                    self.metrics.increment("weather_requests_total")
                    self.metrics.increment("weather_requests_successful")
                    self.metrics.record("weather_request_duration_seconds", processing_time)
                    
                    # Emit completion event
                    await self.event_bus.emit(ExternalAPICallCompleted(
                        api_name="weather",
                        endpoint="get_weather",
                        request_id=request.request_id,
                        session_id=request.session_id,
                        success=True,
                        response_time=processing_time
                    ))
                    
                    self.logger.info(
                        f"Weather request completed for {request.location.name or 'location'} "
                        f"in {processing_time:.2f}s"
                    )
                    
                    return weather_data
                    
            except Exception as e:
                processing_time = time.time() - start_time
                
                self.metrics.increment("weather_requests_failed")
                self.metrics.increment("weather_api_errors_total")
                
                await self.event_bus.emit(ExternalAPICallCompleted(
                    api_name="weather",
                    endpoint="get_weather",
                    request_id=request.request_id,
                    session_id=request.session_id,
                    success=False,
                    response_time=processing_time,
                    error_message=str(e)
                ))
                
                self.logger.error(f"Weather request failed: {str(e)}")
                raise

    async def _get_cached_weather(self, request: WeatherRequest) -> Optional[WeatherData]:
        """Get cached weather data if available and fresh."""
        try:
            # Try each provider's cache
            for provider in self.provider_priorities:
                cache_key = self.cache_manager.generate_cache_key(request, provider)
                cached_data = await self.cache_manager.get_cached_data(cache_key)
                
                if cached_data and self._is_cache_suitable(cached_data, request):
                    self.logger.debug(f"Cache hit for weather data: {cache_key}")
                    return cached_data
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache retrieval error: {str(e)}")
            return None

    def _is_cache_suitable(self, cached_data: WeatherData, request: WeatherRequest) -> bool:
        """Check if cached data meets request requirements."""
        # Check data age
        age_minutes = (datetime.now(timezone.utc) - cached_data.retrieved_at).total_seconds() / 60
        if age_minutes > request.max_age_minutes:
            return False
        
        # Check data types
        if not request.data_types.issubset({cached_data.data_type}):
            return False
        
        # Check location proximity (within ~1km)
        distance = self._calculate_distance(
            request.location.latitude, request.location.longitude,
            cached_data.location.latitude, cached_data.location.longitude
        )
        if distance > 1.0:  # 1 km
            return False
        
        # Check data quality
        if cached_data.data_quality < request.min_accuracy:
            return False
        
        return True

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers."""
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

    async def _fetch_from_providers(self, request: WeatherRequest) -> WeatherData:
        """Fetch weather data from providers with fallback logic."""
        last_error = None
        
        # Try preferred providers first
        providers_to_try = list(request.preferred_providers) + [
            p for p in self.provider_priorities 
            if p not in request.preferred_providers
        ]
        
        for provider_type in providers_to_try:
            if provider_type not in self.providers:
                continue
            
            provider = self.providers[provider_type]
            
            # Check provider health
            if self.provider_health.get(provider_type, 1.0) < 0.5:
                self.logger.warning(f"Skipping unhealthy provider: {provider_type.value}")
                continue
            
            try:
                weather_data = await self._fetch_from_single_provider(provider, provider_type, request)
                if weather_data:
                    self.logger.info(f"Successfully fetched weather data from {provider_type.value}")
                    return weather_data
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider {provider_type.value} failed: {str(e)}")
                
                # Update provider health
                current_health = self.provider_health.get(provider_type, 1.0)
                self.provider_health[provider_type] = max(0.0, current_health - 0.2)
                
                continue
        
        # If all providers failed
        if last_error:
            raise WeatherError(f"All weather providers failed. Last error: {str(last_error)}")
        else:
            raise WeatherError("No weather providers available")

    async def _fetch_from_single_provider(
        self,
        provider: WeatherProviderBase,
        provider_type: WeatherProvider,
        request: WeatherRequest
    ) -> WeatherData:
        """Fetch weather data from a single provider."""
        weather_data = WeatherData(
            location=request.location,
            provider=provider_type,
            data_type=list(request.data_types)[0]  # Primary data type
        )
        
        # Fetch current conditions
        if WeatherDataType.CURRENT in request.data_types:
            try:
                conditions = await provider.get_current_weather(request.location, request.units)
                weather_data.current_conditions = conditions
                weather_data.data_type = WeatherDataType.CURRENT
            except Exception as e:
                self.logger.warning(f"Failed to get current weather: {str(e)}")
        
        # Fetch forecasts
        forecast_types = [
            WeatherDataType.FORECAST_HOURLY,
            WeatherDataType.FORECAST_DAILY,
            WeatherDataType.FORECAST_EXTENDED
        ]
        
        for forecast_type in forecast_types:
            if forecast_type in request.data_types:
                try:
                    forecast_periods = await provider.get_forecast(
                        request.location,
                        forecast_type,
                        request.units,
                        cnt=request.forecast_hours if forecast_type == WeatherDataType.FORECAST_HOURLY 
                            else request.forecast_days
                    )
                    weather_data.forecast_periods.extend(forecast_periods)
                except Exception as e:
                    self.logger.warning(f"Failed to get {forecast_type.value}: {str(e)}")
        
        # Fetch alerts
        if request.include_alerts:
            try:
                alerts = await provider.get_alerts(request.location)
                weather_data.alerts.extend(alerts)
            except Exception as e:
                self.logger.warning(f"Failed to get weather alerts: {str(e)}")
        
        # Calculate quality scores
        weather_data.data_quality = self._calculate_data_quality(weather_data)
        weather_data.freshness_score = 1.0  # Just fetched
        weather_data.completeness_score = self._calculate_completeness(weather_data, request)
        
        return weather_data

    def _calculate_data_quality(self, weather_data: WeatherData) -> float:
        """Calculate overall data quality score."""
        quality_factors = []
        
        # Check current conditions completeness
        if weather_data.current_conditions:
            conditions = weather_data.current_conditions
            required_fields = ['temperature', 'humidity', 'pressure', 'wind_speed']
            present_fields = sum(1 for field in required_fields 
                               if getattr(conditions, field, None) is not None)
            quality_factors.append(present_fields / len(required_fields))
        
        # Check forecast completeness
        if weather_data.forecast_periods:
            avg_period_quality = sum(
                1.0 if period.conditions.temperature else 0.5
                for period in weather_data.forecast_periods
            ) / len(weather_data.forecast_periods)
            quality_factors.append(avg_period_quality)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

    def _calculate_completeness(self, weather_data: WeatherData, request: WeatherRequest) -> float:
        """Calculate data completeness score based on request."""
        requested_types = len(request.data_types)
        satisfied_types = 0
        
        if WeatherDataType.CURRENT in request.data_types and weather_data.current_conditions:
            satisfied_types += 1
        
        forecast_types = [
            WeatherDataType.FORECAST_HOURLY,
            WeatherDataType.FORECAST_DAILY,
            WeatherDataType.FORECAST_EXTENDED
        ]
        
        if any(ft in request.data_types for ft in forecast_types) and weather_data.forecast_periods:
            satisfied_types += sum(1 for ft in forecast_types if ft in request.data_types)
        
        if request.include_alerts and weather_data.alerts:
            satisfied_types += 0.5  # Partial credit for alerts
        
        return min(1.0, satisfied_types / requested_types)

    async def _cache_weather_data(self, request: WeatherRequest, weather_data: WeatherData) -> None:
        """Cache weather data for future requests."""
        try:
            cache_key = self.cache_manager.generate_cache_key(request, weather_data.provider)
            await self.cache_manager.store_cached_data(cache_key, weather_data)
            
            self.logger.debug(f"Cached weather data: {cache_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to cache weather data: {str(e)}")

    async def _store_weather_memory(self, request: WeatherRequest, weather_data: WeatherData) -> None:
        """Store weather data in user memory for context."""
        try:
            memory_data = {
                'type': 'weather_query',
                'location': {
                    'latitude': weather_data.location.latitude,
                    'longitude': weather_data.location.longitude,
                    'name': weather_data.location.name
                },
                'current_conditions': asdict(weather_data.current_conditions) if weather_data.current_conditions else None,
                'forecast_summary': self._create_forecast_summary(weather_data.forecast_periods),
                'alerts_count': len(weather_data.alerts),
                'data_quality': weather_data.data_quality,
                'timestamp': weather_data.retrieved_at.isoformat(),
                'provider': weather_data.provider.value
            }
            
            await self.memory_manager.store_episodic_memory(
                event_type="weather_query",
                data=memory_data,
                session_id=request.session_id
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store weather memory: {str(e)}")

    def _create_forecast_summary(self, forecast_periods: List[WeatherForecastPeriod]) -> Dict[str, Any]:
        """Create a summary of forecast data."""
        if not forecast_periods:
            return {}
        
        temps = [p.conditions.temperature for p in forecast_periods if p.conditions.temperature]
        conditions = [p.conditions.condition for p in forecast_periods]
        
        return {
            'period_count': len(forecast_periods),
            'temperature_range': {
                'min': min(temps) if temps else None,
                'max': max(temps) if temps else None,
                'avg': sum(temps) / len(temps) if temps else None
            },
            'dominant_condition': max(set(conditions), key=conditions.count) if conditions else None,
            'precipitation_expected': any(p.conditions.precipitation_probability > 50 for p in forecast_periods)
        }

    @handle_exceptions
    async def get_weather_by_location_name(
        self,
        location_name: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> WeatherData:
        """
        Get weather data by location name (with geocoding).
        
        Args:
            location_name: Name of the location
            session_id: Optional session ID
            **kwargs: Additional weather request parameters
            
        Returns:
            Weather data for the location
        """
        # Sanitize location name
        sanitized_name = await self.input_sanitizer.sanitize_text(location_name)
        
        # Try to get location from cache
        if sanitized_name in self.location_cache:
            location = self.location_cache[sanitized_name]
        else:
            # Geocode the location
            location = await self._geocode_location(sanitized_name)
            if location:
                self.location_cache[sanitized_name] = location
        
        if not location:
            raise WeatherError(f"Could not find location: {location_name}")
        
        # Create weather request
        request = WeatherRequest(
            location=location,
            session_id=session_id,
            **kwargs
        )
        
        return await self.get_weather(request)

    async def _geocode_location(self, location_name: str) -> Optional[WeatherLocation]:
        """Geocode a location name to coordinates."""
        # This would implement actual geocoding using a service like Google Maps
        # For now, return None to indicate geocoding is not implemented
        self.logger.warning(f"Geocoding not implemented for: {location_name}")
        return None

    @handle_exceptions
    async def get_weather_alerts(self, location: WeatherLocation) -> List[WeatherAlert]:
        """
        Get weather alerts for a specific location.
        
        Args:
            location: Location to check for alerts
            
        Returns:
            List of weather alerts
        """
        all_alerts = []
        
        for provider_type, provider in self.providers.items():
            try:
                alerts = await provider.get_alerts(location)
                all_alerts.extend(alerts)
            except Exception as e:
                self.logger.warning(f"Failed to get alerts from {provider_type.value}: {str(e)}")
        
        # Remove duplicates and sort by severity
        unique_alerts = self._deduplicate_alerts(all_alerts)
        return sorted(unique_alerts, key=lambda a: a.severity.value, reverse=True)

    def _deduplicate_alerts(self, alerts: List[WeatherAlert]) -> List[WeatherAlert]:
        """Remove duplicate alerts."""
        seen_alerts = set()
        unique_alerts = []
        
        for alert in alerts:
            # Create a hash based on title, start time, and areas
            alert_hash = hashlib.md5(
                f"{alert.title}:{alert.start_time}:{','.join(alert.areas)}".encode()
            ).hexdigest()
            
            if alert_hash not in seen_alerts:
                seen_alerts.add(alert_hash)
                unique_alerts.append(alert)
        
        return unique_alerts

    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all weather providers."""
        status = {}
        
        for provider_type, provider in self.providers.items():
            status[provider_type.value] = {
                'health_score': self.provider_health.get(provider_type, 1.0),
                'success_rate': provider.success_rate,
                'average_response_time': provider.average_response_time,
                'error_count': provider.error_count,
                'requests_today': provider.request_count,
                'rate_limit': provider.rate_limit
            }
        
        return status

    async def _request_processor_loop(self) -> None:
        """Background task to process queued weather requests."""
        while True:
            try:
                # Process any queued requests
                if not self.request_queue.empty():
                    request = await self.request_queue.get()
                    try:
                        await self.get_weather(request)
                    except Exception as e:
                        self.logger.error(f"Failed to process queued request: {str(e)}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Request processor error: {str(e)}")
                await asyncio.sleep(1)

    async def _health_monitor_loop(self) -> None:
        """Background task to monitor provider health."""
        while True:
            try:
                for provider_type, provider in self.providers.items():
                    try:
                        # Test provider with a simple request
                        test_location = WeatherLocation(latitude=0.0, longitude=0.0)
                        
                        start_time = time.time()
                        await provider.get_current_weather(test_location)
                        response_time = time.time() - start_time
                        
                        # Update health score
                        current_health = self.provider_health.get(provider_type, 1.0)
                        if response_time < 5.0:  # Good response time
                            new_health = min(1.0, current_health + 0.1)
                        else:
                            new_health = max(0.0, current_health - 0.1)
                        
                        self.provider_health[provider_type] = new_health
                        
                        # Update metrics
                        self.metrics.set(
                            "weather_provider_health",
                            new_health,
                            tags={'provider': provider_type.value}
                        )
                        
                    except Exception as e:
                        # Provider failed health check
                        current_health = self.provider_health.get(provider_type, 1.0)
                        self.provider_health[provider_type] = max(0.0, current_health - 0.2)
                        
                        self.logger.warning(f"Provider health check failed for {provider_type.value}: {str(e)}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")
                await asyncio.sleep(300)

    async def _cache_cleanup_loop(self) -> None:
        """Background task to cleanup expired cache entries."""
        while True:
            try:
                # Clean up memory cache
                current_time = datetime.now(timezone.utc)
                expired_keys = []
                
                for cache_key, weather_data in self.cache_manager.memory_cache.items():
                    if (weather_data.expires_at and 
                        current_time > weather_data.expires_at):
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    self.cache_manager.memory_cache.pop(key, None)
                    self.cache_manager.cache_access_times.pop(key, None)
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(1800)  # Clean every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(1800)

    async def _handle_component_health_change(self, event) -> None:
        """Handle component health change events."""
        if event.component in ["redis_cache", "database"]:
            self.logger.info(f"Component {event.component} health changed: {event.healthy}")

    async def _handle_session_started(self, event) -> None:
        """Handle session start events."""
        # Coul
