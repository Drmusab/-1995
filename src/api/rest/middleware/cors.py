"""
Cross-Origin Resource Sharing (CORS) Middleware
Author: Drmusab
Last Modified: 2025-01-20 11:34:00 UTC

This module provides comprehensive CORS handling for the AI assistant API,
supporting dynamic origin validation, credential management, preflight caching,
and integration with the core system's security and configuration components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Union, Pattern, Tuple
import asyncio
import re
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from urllib.parse import urlparse
import json

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SecurityViolationDetected, AccessAttempt, CorsViolationDetected,
    SecurityPolicyUpdated, ComponentHealthChanged, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Third-party imports (FastAPI/Starlette for this example)
try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response, PlainTextResponse
    from starlette.types import ASGIApp
except ImportError:
    # Fallback for environments without Starlette
    BaseHTTPMiddleware = object
    Request = object
    Response = object
    PlainTextResponse = object
    ASGIApp = object


class CorsPolicy(Enum):
    """CORS policy levels for different environments."""
    STRICT = "strict"           # Production-ready strict policy
    DEVELOPMENT = "development" # Relaxed for development
    TESTING = "testing"         # Permissive for testing
    CUSTOM = "custom"           # User-defined policy


class OriginValidationType(Enum):
    """Types of origin validation."""
    EXACT_MATCH = "exact_match"
    WILDCARD = "wildcard"
    REGEX = "regex"
    DOMAIN_SUFFIX = "domain_suffix"
    SUBDOMAIN = "subdomain"


@dataclass
class OriginRule:
    """Rule for validating allowed origins."""
    pattern: str
    validation_type: OriginValidationType = OriginValidationType.EXACT_MATCH
    allowed_methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE", "OPTIONS"})
    allowed_headers: Set[str] = field(default_factory=set)
    expose_headers: Set[str] = field(default_factory=set)
    allow_credentials: bool = True
    max_age: int = 86400  # 24 hours
    description: Optional[str] = None
    priority: int = 0
    
    def matches(self, origin: str) -> bool:
        """Check if the origin matches this rule."""
        try:
            if self.validation_type == OriginValidationType.EXACT_MATCH:
                return origin.lower() == self.pattern.lower()
            
            elif self.validation_type == OriginValidationType.WILDCARD:
                # Convert wildcard pattern to regex
                regex_pattern = self.pattern.replace("*", ".*")
                return bool(re.match(f"^{regex_pattern}$", origin, re.IGNORECASE))
            
            elif self.validation_type == OriginValidationType.REGEX:
                return bool(re.match(self.pattern, origin, re.IGNORECASE))
            
            elif self.validation_type == OriginValidationType.DOMAIN_SUFFIX:
                return origin.lower().endswith(self.pattern.lower())
            
            elif self.validation_type == OriginValidationType.SUBDOMAIN:
                # Allow subdomains of the specified domain
                domain = self.pattern.lower()
                origin_lower = origin.lower()
                return (origin_lower == domain or 
                       origin_lower.endswith(f".{domain}"))
            
            return False
            
        except Exception:
            return False


@dataclass
class CorsConfiguration:
    """Comprehensive CORS configuration."""
    # Policy and security
    policy: CorsPolicy = CorsPolicy.STRICT
    enable_cors: bool = True
    
    # Origin rules
    origin_rules: List[OriginRule] = field(default_factory=list)
    default_allow_all_origins: bool = False
    
    # Methods and headers
    allowed_methods: Set[str] = field(default_factory=lambda: {
        "GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"
    })
    allowed_headers: Set[str] = field(default_factory=lambda: {
        "Accept", "Accept-Language", "Content-Language", "Content-Type",
        "Authorization", "X-Requested-With", "X-Session-ID", "X-User-ID",
        "X-Request-ID", "X-API-Key", "Cache-Control"
    })
    expose_headers: Set[str] = field(default_factory=lambda: {
        "X-Request-ID", "X-Response-Time", "X-Rate-Limit-Remaining",
        "X-Rate-Limit-Reset", "X-Total-Count"
    })
    
    # Credentials and caching
    allow_credentials: bool = True
    max_age: int = 86400  # 24 hours
    preflight_continue: bool = False
    options_success_status: int = 204
    
    # Security features
    enable_origin_validation: bool = True
    enable_method_validation: bool = True
    enable_header_validation: bool = True
    log_violations: bool = True
    block_suspicious_requests: bool = True
    
    # Rate limiting for preflight requests
    preflight_rate_limit: int = 100  # per minute
    enable_preflight_caching: bool = True
    
    # Monitoring and debugging
    enable_detailed_logging: bool = False
    track_metrics: bool = True
    include_debug_headers: bool = False


class CorsViolation(Exception):
    """Exception raised for CORS violations."""
    
    def __init__(self, message: str, violation_type: str, origin: str = None,
                 method: str = None, headers: List[str] = None):
        super().__init__(message)
        self.violation_type = violation_type
        self.origin = origin
        self.method = method
        self.headers = headers or []
        self.timestamp = datetime.now(timezone.utc)


class PreflightCache:
    """Cache for CORS preflight responses."""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    def _generate_cache_key(self, origin: str, method: str, headers: Set[str]) -> str:
        """Generate cache key for preflight request."""
        headers_str = ",".join(sorted(headers))
        return f"{origin}:{method}:{hash(headers_str)}"
    
    async def get(self, origin: str, method: str, headers: Set[str]) -> Optional[Dict[str, Any]]:
        """Get cached preflight response."""
        async with self.lock:
            cache_key = self._generate_cache_key(origin, method, headers)
            
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                
                # Check if cache entry is still valid
                if time.time() - cached_data["timestamp"] < cached_data["max_age"]:
                    self.access_times[cache_key] = time.time()
                    return cached_data["response"]
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
                    self.access_times.pop(cache_key, None)
            
            return None
    
    async def set(self, origin: str, method: str, headers: Set[str], 
                  response: Dict[str, Any], max_age: int) -> None:
        """Cache preflight response."""
        async with self.lock:
            cache_key = self._generate_cache_key(origin, method, headers)
            
            # Clean up old entries if cache is full
            if len(self.cache) >= self.max_size:
                await self._cleanup_old_entries()
            
            self.cache[cache_key] = {
                "response": response,
                "timestamp": time.time(),
                "max_age": max_age
            }
            self.access_times[cache_key] = time.time()
    
    async def _cleanup_old_entries(self) -> None:
        """Remove old cache entries."""
        if not self.access_times:
            return
        
        # Sort by access time and remove oldest 25%
        sorted_entries = sorted(self.access_times.items(), key=lambda x: x[1])
        entries_to_remove = len(sorted_entries) // 4
        
        for cache_key, _ in sorted_entries[:entries_to_remove]:
            self.cache.pop(cache_key, None)
            self.access_times.pop(cache_key, None)
    
    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self.lock:
            self.cache.clear()
            self.access_times.clear()


class CorsMiddleware(BaseHTTPMiddleware):
    """
    Advanced CORS Middleware for the AI Assistant API.
    
    Features:
    - Dynamic origin validation with multiple rule types
    - Intelligent preflight caching
    - Comprehensive security validation
    - Integration with core system components
    - Detailed monitoring and logging
    - Rate limiting for preflight requests
    - Violation detection and blocking
    - Development-friendly debugging features
    """
    
    def __init__(
        self,
        app: ASGIApp,
        container: Container,
        config: Optional[CorsConfiguration] = None
    ):
        """
        Initialize CORS middleware.
        
        Args:
            app: ASGI application
            container: Dependency injection container
            config: CORS configuration
        """
        super().__init__(app)
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core services
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Security components
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.logger.warning("Security components not available")
        
        # Observability
        try:
            self.metrics = container.get(MetricsCollector)
            self.tracer = container.get(TraceManager)
        except Exception:
            self.metrics = None
            self.tracer = None
            self.logger.warning("Observability components not available")
        
        # Configuration
        self.cors_config = config or self._load_default_configuration()
        
        # Runtime state
        self.preflight_cache = PreflightCache()
        self.violation_counts: Dict[str, int] = {}
        self.blocked_origins: Set[str] = set()
        self.request_counts: Dict[str, Dict[str, int]] = {}
        
        # Setup
        self._setup_monitoring()
        self._setup_health_check()
        self._load_origin_rules()
        
        self.logger.info("CorsMiddleware initialized successfully")
    
    def _load_default_configuration(self) -> CorsConfiguration:
        """Load default CORS configuration from system config."""
        cors_settings = self.config_loader.get("cors", {})
        
        # Determine policy based on environment
        environment = self.config_loader.get("environment", "development")
        if environment == "production":
            policy = CorsPolicy.STRICT
        elif environment == "testing":
            policy = CorsPolicy.TESTING
        else:
            policy = CorsPolicy.DEVELOPMENT
        
        config = CorsConfiguration(
            policy=policy,
            enable_cors=cors_settings.get("enable", True),
            allow_credentials=cors_settings.get("allow_credentials", True),
            max_age=cors_settings.get("max_age", 86400),
            enable_detailed_logging=cors_settings.get("detailed_logging", environment != "production"),
            track_metrics=cors_settings.get("track_metrics", True)
        )
        
        # Load allowed methods
        if "allowed_methods" in cors_settings:
            config.allowed_methods = set(cors_settings["allowed_methods"])
        
        # Load allowed headers
        if "allowed_headers" in cors_settings:
            config.allowed_headers.update(cors_settings["allowed_headers"])
        
        # Load expose headers
        if "expose_headers" in cors_settings:
            config.expose_headers.update(cors_settings["expose_headers"])
        
        return config
    
    def _setup_monitoring(self) -> None:
        """Setup monitoring metrics."""
        if not self.metrics:
            return
        
        try:
            # Register CORS metrics
            self.metrics.register_counter("cors_requests_total")
            self.metrics.register_counter("cors_preflight_requests_total")
            self.metrics.register_counter("cors_violations_total")
            self.metrics.register_counter("cors_blocked_requests_total")
            self.metrics.register_histogram("cors_processing_duration_seconds")
            self.metrics.register_gauge("cors_preflight_cache_size")
            self.metrics.register_counter("cors_cache_hits_total")
            self.metrics.register_counter("cors_cache_misses_total")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup CORS monitoring: {str(e)}")
    
    def _setup_health_check(self) -> None:
        """Setup health check for CORS middleware."""
        if not self.health_check:
            return
        
        self.health_check.register_component("cors_middleware", self._health_check_callback)
    
    def _load_origin_rules(self) -> None:
        """Load origin rules from configuration."""
        origins_config = self.config_loader.get("cors.origins", [])
        
        # Add default rules based on policy
        if self.cors_config.policy == CorsPolicy.DEVELOPMENT:
            # Allow localhost and common development origins
            dev_rules = [
                OriginRule("http://localhost:3000", OriginValidationType.EXACT_MATCH),
                OriginRule("http://localhost:8080", OriginValidationType.EXACT_MATCH),
                OriginRule("http://127.0.0.1:*", OriginValidationType.WILDCARD),
                OriginRule("http://localhost:*", OriginValidationType.WILDCARD),
            ]
            self.cors_config.origin_rules.extend(dev_rules)
        
        elif self.cors_config.policy == CorsPolicy.TESTING:
            # Allow all origins for testing
            test_rule = OriginRule("*", OriginValidationType.WILDCARD)
            self.cors_config.origin_rules.append(test_rule)
        
        # Load custom rules from configuration
        for origin_config in origins_config:
            if isinstance(origin_config, str):
                # Simple string origin
                rule = OriginRule(origin_config, OriginValidationType.EXACT_MATCH)
            elif isinstance(origin_config, dict):
                # Detailed origin configuration
                rule = OriginRule(
                    pattern=origin_config["pattern"],
                    validation_type=OriginValidationType(origin_config.get("type", "exact_match")),
                    allowed_methods=set(origin_config.get("methods", [])) or self.cors_config.allowed_methods,
                    allowed_headers=set(origin_config.get("headers", [])),
                    allow_credentials=origin_config.get("allow_credentials", True),
                    max_age=origin_config.get("max_age", self.cors_config.max_age),
                    description=origin_config.get("description"),
                    priority=origin_config.get("priority", 0)
                )
            else:
                continue
            
            self.cors_config.origin_rules.append(rule)
        
        # Sort rules by priority (higher priority first)
        self.cors_config.origin_rules.sort(key=lambda x: x.priority, reverse=True)
        
        self.logger.info(f"Loaded {len(self.cors_config.origin_rules)} CORS origin rules")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process CORS for incoming requests.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/application in chain
            
        Returns:
            HTTP response with appropriate CORS headers
        """
        if not self.cors_config.enable_cors:
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            with self.tracer.trace("cors_processing") if self.tracer else None as span:
                if span:
                    span.set_attributes({
                        "cors.origin": request.headers.get("origin", ""),
                        "cors.method": request.method,
                        "cors.is_preflight": request.method == "OPTIONS"
                    })
                
                # Extract CORS-related information
                origin = request.headers.get("origin")
                method = request.method
                
                # Update metrics
                if self.metrics:
                    self.metrics.increment("cors_requests_total")
                
                # Handle preflight requests
                if method == "OPTIONS" and origin:
                    return await self._handle_preflight_request(request)
                
                # Handle actual requests
                response = await call_next(request)
                
                # Add CORS headers to response
                if origin:
                    response = await self._add_cors_headers(request, response)
                
                # Record processing time
                processing_time = time.time() - start_time
                if self.metrics:
                    self.metrics.record("cors_processing_duration_seconds", processing_time)
                
                return response
                
        except CorsViolation as e:
            # Handle CORS violations
            return await self._handle_cors_violation(request, e)
        
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"CORS middleware error: {str(e)}")
            
            if self.event_bus:
                await self.event_bus.emit(ErrorOccurred(
                    component="cors_middleware",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity="medium"
                ))
            
            # Continue with request processing
            return await call_next(request)
    
    async def _handle_preflight_request(self, request: Request) -> Response:
        """
        Handle CORS preflight OPTIONS requests.
        
        Args:
            request: Preflight request
            
        Returns:
            Preflight response
        """
        origin = request.headers.get("origin")
        requested_method = request.headers.get("access-control-request-method")
        requested_headers = self._parse_requested_headers(
            request.headers.get("access-control-request-headers", "")
        )
        
        if self.metrics:
            self.metrics.increment("cors_preflight_requests_total")
        
        # Check preflight cache first
        if self.cors_config.enable_preflight_caching:
            cached_response = await self.preflight_cache.get(
                origin, requested_method or "", requested_headers
            )
            if cached_response:
                if self.metrics:
                    self.metrics.increment("cors_cache_hits_total")
                return self._create_preflight_response(cached_response)
        
        if self.metrics:
            self.metrics.increment("cors_cache_misses_total")
        
        # Validate origin
        origin_rule = await self._validate_origin(origin)
        if not origin_rule:
            raise CorsViolation(
                f"Origin '{origin}' not allowed",
                "origin_not_allowed",
                origin=origin
            )
        
        # Validate method
        if (requested_method and 
            self.cors_config.enable_method_validation and
            requested_method not in origin_rule.allowed_methods):
            raise CorsViolation(
                f"Method '{requested_method}' not allowed for origin '{origin}'",
                "method_not_allowed",
                origin=origin,
                method=requested_method
            )
        
        # Validate headers
        if self.cors_config.enable_header_validation:
            forbidden_headers = requested_headers - origin_rule.allowed_headers - self.cors_config.allowed_headers
            if forbidden_headers:
                raise CorsViolation(
                    f"Headers {forbidden_headers} not allowed for origin '{origin}'",
                    "headers_not_allowed",
                    origin=origin,
                    headers=list(forbidden_headers)
                )
        
        # Create preflight response data
        response_data = {
            "origin": origin,
            "methods": list(origin_rule.allowed_methods),
            "headers": list(origin_rule.allowed_headers | self.cors_config.allowed_headers),
            "expose_headers": list(origin_rule.expose_headers | self.cors_config.expose_headers),
            "allow_credentials": origin_rule.allow_credentials,
            "max_age": origin_rule.max_age
        }
        
        # Cache the response
        if self.cors_config.enable_preflight_caching:
            await self.preflight_cache.set(
                origin, requested_method or "", requested_headers,
                response_data, origin_rule.max_age
            )
        
        # Log successful preflight
        if self.cors_config.enable_detailed_logging:
            self.logger.debug(
                f"CORS preflight approved: origin={origin}, "
                f"method={requested_method}, headers={requested_headers}"
            )
        
        return self._create_preflight_response(response_data)
    
    def _parse_requested_headers(self, headers_str: str) -> Set[str]:
        """Parse requested headers from preflight request."""
        if not headers_str:
            return set()
        
        # Split by comma and clean up
        headers = {h.strip().lower() for h in headers_str.split(",") if h.strip()}
        return headers
    
    def _create_preflight_response(self, response_data: Dict[str, Any]) -> Response:
        """Create preflight response with CORS headers."""
        headers = {
            "Access-Control-Allow-Origin": response_data["origin"],
            "Access-Control-Allow-Methods": ",".join(response_data["methods"]),
            "Access-Control-Allow-Headers": ",".join(response_data["headers"]),
            "Access-Control-Max-Age": str(response_data["max_age"]),
            "Vary": "Origin"
        }
        
        if response_data["expose_headers"]:
            headers["Access-Control-Expose-Headers"] = ",".join(response_data["expose_headers"])
        
        if response_data["allow_credentials"]:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        # Add debug headers if enabled
        if self.cors_config.include_debug_headers:
            headers["X-CORS-Debug"] = "preflight-approved"
            headers["X-CORS-Cache"] = "miss"  # Could be enhanced to track cache status
        
        return Response(
            status_code=self.cors_config.options_success_status,
            headers=headers
        )
    
    async def _add_cors_headers(self, request: Request, response: Response) -> Response:
        """
        Add CORS headers to actual request response.
        
        Args:
            request: Original request
            response: Response to modify
            
        Returns:
            Response with CORS headers added
        """
        origin = request.headers.get("origin")
        if not origin:
            return response
        
        # Validate origin
        origin_rule = await self._validate_origin(origin)
        if not origin_rule:
            # Don't add CORS headers for invalid origins
            return response
        
        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        
        if origin_rule.expose_headers or self.cors_config.expose_headers:
            expose_headers = origin_rule.expose_headers | self.cors_config.expose_headers
            response.headers["Access-Control-Expose-Headers"] = ",".join(expose_headers)
        
        if origin_rule.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Add debug headers if enabled
        if self.cors_config.include_debug_headers:
            response.headers["X-CORS-Debug"] = "actual-request-approved"
            response.headers["X-CORS-Rule-Priority"] = str(origin_rule.priority)
        
        return response
    
    async def _validate_origin(self, origin: str) -> Optional[OriginRule]:
        """
        Validate origin against configured rules.
        
        Args:
            origin: Origin to validate
            
        Returns:
            Matching origin rule or None if not allowed
        """
        if not origin:
            return None
        
        # Check if origin is blocked
        if origin in self.blocked_origins:
            return None
        
        # Check against origin rules
        for rule in self.cors_config.origin_rules:
            if rule.matches(origin):
                return rule
        
        # Check default allow all
        if self.cors_config.default_allow_all_origins:
            return OriginRule("*", OriginValidationType.WILDCARD)
        
        return None
    
    async def _handle_cors_violation(self, request: Request, violation: CorsViolation) -> Response:
        """
        Handle CORS violations.
        
        Args:
            request: Request that caused violation
            violation: CORS violation details
            
        Returns:
            Error response
        """
        origin = violation.origin or request.headers.get("origin", "unknown")
        
        # Update violation tracking
        self.violation_counts[origin] = self.violation_counts.get(origin, 0) + 1
        
        # Block suspicious origins
        if (self.cors_config.block_suspicious_requests and 
            self.violation_counts[origin] > 10):
            self.blocked_origins.add(origin)
            self.logger.warning(f"Blocked origin due to repeated violations: {origin}")
        
        # Emit security event
        if self.event_bus:
            await self.event_bus.emit(CorsViolationDetected(
                origin=origin,
                violation_type=violation.violation_type,
                method=violation.method,
                headers=violation.headers,
                user_agent=request.headers.get("user-agent", ""),
                ip_address=request.client.host if request.client else "unknown"
            ))
        
        # Update metrics
        if self.metrics:
            self.metrics.increment("cors_violations_total", 
                                tags={"violation_type": violation.violation_type})
            if origin in self.blocked_origins:
                self.metrics.increment("cors_blocked_requests_total")
        
        # Log violation
        if self.cors_config.log_violations:
            self.logger.warning(
                f"CORS violation: {violation.violation_type} - {str(violation)} "
                f"(origin: {origin}, method: {violation.method})"
            )
        
        # Return appropriate error response
        if violation.violation_type == "origin_not_allowed":
            status_code = 403
            message = "Origin not allowed by CORS policy"
        elif violation.violation_type == "method_not_allowed":
            status_code = 405
            message = "Method not allowed by CORS policy"
        elif violation.violation_type == "headers_not_allowed":
            status_code = 400
            message = "Headers not allowed by CORS policy"
        else:
            status_code = 400
            message = "CORS policy violation"
        
        return PlainTextResponse(
            content=message,
            status_code=status_code,
            headers={"X-CORS-Error": violation.violation_type}
        )
    
    async def update_origin_rules(self, rules: List[OriginRule]) -> None:
        """
        Update origin rules dynamically.
        
        Args:
            rules: New list of origin rules
        """
        self.cors_config.origin_rules = sorted(rules, key=lambda x: x.priority, reverse=True)
        
        # Clear caches
        await self.preflight_cache.clear()
        self.blocked_origins.clear()
        self.violation_counts.clear()
        
        # Emit policy update event
        if self.event_bus:
            await self.event_bus.emit(SecurityPolicyUpdated(
                policy_type="cors_origins",
                rules_count=len(rules),
                updated_by="system"
            ))
        
        self.logger.info(f"Updated CORS origin rules: {len(rules)} rules loaded")
    
    async def add_origin_rule(self, rule: OriginRule) -> None:
        """
        Add a new origin rule.
        
        Args:
            rule: Origin rule to add
        """
        self.cors_config.origin_rules.append(rule)
        self.cors_config.origin_rules.sort(key=lambda x: x.priority, reverse=True)
        
        self.logger.info(f"Added CORS origin rule: {rule.pattern} ({rule.validation_type.value})")
    
    async def remove_origin_rule(self, pattern: str) -> bool:
        """
        Remove an origin rule by pattern.
        
        Args:
            pattern: Pattern to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        original_count = len(self.cors_config.origin_rules)
        self.cors_config.origin_rules = [
            rule for rule in self.cors_config.origin_rules 
            if rule.pattern != pattern
        ]
        
        removed = len(self.cors_config.origin_rules) < original_count
        if removed:
            self.logger.info(f"Removed CORS origin rule: {pattern}")
        
        return removed
    
    def get_origin_rules(self) -> List[Dict[str, Any]]:
        """Get current origin rules."""
        return [
            {
                "pattern": rule.pattern,
                "validation_type": rule.validation_type.value,
                "allowed_methods": list(rule.allowed_methods),
                "allowed_headers": list(rule.allowed_headers),
                "allow_credentials": rule.allow_credentials,
                "max_age": rule.max_age,
                "priority": rule.priority,
                "description": rule.description
            }
            for rule in self.cors_config.origin_rules
        ]
    
    def get_cors_statistics(self) -> Dict[str, Any]:
        """Get CORS middleware statistics."""
        return {
            "configuration": {
                "policy": self.cors_config.policy.value,
                "enabled": self.cors_config.enable_cors,
                "allow_credentials": self.cors_config.allow_credentials,
                "max_age": self.cors_config.max_age,
                "origin_rules_count": len(self.cors_config.origin_rules)
            },
            "runtime": {
                "blocked_origins_count": len(self.blocked_origins),
                "violation_counts": dict(self.violation_counts),
                "preflight_cache_size": len(self.preflight_cache.cache),
                "blocked_origins": list(self.blocked_origins)
            },
            "policy_details": {
                "allowed_methods": list(self.cors_config.allowed_methods),
                "allowed_headers": list(self.cors_config.allowed_headers),
                "expose_headers": list(self.cors_config.expose_headers),
                "preflight_caching": self.cors_config.enable_preflight_caching,
                "security_features": {
                    "origin_validation": self.cors_config.enable_origin_validation,
                    "method_validation": self.cors_config.enable_method_validation,
                    "header_validation": self.cors_config.enable_header_validation,
                    "violation_blocking": self.cors_config.block_suspicious_requests
                }
            }
        }
    
    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for CORS middleware."""
        try:
            cache_size = len(self.preflight_cache.cache)
            blocked_count = len(self.blocked_origins)
            rules_count = len(self.cors_config.origin_rules)
            
            status = "healthy"
            if blocked_count > 100:  # Too many blocked origins might indicate issues
                status = "degraded"
            
            return {
                "status": status,
                "enabled": self.cors_config.enable_cors,
                "policy": self.cors_config.policy.value,
                "origin_rules_count": rules_count,
                "preflight_cache_size": cache_size,
                "blocked_origins_count": blocked_count,
                "total_violations": sum(self.violation_counts.values())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup CORS middleware resources."""
        try:
            await self.preflight_cache.clear()
            self.blocked_origins.clear()
            self.violation_counts.clear()
            self.request_counts.clear()
            
            self.logger.info("CORS middleware cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during CORS middleware cleanup: {str(e)}")


# Convenience functions for common CORS configurations

def create_development_cors_middleware(app: ASGIApp, container: Container) -> CorsMiddleware:
    """Create CORS middleware with development-friendly settings."""
    config = CorsConfiguration(
        policy=CorsPolicy.DEVELOPMENT,
        enable_cors=True,
        default_allow_all_origins=True,
        allow_credentials=True,
        enable_detailed_logging=True,
        include_debug_headers=True,
        block_suspicious_requests=False
    )
    
    return CorsMiddleware(app, container, config)


def create_production_cors_middleware(app: ASGIApp, container: Container, 
                                    allowed_origins: List[str]) -> CorsMiddleware:
    """Create CORS middleware with production-ready settings."""
    origin_rules = [
        OriginRule(origin, OriginValidationType.EXACT_MATCH)
        for origin in allowed_origins
    ]
    
    config = CorsConfiguration(
        policy=CorsPolicy.STRICT,
        enable_cors=True,
        origin_rules=origin_rules,
        allow_credentials=True,
        enable_detailed_logging=False,
        include_debug_headers=False,
        block_suspicious_requests=True,
        log_violations=True
    )
    
    return CorsMiddleware(app, container, config)


def create_api_cors_middleware(app: ASGIApp, container: Container) -> CorsMiddleware:
    """Create CORS middleware optimized for API usage."""
    config = CorsConfiguration(
        policy=CorsPolicy.CUSTOM,
        enable_cors=True,
        allowed_methods={"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        allowed_headers={
            "Content-Type", "Authorization", "X-API-Key", 
            "X-Requested-With", "X-Session-ID"
        },
        expose_headers={
            "X-Request-ID", "X-Rate-Limit-Remaining", 
            "X-Rate-Limit-Reset", "X-Total-Count"
        },
        allow_credentials=False,  # API keys instead of credentials
        max_age=3600,  # 1 hour
        enable_preflight_caching=True,
        track_metrics=True
    )
    
    return CorsMiddleware(app, container, config)
