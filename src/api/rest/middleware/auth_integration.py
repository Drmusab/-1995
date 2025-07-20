"""
Enhanced Authentication Middleware for API Endpoints
Author: Drmusab
Last Modified: 2025-07-19 21:43:58 UTC

This module provides comprehensive authentication integration for all API endpoints.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Set, Union
from datetime import datetime, timezone
import json
import functools
from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from src.core.security.authentication import (
    AuthenticationManager, AuthenticationRequest, AuthenticationResult,
    AuthenticationMethod, UserRole, AccountStatus
)
from src.core.security.authorization import (
    AuthorizationManager, AuthorizationContext, ResourceType,
    PermissionType, AccessDecision
)
from src.core.security.sanitization import (
    SecuritySanitizer, SanitizationContext, SanitizationType,
    ContentType, SecurityLevel
)
from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger


security = HTTPBearer()


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enforces authentication on all API endpoints.
    """
    
    def __init__(self, app, container: Container):
        super().__init__(app)
        self.container = container
        self.logger = get_logger(__name__)
        
        # Get security components
        self.auth_manager = container.get(AuthenticationManager)
        self.authz_manager = container.get(AuthorizationManager)
        self.sanitizer = container.get(SecuritySanitizer)
        
        # Configuration
        self.exempt_paths = {
            "/health", "/healthz", "/metrics", "/docs", "/openapi.json",
            "/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/refresh"
        }
        self.require_https = True
        self.session_timeout = 3600  # 1 hour
        
        self.logger.info("Authentication middleware initialized")
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process requests with authentication checks.
        """
        # Check if path is exempt
        if self._is_exempt_path(request.url.path):
            return await call_next(request)
        
        # Enforce HTTPS in production
        if self.require_https and not request.url.scheme == "https":
            if request.headers.get("X-Forwarded-Proto") != "https":
                return JSONResponse(
                    status_code=400,
                    content={"error": "HTTPS required"}
                )
        
        try:
            # Extract authentication token
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return self._unauthorized_response("Missing authorization header")
            
            # Validate token format
            if not auth_header.startswith("Bearer "):
                return self._unauthorized_response("Invalid authorization format")
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Validate token with authentication manager
            auth_token = await self.auth_manager.validate_token(token)
            if not auth_token:
                return self._unauthorized_response("Invalid or expired token")
            
            # Check if user is active
            user_info = await self.auth_manager.get_user_info(auth_token.user_id)
            if not user_info or user_info.get("status") != AccountStatus.ACTIVE.value:
                return self._unauthorized_response("Account inactive or suspended")
            
            # Create authorization context
            auth_context = AuthorizationContext(
                user_id=auth_token.user_id,
                session_id=request.state.session_id if hasattr(request.state, "session_id") else None,
                user_roles=set(user_info.get("roles", [])),
                user_permissions=set(user_info.get("permissions", [])),
                resource_type=self._get_resource_type(request.url.path),
                requested_permission=self._get_permission_type(request.method),
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                authentication_method=auth_token.token_type.value
            )
            
            # Check authorization
            auth_result = await self.authz_manager.check_permission(
                user_id=auth_token.user_id,
                resource_id=request.url.path,
                permission=auth_context.requested_permission.value,
                context=auth_context
            )
            
            if auth_result.decision != AccessDecision.ALLOW:
                return self._forbidden_response(
                    f"Access denied: {auth_result.reason}"
                )
            
            # Attach user context to request
            request.state.user_id = auth_token.user_id
            request.state.user_roles = user_info.get("roles", [])
            request.state.auth_token = auth_token
            request.state.auth_context = auth_context
            
            # Call next middleware/endpoint
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Authentication middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    def _is_exempt_path(self, path: str) -> bool:
        """Check if path is exempt from authentication."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    def _get_resource_type(self, path: str) -> ResourceType:
        """Determine resource type from path."""
        if "/api/" in path:
            return ResourceType.API_ENDPOINT
        elif "/workflow" in path:
            return ResourceType.WORKFLOW
        elif "/session" in path:
            return ResourceType.SESSION
        else:
            return ResourceType.SYSTEM
    
    def _get_permission_type(self, method: str) -> PermissionType:
        """Determine permission type from HTTP method."""
        method_mapping = {
            "GET": PermissionType.READ,
            "POST": PermissionType.CREATE,
            "PUT": PermissionType.UPDATE,
            "PATCH": PermissionType.UPDATE,
            "DELETE": PermissionType.DELETE
        }
        return method_mapping.get(method.upper(), PermissionType.READ)
    
    def _unauthorized_response(self, message: str) -> JSONResponse:
        """Create unauthorized response."""
        return JSONResponse(
            status_code=401,
            content={"error": message},
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _forbidden_response(self, message: str) -> JSONResponse:
        """Create forbidden response."""
        return JSONResponse(
            status_code=403,
            content={"error": message}
        )


def require_auth(
    required_roles: Optional[List[UserRole]] = None,
    required_permissions: Optional[List[str]] = None
):
    """
    Decorator for endpoint-specific authentication requirements.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            *args,
            **kwargs
        ):
            # Verify user has required roles
            if required_roles:
                user_roles = set(request.state.user_roles)
                if not any(role.value in user_roles for role in required_roles):
                    raise HTTPException(
                        status_code=403,
                        detail="Insufficient role privileges"
                    )
            
            # Verify user has required permissions
            if required_permissions:
                auth_context = request.state.auth_context
                for permission in required_permissions:
                    result = await request.app.state.authz_manager.check_permission(
                        user_id=request.state.user_id,
                        resource_id=request.url.path,
                        permission=permission,
                        context=auth_context
                    )
                    if result.decision != AccessDecision.ALLOW:
                        raise HTTPException(
                            status_code=403,
                            detail=f"Missing permission: {permission}"
                        )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user.
    """
    if not hasattr(request.state, "user_id"):
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )
    
    auth_manager = request.app.state.container.get(AuthenticationManager)
    user_info = await auth_manager.get_user_info(request.state.user_id)
    
    if not user_info:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    return user_info
