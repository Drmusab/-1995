"""REST API Setup Module"""

from typing import Any, Optional
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.api.rest.endpoints import create_api_endpoints
from src.observability.logging.config import get_logger


async def setup_rest_api(container: Container) -> FastAPI:
    """
    Setup and configure the REST API with proper component integration.
    
    Args:
        container: Dependency injection container with all components
        
    Returns:
        FastAPI application instance
    """
    logger = get_logger(__name__)
    
    app = FastAPI(
        title="AI Assistant API",
        description="Advanced AI Assistant REST API with Memory Integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Get configuration
    config_loader = container.get(ConfigLoader)
    api_config = config_loader.get('api.rest', {})
    cors_config = api_config.get('middleware', {}).get('cors', {})
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get('allow_origins', ["*"]),
        allow_credentials=cors_config.get('allow_credentials', True),
        allow_methods=cors_config.get('allow_methods', ["*"]),
        allow_headers=cors_config.get('allow_headers', ["*"]),
    )
    
    # Create and initialize API endpoints
    api_endpoints = create_api_endpoints(container)
    await api_endpoints.initialize()
    
    # Include API routes
    app.include_router(api_endpoints.router)
    
    # Basic health endpoint (legacy)
    @app.get("/health")
    async def health_check():
        """Legacy health check endpoint."""
        return {"status": "healthy", "service": "ai-assistant"}
    
    # Basic status endpoint
    @app.get("/api/v1/status")
    async def get_status():
        """Get API status."""
        try:
            # Get component statuses
            components = {}
            
            try:
                from src.assistant.session_manager import EnhancedSessionManager
                session_manager = container.get(EnhancedSessionManager)
                components["session_manager"] = "available"
            except:
                components["session_manager"] = "unavailable"
            
            try:
                from src.assistant.session_memory_integrator import SessionMemoryIntegrator
                memory_integrator = container.get(SessionMemoryIntegrator)
                components["memory_integrator"] = "available"
            except:
                components["memory_integrator"] = "unavailable"
            
            try:
                from src.assistant.core_engine import EnhancedCoreEngine
                core_engine = container.get(EnhancedCoreEngine)
                components["core_engine"] = "available"
            except:
                components["core_engine"] = "unavailable"
            
            return {
                "status": "online",
                "version": "1.0.0",
                "components": components,
                "endpoints": {
                    "chat": "/api/v1/chat",
                    "sessions": "/api/v1/sessions",
                    "memory": "/api/v1/memory/query",
                    "health": "/api/v1/health"
                }
            }
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    logger.info("REST API setup completed with integrated endpoints")
    return app


async def start_api_server(app: FastAPI, config: dict):
    """Start the API server."""
    logger = get_logger(__name__)
    
    host = config.get('host', '0.0.0.0')
    port = config.get('port', 8000)
    
    logger.info(f"Starting REST API server on {host}:{port}")
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()