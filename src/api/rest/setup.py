"""REST API Setup Module"""

from typing import Any, Optional
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config.loader import ConfigLoader
from src.observability.logging.config import get_logger


async def setup_rest_api(assistant: Any) -> Any:
    """
    Setup and configure the REST API.
    
    Args:
        assistant: AI Assistant instance
        
    Returns:
        FastAPI application instance
    """
    logger = get_logger(__name__)
    
    app = FastAPI(
        title="AI Assistant API",
        description="Advanced AI Assistant REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Basic health endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "ai-assistant"}
    
    # Basic status endpoint
    @app.get("/api/v1/status")
    async def get_status():
        """Get system status."""
        return assistant.get_status() if assistant else {"status": "unavailable"}
    
    logger.info("REST API setup completed")
    return app