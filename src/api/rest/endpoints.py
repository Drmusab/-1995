"""REST API Endpoints with Memory and Session Integration"""

from typing import Any, Dict, List, Optional
import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends, Body, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from src.core.dependency_injection import Container
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.assistant.session_manager import EnhancedSessionManager, SessionType
from src.assistant.session_memory_integrator import SessionMemoryIntegrator
from src.assistant.core_engine import EnhancedCoreEngine, MultimodalInput, ProcessingContext
from src.observability.logging.config import get_logger

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    """Chat message request."""
    message: str = Field(..., description="User message", max_length=4096)
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    memory_enhanced: bool = Field(True, description="Enable memory enhancement")


class ChatResponse(BaseModel):
    """Chat message response."""
    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Session ID")
    confidence: float = Field(..., description="Response confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    memory_context_used: bool = Field(..., description="Whether memory context was used")
    context_elements_count: int = Field(..., description="Number of context elements used")


class SessionCreateRequest(BaseModel):
    """Session creation request."""
    user_id: Optional[str] = Field(None, description="User ID")
    session_type: str = Field("interactive", description="Session type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session metadata")


class SessionResponse(BaseModel):
    """Session response."""
    session_id: str = Field(..., description="Session ID")
    user_id: Optional[str] = Field(None, description="User ID")
    state: str = Field(..., description="Session state")
    created_at: str = Field(..., description="Creation timestamp")
    interaction_count: int = Field(..., description="Number of interactions")
    memory_item_count: int = Field(..., description="Number of memory items")


class MemoryQueryRequest(BaseModel):
    """Memory query request."""
    query: str = Field(..., description="Query string", max_length=1024)
    session_id: Optional[str] = Field(None, description="Session ID filter")
    limit: int = Field(10, description="Maximum results", ge=1, le=50)
    memory_types: Optional[List[str]] = Field(default_factory=list, description="Memory types to search")


class MemoryItem(BaseModel):
    """Memory item response."""
    memory_id: str = Field(..., description="Memory ID")
    content: str = Field(..., description="Memory content")
    memory_type: str = Field(..., description="Memory type")
    relevance: float = Field(..., description="Relevance score")
    timestamp: str = Field(..., description="Creation timestamp")


class MemoryResponse(BaseModel):
    """Memory query response."""
    memories: List[MemoryItem] = Field(..., description="Retrieved memories")
    total_count: int = Field(..., description="Total matching memories")
    query_time: float = Field(..., description="Query time in seconds")


# Security dependency
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Get current user from JWT token."""
    if not credentials:
        return None
    
    try:
        # Import JWT library
        import jwt
        from src.core.security.authentication import AuthenticationManager
        
        # Get the token from the Authorization header
        token = credentials.credentials
        
        # For now, implement basic JWT validation without container dependency
        # In a real implementation, we would get the secret from configuration
        jwt_secret = "your-secret-key-here"  # This should come from config
        jwt_algorithm = "HS256"
        
        try:
            # Decode and validate the JWT token
            payload = jwt.decode(token, jwt_secret, algorithms=[jwt_algorithm])
            user_id = payload.get("user_id")
            
            # Check token expiration (jwt.decode already handles this)
            # Additional validation could be added here
            
            return user_id
            
        except jwt.ExpiredSignatureError:
            # Token has expired
            return None
        except jwt.InvalidTokenError:
            # Invalid token
            return None
            
    except ImportError:
        # JWT library not available, use basic validation
        # This is a fallback for development environments
        if credentials.credentials.startswith("dev-"):
            return "user-123"
        return None
    except Exception:
        # Any other error, reject the token
        return None


class APIEndpoints:
    """REST API endpoints with memory and session integration."""
    
    def __init__(self, container: Container):
        """Initialize API endpoints."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Get dependencies
        self.config_loader = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        
        # Components (will be available after initialization)
        self.session_manager: Optional[EnhancedSessionManager] = None
        self.memory_integrator: Optional[SessionMemoryIntegrator] = None
        self.core_engine: Optional[EnhancedCoreEngine] = None
        
        # Create router
        self.router = APIRouter(prefix="/api/v1", tags=["assistant"])
        self._setup_routes()
    
    async def initialize(self):
        """Initialize API endpoints with dependencies."""
        try:
            self.session_manager = self.container.get(EnhancedSessionManager)
            self.memory_integrator = self.container.get(SessionMemoryIntegrator)
            self.core_engine = self.container.get(EnhancedCoreEngine)
            
            self.logger.info("API endpoints initialized with core components")
        except Exception as e:
            self.logger.error(f"Failed to initialize API components: {e}")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.router.post("/chat", response_model=ChatResponse)
        async def chat(
            request: ChatRequest,
            current_user: Optional[str] = Depends(get_current_user)
        ) -> ChatResponse:
            """Process chat message with memory-enhanced context."""
            try:
                start_time = datetime.now(timezone.utc)
                
                # Get or create session
                session_id = request.session_id
                if not session_id:
                    if not self.session_manager:
                        raise HTTPException(status_code=503, detail="Session manager not available")
                    
                    session_id = await self.session_manager.create_session(
                        user_id=request.user_id or current_user,
                        session_type=SessionType.INTERACTIVE,
                        metadata=request.context
                    )
                
                # Process message
                if self.core_engine:
                    # Use core engine for advanced processing
                    result = await self.core_engine.process_message(
                        message=request.message,
                        session_id=session_id,
                        user_id=request.user_id or current_user,
                        context_data=request.context
                    )
                    
                    response_text = result.get("text", "I'm here to help!")
                    confidence = result.get("confidence", 0.8)
                    memory_enhanced = result.get("memory_enhanced", False)
                    context_used = result.get("context_used", 0)
                    
                elif self.session_manager:
                    # Fallback to session manager
                    result = await self.session_manager.process_message(
                        session_id=session_id,
                        message=request.message,
                        user_id=request.user_id or current_user,
                        metadata=request.context
                    )
                    
                    response_text = result.get("text", "I'm here to help!")
                    confidence = 0.8
                    memory_enhanced = result.get("context_used", False)
                    context_used = result.get("memory_context_size", 0)
                    
                else:
                    raise HTTPException(status_code=503, detail="Processing engines not available")
                
                processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                return ChatResponse(
                    response=response_text,
                    session_id=session_id,
                    confidence=confidence,
                    processing_time=processing_time,
                    memory_context_used=memory_enhanced,
                    context_elements_count=context_used
                )
                
            except Exception as e:
                self.logger.error(f"Chat processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
        @self.router.post("/sessions", response_model=SessionResponse)
        async def create_session(
            request: SessionCreateRequest,
            current_user: Optional[str] = Depends(get_current_user)
        ) -> SessionResponse:
            """Create a new session."""
            try:
                if not self.session_manager:
                    raise HTTPException(status_code=503, detail="Session manager not available")
                
                # Map session type
                session_type_map = {
                    "interactive": SessionType.INTERACTIVE,
                    "batch": SessionType.BATCH,
                    "api": SessionType.API,
                    "authenticated": SessionType.AUTHENTICATED,
                    "guest": SessionType.GUEST
                }
                
                session_type = session_type_map.get(request.session_type, SessionType.INTERACTIVE)
                
                session_id = await self.session_manager.create_session(
                    user_id=request.user_id or current_user,
                    session_type=session_type,
                    metadata=request.metadata
                )
                
                session = await self.session_manager.get_session(session_id)
                
                return SessionResponse(
                    session_id=session_id,
                    user_id=session.context.user_id,
                    state=session.state.value,
                    created_at=session.created_at.isoformat(),
                    interaction_count=session.interaction_count,
                    memory_item_count=session.memory_item_count
                )
                
            except Exception as e:
                self.logger.error(f"Session creation error: {e}")
                raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")
        
        @self.router.get("/sessions/{session_id}", response_model=SessionResponse)
        async def get_session(
            session_id: str,
            current_user: Optional[str] = Depends(get_current_user)
        ) -> SessionResponse:
            """Get session information."""
            try:
                if not self.session_manager:
                    raise HTTPException(status_code=503, detail="Session manager not available")
                
                session = await self.session_manager.get_session(session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                # Check user authorization
                if session.context.user_id != current_user and current_user != "admin":
                    raise HTTPException(status_code=403, detail="Access denied")
                
                return SessionResponse(
                    session_id=session_id,
                    user_id=session.context.user_id,
                    state=session.state.value,
                    created_at=session.created_at.isoformat(),
                    interaction_count=session.interaction_count,
                    memory_item_count=session.memory_item_count
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Get session error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")
        
        @self.router.delete("/sessions/{session_id}")
        async def end_session(
            session_id: str,
            current_user: Optional[str] = Depends(get_current_user)
        ) -> Dict[str, str]:
            """End a session."""
            try:
                if not self.session_manager:
                    raise HTTPException(status_code=503, detail="Session manager not available")
                
                session = await self.session_manager.get_session(session_id)
                if not session:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                # Check user authorization
                if session.context.user_id != current_user and current_user != "admin":
                    raise HTTPException(status_code=403, detail="Access denied")
                
                success = await self.session_manager.end_session(session_id, reason="user_request")
                
                if success:
                    return {"message": "Session ended successfully", "session_id": session_id}
                else:
                    raise HTTPException(status_code=500, detail="Failed to end session")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"End session error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")
        
        @self.router.post("/memory/query", response_model=MemoryResponse)
        async def query_memory(
            request: MemoryQueryRequest,
            current_user: Optional[str] = Depends(get_current_user)
        ) -> MemoryResponse:
            """Query memory with semantic search."""
            try:
                start_time = datetime.now(timezone.utc)
                
                if not self.memory_integrator:
                    raise HTTPException(status_code=503, detail="Memory system not available")
                
                # Check session authorization if session_id is provided
                if request.session_id:
                    if not self.session_manager:
                        raise HTTPException(status_code=503, detail="Session manager not available")
                    
                    session = await self.session_manager.get_session(request.session_id)
                    if not session:
                        raise HTTPException(status_code=404, detail="Session not found")
                    
                    if session.context.user_id != current_user and current_user != "admin":
                        raise HTTPException(status_code=403, detail="Access denied")
                
                # Query memories
                if request.session_id:
                    memories = await self.memory_integrator.retrieve_session_memories(
                        session_id=request.session_id,
                        query=request.query,
                        limit=request.limit
                    )
                else:
                    # Implement general memory search across user's memories
                    if not current_user:
                        raise HTTPException(status_code=401, detail="Authentication required for memory search")
                    
                    # Get all user sessions to search across their memories
                    user_sessions = await self.session_manager.list_user_sessions(current_user)
                    
                    memories = []
                    for session_id in user_sessions[:10]:  # Limit to recent 10 sessions for performance
                        try:
                            session_memories = await self.memory_integrator.retrieve_session_memories(
                                session_id=session_id,
                                query=request.query,
                                limit=min(request.limit // len(user_sessions) + 1, 5)  # Distribute limit across sessions
                            )
                            memories.extend(session_memories)
                        except Exception as e:
                            self.logger.warning(f"Failed to search memories in session {session_id}: {e}")
                            continue
                    
                    # Sort by relevance and limit results
                    memories.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
                    memories = memories[:request.limit]
                
                query_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                memory_items = [
                    MemoryItem(
                        memory_id=mem.get("memory_id", ""),
                        content=mem.get("content", ""),
                        memory_type=mem.get("type", "unknown"),
                        relevance=mem.get("relevance", 0.0),
                        timestamp=mem.get("timestamp", "")
                    )
                    for mem in memories
                ]
                
                return MemoryResponse(
                    memories=memory_items,
                    total_count=len(memory_items),
                    query_time=query_time
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Memory query error: {e}")
                raise HTTPException(status_code=500, detail=f"Memory query failed: {str(e)}")
        
        @self.router.post("/memory/store")
        async def store_memory_fact(
            fact: str = Body(..., description="Fact to store"),
            session_id: str = Body(..., description="Session ID"),
            importance: float = Body(0.7, description="Importance score"),
            tags: List[str] = Body(default_factory=list, description="Tags"),
            current_user: Optional[str] = Depends(get_current_user)
        ) -> Dict[str, str]:
            """Store an important fact in memory."""
            try:
                if not self.memory_integrator:
                    raise HTTPException(status_code=503, detail="Memory system not available")
                
                # Check session authorization
                if self.session_manager:
                    session = await self.session_manager.get_session(session_id)
                    if not session:
                        raise HTTPException(status_code=404, detail="Session not found")
                    
                    if session.context.user_id != current_user and current_user != "admin":
                        raise HTTPException(status_code=403, detail="Access denied")
                
                memory_id = await self.memory_integrator.store_session_fact(
                    session_id=session_id,
                    user_id=current_user,
                    fact=fact,
                    importance=importance,
                    tags=set(tags)
                )
                
                if memory_id:
                    return {"message": "Fact stored successfully", "memory_id": memory_id}
                else:
                    raise HTTPException(status_code=500, detail="Failed to store fact")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Store memory error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")
        
        @self.router.get("/health")
        async def health_check() -> Dict[str, Any]:
            """Health check with component status."""
            try:
                status = {
                    "status": "healthy",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "components": {
                        "session_manager": self.session_manager is not None,
                        "memory_integrator": self.memory_integrator is not None,
                        "core_engine": self.core_engine is not None
                    }
                }
                
                # Get detailed component status
                if self.session_manager:
                    session_stats = self.session_manager.get_session_statistics()
                    status["session_stats"] = session_stats
                
                if self.memory_integrator:
                    memory_stats = self.memory_integrator.get_session_statistics()
                    status["memory_stats"] = memory_stats
                
                return status
                
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }


def create_api_endpoints(container: Container) -> APIEndpoints:
    """Create API endpoints instance."""
    return APIEndpoints(container)