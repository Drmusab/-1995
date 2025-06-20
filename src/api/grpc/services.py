"""
Comprehensive gRPC Services for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 17:30:00 UTC

This module provides comprehensive gRPC service implementations that integrate
with all core system components including the core engine, interaction handler,
workflow orchestrator, session manager, component manager, plugin manager,
memory systems, and learning components.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import asdict

import grpc
from grpc import aio
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager

# Assistant components
from src.assistant.core_engine import (
    EnhancedCoreEngine, MultimodalInput, ProcessingContext, ProcessingResult,
    EngineState, ProcessingMode, ModalityType, PriorityLevel
)
from src.assistant.interaction_handler import (
    InteractionHandler, UserMessage, AssistantResponse, InteractionContext,
    InteractionState, InteractionMode, InputModality, OutputModality
)
from src.assistant.workflow_orchestrator import (
    WorkflowOrchestrator, WorkflowDefinition, WorkflowExecution,
    WorkflowState, WorkflowPriority
)
from src.assistant.session_manager import (
    EnhancedSessionManager, SessionInfo, SessionState, SessionConfiguration
)
from src.assistant.component_manager import (
    EnhancedComponentManager, ComponentInfo, ComponentState
)
from src.assistant.plugin_manager import (
    EnhancedPluginManager, PluginInfo, PluginState
)

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory

# Learning systems
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner
from src.learning.feedback_processor import FeedbackProcessor

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# gRPC imports - These would be generated from .proto files
try:
    from src.api.grpc.protos import (
        assistant_pb2,
        assistant_pb2_grpc,
        health_pb2,
        health_pb2_grpc,
        metrics_pb2,
        metrics_pb2_grpc
    )
except ImportError:
    # Fallback for when proto files are not generated yet
    class MockProto:
        class Empty:
            pass
        class HealthCheckRequest:
            def __init__(self, service=""):
                self.service = service
        class HealthCheckResponse:
            UNKNOWN = 0
            SERVING = 1
            NOT_SERVING = 2
            SERVICE_UNKNOWN = 3
            def __init__(self, status=1):
                self.status = status
    
    assistant_pb2 = MockProto()
    assistant_pb2_grpc = MockProto()
    health_pb2 = MockProto()
    health_pb2_grpc = MockProto()
    metrics_pb2 = MockProto()
    metrics_pb2_grpc = MockProto()


class GrpcServiceError(Exception):
    """Custom exception for gRPC service operations."""
    
    def __init__(self, message: str, code: grpc.StatusCode = grpc.StatusCode.INTERNAL,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class AuthenticationInterceptor(aio.ServerInterceptor):
    """gRPC interceptor for authentication."""
    
    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.logger = get_logger(__name__)
        self.exempt_methods = {
            '/grpc.health.v1.Health/Check',
            '/grpc.health.v1.Health/Watch',
            '/ai_assistant.v1.AuthService/Login',
            '/ai_assistant.v1.AuthService/Register'
        }
    
    async def intercept_service(self, continuation, handler_call_details):
        """Intercept and authenticate gRPC calls."""
        method_name = handler_call_details.method
        
        # Skip authentication for exempt methods
        if method_name in self.exempt_methods:
            return await continuation(handler_call_details)
        
        # Extract token from metadata
        metadata = dict(handler_call_details.invocation_metadata)
        auth_header = metadata.get('authorization', '')
        
        if not auth_header.startswith('Bearer '):
            await handler_call_details.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                'Missing or invalid authorization header'
            )
            return
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            # Validate token
            user_info = await self.auth_manager.validate_token(token)
            if not user_info:
                await handler_call_details.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    'Invalid token'
                )
                return
            
            # Add user info to context
            handler_call_details.user_info = user_info
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            await handler_call_details.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                'Authentication failed'
            )
            return
        
        return await continuation(handler_call_details)


class RateLimitInterceptor(aio.ServerInterceptor):
    """gRPC interceptor for rate limiting."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.request_counts: Dict[str, List[float]] = {}
        self.logger = get_logger(__name__)
    
    async def intercept_service(self, continuation, handler_call_details):
        """Intercept and rate limit gRPC calls."""
        # Get client IP or user ID
        metadata = dict(handler_call_details.invocation_metadata)
        client_id = metadata.get('x-forwarded-for', 'unknown')
        
        if hasattr(handler_call_details, 'user_info'):
            client_id = handler_call_details.user_info.get('user_id', client_id)
        
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if client_id in self.request_counts:
            self.request_counts[client_id] = [
                req_time for req_time in self.request_counts[client_id]
                if current_time - req_time < 60
            ]
        else:
            self.request_counts[client_id] = []
        
        # Check rate limit
        if len(self.request_counts[client_id]) >= self.max_requests:
            await handler_call_details.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f'Rate limit exceeded: {self.max_requests} requests per minute'
            )
            return
        
        # Add current request
        self.request_counts[client_id].append(current_time)
        
        return await continuation(handler_call_details)


class CoreEngineService(assistant_pb2_grpc.CoreEngineServiceServicer):
    """gRPC service for core engine operations."""
    
    def __init__(self, container: Container):
        self.container = container
        self.core_engine = container.get(EnhancedCoreEngine)
        self.logger = get_logger(__name__)
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
    
    @handle_exceptions
    async def ProcessMultimodalInput(self, request, context):
        """Process multimodal input through the core engine."""
        try:
            with self.tracer.trace("grpc_process_multimodal_input") as span:
                # Extract user info from context
                user_id = getattr(context, 'user_info', {}).get('user_id')
                
                # Convert request to MultimodalInput
                multimodal_input = MultimodalInput(
                    text=request.text if request.text else None,
                    audio=np.frombuffer(request.audio, dtype=np.float32) if request.audio else None,
                    image=np.frombuffer(request.image, dtype=np.uint8) if request.image else None,
                    context=ProcessingContext(
                        session_id=request.session_id,
                        user_id=user_id,
                        priority=PriorityLevel(request.priority) if request.priority else PriorityLevel.NORMAL
                    )
                )
                
                # Process through core engine
                result = await self.core_engine.process_multimodal_input(
                    multimodal_input,
                    multimodal_input.context
                )
                
                # Convert result to response
                response = assistant_pb2.ProcessingResponse(
                    success=result.success,
                    request_id=result.request_id,
                    session_id=result.session_id,
                    processing_time=result.processing_time,
                    response_text=result.response_text or "",
                    confidence=result.overall_confidence,
                    errors=result.errors,
                    warnings=result.warnings
                )
                
                # Add audio data if available
                if result.synthesized_audio is not None:
                    response.synthesized_audio = result.synthesized_audio.tobytes()
                
                # Update metrics
                self.metrics.increment("grpc_multimodal_requests_total")
                self.metrics.record("grpc_processing_duration_seconds", result.processing_time)
                
                return response
                
        except Exception as e:
            self.logger.error(f"Error processing multimodal input: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def StreamProcessing(self, request_iterator, context):
        """Stream processing for real-time input."""
        try:
            async for request in request_iterator:
                # Process each request
                result = await self.ProcessMultimodalInput(request, context)
                yield result
                
        except Exception as e:
            self.logger.error(f"Error in stream processing: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetEngineStatus(self, request, context):
        """Get current engine status."""
        try:
            status = await self.core_engine.get_engine_status()
            
            response = assistant_pb2.EngineStatusResponse(
                state=status['state'],
                uptime_seconds=status['uptime_seconds'],
                active_sessions=status['active_sessions'],
                component_count=status['component_count'],
                healthy_components=status['healthy_components'],
                memory_usage_mb=status.get('memory_usage', {}).get('rss_mb', 0),
                version=status['version']
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting engine status: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class InteractionService(assistant_pb2_grpc.InteractionServiceServicer):
    """gRPC service for user interaction management."""
    
    def __init__(self, container: Container):
        self.container = container
        self.interaction_handler = container.get(InteractionHandler)
        self.logger = get_logger(__name__)
        self.metrics = container.get(MetricsCollector)
    
    async def StartInteraction(self, request, context):
        """Start a new user interaction."""
        try:
            user_id = getattr(context, 'user_info', {}).get('user_id')
            
            interaction_id = await self.interaction_handler.start_interaction(
                user_id=user_id,
                session_id=request.session_id,
                interaction_mode=InteractionMode(request.interaction_mode),
                input_modalities={InputModality(m) for m in request.input_modalities},
                output_modalities={OutputModality(m) for m in request.output_modalities},
                device_info=json.loads(request.device_info) if request.device_info else {}
            )
            
            response = assistant_pb2.StartInteractionResponse(
                interaction_id=interaction_id,
                success=True
            )
            
            self.metrics.increment("grpc_interactions_started_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error starting interaction: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def ProcessMessage(self, request, context):
        """Process a user message."""
        try:
            user_id = getattr(context, 'user_info', {}).get('user_id')
            
            # Create user message
            message = UserMessage(
                message_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=request.session_id,
                interaction_id=request.interaction_id,
                text=request.text if request.text else None,
                modality=InputModality(request.modality),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Add audio/image data if present
            if request.audio_data:
                message.audio_data = np.frombuffer(request.audio_data, dtype=np.float32)
            if request.image_data:
                message.image_data = np.frombuffer(request.image_data, dtype=np.uint8)
            
            # Process message
            assistant_response = await self.interaction_handler.process_user_message(
                request.interaction_id,
                message,
                real_time=request.real_time,
                streaming=request.streaming
            )
            
            # Convert to gRPC response
            response = assistant_pb2.MessageResponse(
                response_id=assistant_response.response_id,
                interaction_id=assistant_response.interaction_id,
                text=assistant_response.text or "",
                processing_time=assistant_response.processing_time,
                confidence=assistant_response.confidence,
                modalities=[m.value for m in assistant_response.modalities],
                timestamp=assistant_response.timestamp.isoformat()
            )
            
            # Add audio data if available
            if assistant_response.audio_data is not None:
                response.audio_data = assistant_response.audio_data.tobytes()
            
            self.metrics.increment("grpc_messages_processed_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def EndInteraction(self, request, context):
        """End an interaction."""
        try:
            await self.interaction_handler.end_interaction(
                request.interaction_id,
                request.reason or "user_ended"
            )
            
            response = assistant_pb2.EndInteractionResponse(success=True)
            self.metrics.increment("grpc_interactions_ended_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error ending interaction: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetInteractionStatus(self, request, context):
        """Get interaction status."""
        try:
            status = self.interaction_handler.get_interaction_status(request.interaction_id)
            
            response = assistant_pb2.InteractionStatusResponse(
                interaction_id=status['interaction_id'],
                session_id=status['session_id'],
                user_id=status.get('user_id', ''),
                state=status['state'],
                interaction_mode=status['interaction_mode'],
                duration=status['duration'],
                message_count=status['message_count']
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting interaction status: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class WorkflowService(assistant_pb2_grpc.WorkflowServiceServicer):
    """gRPC service for workflow orchestration."""
    
    def __init__(self, container: Container):
        self.container = container
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.logger = get_logger(__name__)
        self.metrics = container.get(MetricsCollector)
    
    async def ExecuteWorkflow(self, request, context):
        """Execute a workflow."""
        try:
            user_id = getattr(context, 'user_info', {}).get('user_id')
            
            # Parse input data
            input_data = json.loads(request.input_data) if request.input_data else {}
            workflow_context = json.loads(request.context) if request.context else {}
            
            execution_id = await self.workflow_orchestrator.execute_workflow(
                workflow_id=request.workflow_id,
                session_id=request.session_id,
                input_data=input_data,
                user_id=user_id,
                priority=WorkflowPriority(request.priority) if request.priority else WorkflowPriority.NORMAL,
                context=workflow_context
            )
            
            response = assistant_pb2.ExecuteWorkflowResponse(
                execution_id=execution_id,
                success=True
            )
            
            self.metrics.increment("grpc_workflows_executed_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error executing workflow: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetWorkflowStatus(self, request, context):
        """Get workflow execution status."""
        try:
            status = await self.workflow_orchestrator.get_execution_status(request.execution_id)
            
            response = assistant_pb2.WorkflowStatusResponse(
                execution_id=status['execution_id'],
                workflow_id=status['workflow_id'],
                session_id=status['session_id'],
                state=status['state'],
                execution_time=status['execution_time'],
                progress=status['progress'],
                current_steps=status['current_steps'],
                completed_steps=status['completed_steps'],
                failed_steps=status['failed_steps'],
                errors=status['errors']
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def CancelWorkflow(self, request, context):
        """Cancel a workflow execution."""
        try:
            await self.workflow_orchestrator.cancel_execution(request.execution_id)
            
            response = assistant_pb2.CancelWorkflowResponse(success=True)
            self.metrics.increment("grpc_workflows_cancelled_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error cancelling workflow: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def ListWorkflows(self, request, context):
        """List available workflows."""
        try:
            workflows = self.workflow_orchestrator.list_workflows()
            
            workflow_list = [
                assistant_pb2.WorkflowInfo(
                    workflow_id=wf['workflow_id'],
                    name=wf['name'],
                    version=wf['version'],
                    description=wf.get('description', ''),
                    step_count=wf['step_count'],
                    execution_mode=wf['execution_mode']
                )
                for wf in workflows
            ]
            
            response = assistant_pb2.ListWorkflowsResponse(workflows=workflow_list)
            return response
            
        except Exception as e:
            self.logger.error(f"Error listing workflows: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class SessionService(assistant_pb2_grpc.SessionServiceServicer):
    """gRPC service for session management."""
    
    def __init__(self, container: Container):
        self.container = container
        self.session_manager = container.get(EnhancedSessionManager)
        self.logger = get_logger(__name__)
        self.metrics = container.get(MetricsCollector)
    
    async def CreateSession(self, request, context):
        """Create a new session."""
        try:
            user_id = getattr(context, 'user_info', {}).get('user_id')
            
            # Parse session configuration
            session_config = None
            if request.config:
                config_dict = json.loads(request.config)
                session_config = SessionConfiguration(**config_dict)
            
            # Parse context data
            context_data = json.loads(request.context_data) if request.context_data else None
            
            session_id = await self.session_manager.create_session(
                user_id=user_id,
                session_config=session_config,
                context_data=context_data
            )
            
            response = assistant_pb2.CreateSessionResponse(
                session_id=session_id,
                success=True
            )
            
            self.metrics.increment("grpc_sessions_created_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetSession(self, request, context):
        """Get session information."""
        try:
            session_info = await self.session_manager.get_session(request.session_id)
            
            if not session_info:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Session not found")
                return
            
            response = assistant_pb2.GetSessionResponse(
                session_id=session_info.session_id,
                user_id=session_info.context.user_id or "",
                state=session_info.state.value,
                created_at=session_info.created_at.isoformat(),
                last_activity=session_info.last_activity.isoformat(),
                interaction_count=session_info.interaction_count,
                memory_usage_mb=session_info.memory_usage_mb
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting session: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def EndSession(self, request, context):
        """End a session."""
        try:
            await self.session_manager.end_session(
                request.session_id,
                request.reason or "user_ended"
            )
            
            response = assistant_pb2.EndSessionResponse(success=True)
            self.metrics.increment("grpc_sessions_ended_total")
            return response
            
        except Exception as e:
            self.logger.error(f"Error ending session: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class ComponentService(assistant_pb2_grpc.ComponentServiceServicer):
    """gRPC service for component management."""
    
    def __init__(self, container: Container):
        self.container = container
        self.component_manager = container.get(EnhancedComponentManager)
        self.logger = get_logger(__name__)
    
    async def GetComponentStatus(self, request, context):
        """Get component status."""
        try:
            if request.component_id:
                status = self.component_manager.get_component_status(request.component_id)
            else:
                status = self.component_manager.get_component_status()
            
            response = assistant_pb2.ComponentStatusResponse(
                status=json.dumps(status)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting component status: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def RestartComponent(self, request, context):
        """Restart a component."""
        try:
            await self.component_manager.restart_component(request.component_id)
            
            response = assistant_pb2.RestartComponentResponse(success=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error restarting component: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class PluginService(assistant_pb2_grpc.PluginServiceServicer):
    """gRPC service for plugin management."""
    
    def __init__(self, container: Container):
        self.container = container
        self.plugin_manager = container.get(EnhancedPluginManager)
        self.logger = get_logger(__name__)
    
    async def ListPlugins(self, request, context):
        """List all plugins."""
        try:
            plugins = self.plugin_manager.list_plugins()
            
            plugin_list = [
                assistant_pb2.PluginInfo(
                    plugin_id=plugin['plugin_id'],
                    name=plugin['name'],
                    version=plugin['version'],
                    type=plugin['type'],
                    state=plugin['state'],
                    description=plugin['description']
                )
                for plugin in plugins
            ]
            
            response = assistant_pb2.ListPluginsResponse(plugins=plugin_list)
            return response
            
        except Exception as e:
            self.logger.error(f"Error listing plugins: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def LoadPlugin(self, request, context):
        """Load a plugin."""
        try:
            await self.plugin_manager.load_plugin(request.plugin_id)
            
            response = assistant_pb2.LoadPluginResponse(success=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error loading plugin: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def UnloadPlugin(self, request, context):
        """Unload a plugin."""
        try:
            await self.plugin_manager.unload_plugin(request.plugin_id)
            
            response = assistant_pb2.UnloadPluginResponse(success=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error unloading plugin: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class MemoryService(assistant_pb2_grpc.MemoryServiceServicer):
    """gRPC service for memory operations."""
    
    def __init__(self, container: Container):
        self.container = container
        self.memory_manager = container.get(MemoryManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        self.logger = get_logger(__name__)
    
    async def StoreMemory(self, request, context):
        """Store a memory."""
        try:
            data = json.loads(request.data)
            
            if request.memory_type == "working":
                await self.working_memory.store(data)
            elif request.memory_type == "episodic":
                await self.episodic_memory.store(data)
            elif request.memory_type == "semantic":
                await self.semantic_memory.store(data)
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid memory type")
                return
            
            response = assistant_pb2.StoreMemoryResponse(success=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def RetrieveMemory(self, request, context):
        """Retrieve memories."""
        try:
            if request.memory_type == "working":
                memories = await self.working_memory.retrieve(request.query, limit=request.limit)
            elif request.memory_type == "episodic":
                memories = await self.episodic_memory.retrieve(request.query, limit=request.limit)
            elif request.memory_type == "semantic":
                memories = await self.semantic_memory.retrieve(request.query, limit=request.limit)
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid memory type")
                return
            
            response = assistant_pb2.RetrieveMemoryResponse(
                memories=[json.dumps(memory) for memory in memories]
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class LearningService(assistant_pb2_grpc.LearningServiceServicer):
    """gRPC service for learning and adaptation."""
    
    def __init__(self, container: Container):
        self.container = container
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)
        self.logger = get_logger(__name__)
    
    async def ProcessFeedback(self, request, context):
        """Process user feedback."""
        try:
            user_id = getattr(context, 'user_info', {}).get('user_id')
            feedback_data = json.loads(request.feedback_data)
            
            await self.feedback_processor.process_feedback(
                interaction_id=request.interaction_id,
                feedback_type=request.feedback_type,
                feedback_data=feedback_data,
                user_id=user_id
            )
            
            response = assistant_pb2.ProcessFeedbackResponse(success=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
    
    async def GetUserPreferences(self, request, context):
        """Get user preferences."""
        try:
            user_id = getattr(context, 'user_info', {}).get('user_id')
            if not user_id:
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "User not authenticated")
                return
            
            preferences = await self.preference_learner.get_user_preferences(user_id)
            
            response = assistant_pb2.GetUserPreferencesResponse(
                preferences=json.dumps(preferences or {})
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class HealthService(health_pb2_grpc.HealthServicer):
    """gRPC health check service."""
    
    def __init__(self, container: Container):
        self.container = container
        self.health_check = container.get(HealthCheck)
        self.logger = get_logger(__name__)
    
    async def Check(self, request, context):
        """Perform health check."""
        try:
            service_name = request.service
            
            if service_name:
                # Check specific service
                health_status = await self.health_check.check_component(service_name)
                is_healthy = health_status.get('status') == 'healthy'
            else:
                # Check overall system health
                health_status = await self.health_check.get_system_health()
                is_healthy = health_status.get('overall_status') == 'healthy'
            
            status = (health_pb2.HealthCheckResponse.SERVING if is_healthy 
                     else health_pb2.HealthCheckResponse.NOT_SERVING)
            
            return health_pb2.HealthCheckResponse(status=status)
            
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}")
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )
    
    async def Watch(self, request, context):
        """Watch health status changes."""
        try:
            service_name = request.service
            
            while True:
                # Get current health status
                if service_name:
                    health_status = await self.health_check.check_component(service_name)
                    is_healthy = health_status.get('status') == 'healthy'
                else:
                    health_status = await self.health_check.get_system_health()
                    is_healthy = health_status.get('overall_status') == 'healthy'
                
                status = (health_pb2.HealthCheckResponse.SERVING if is_healthy 
                         else health_pb2.HealthCheckResponse.NOT_SERVING)
                
                yield health_pb2.HealthCheckResponse(status=status)
                
                # Wait before next check
                await asyncio.sleep(30)
                
        except Exception as e:
            self.logger.error(f"Health watch error: {str(e)}")


class MetricsService(metrics_pb2_grpc.MetricsServiceServicer):
    """gRPC service for metrics and monitoring."""
    
    def __init__(self, container: Container):
        self.container = container
        self.metrics = container.get(MetricsCollector)
        self.logger = get_logger(__name__)
    
    async def GetMetrics(self, request, context):
        """Get system metrics."""
        try:
            metrics_data = await self.metrics.get_all_metrics()
            
            response = metrics_pb2.GetMetricsResponse(
                metrics=json.dumps(metrics_data)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


class GrpcServer:
    """Main gRPC server class."""
    
    def __init__(self, container: Container, host: str = "0.0.0.0", port: int = 50051):
        self.container = container
        self.host = host
        self.port = port
        self.logger = get_logger(__name__)
        self.server = None
        
        # Get configuration
        self.config = container.get(ConfigLoader)
        self.auth_manager = container.get(AuthenticationManager)
        
        # Configure server options
        self.server_options = [
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000)
        ]
    
    async def start(self) -> None:
        """Start the gRPC server."""
        try:
            # Create server with interceptors
            interceptors = [
                AuthenticationInterceptor(self.auth_manager),
                RateLimitInterceptor(max_requests_per_minute=120)
            ]
            
            self.server = aio.server(
                interceptors=interceptors,
                options=self.server_options
            )
            
            # Add services
            assistant_pb2_grpc.add_CoreEngineServiceServicer_to_server(
                CoreEngineService(self.container), self.server
            )
            assistant_pb2_grpc.add_InteractionServiceServicer_to_server(
                InteractionService(self.container), self.server
            )
            assistant_pb2_grpc.add_WorkflowServiceServicer_to_server(
                WorkflowService(self.container), self.server
            )
            assistant_pb2_grpc.add_SessionServiceServicer_to_server(
                SessionService(self.container), self.server
            )
            assistant_pb2_grpc.add_ComponentServiceServicer_to_server(
                ComponentService(self.container), self.server
            )
            assistant_pb2_grpc.add_PluginServiceServicer_to_server(
                PluginService(self.container), self.server
            )
            assistant_pb2_grpc.add_MemoryServiceServicer_to_server(
                MemoryService(self.container), self.server
            )
            assistant_pb2_grpc.add_LearningServiceServicer_to_server(
                LearningService(self.container), self.server
            )
            
            # Add health service
            health_pb2_grpc.add_HealthServicer_to_server(
                HealthService(self.container), self.server
            )
            
            # Add metrics service
            metrics_pb2_grpc.add_MetricsServiceServicer_to_server(
                MetricsService(self.container), self.server
            )
            
            # Configure SSL if enabled
            server_credentials = None
            if self.config.get("grpc.ssl.enabled", False):
                with open(self.config.get("grpc.ssl.cert_file"), 'rb') as f:
                    cert_chain = f.read()
                with open(self.config.get("grpc.ssl.key_file"), 'rb') as f:
                    private_key = f.read()
                
                server_credentials = grpc.ssl_server_credentials(
                    [(private_key, cert_chain)]
                )
            
            # Add port
            if server_credentials:
                listen_addr = f'{self.host}:{self.port}'
                self.server.add_secure_port(listen_addr, server_credentials)
                self.logger.info(f"Starting secure gRPC server on {listen_addr}")
            else:
                listen_addr = f'{self.host}:{self.port}'
                self.server.add_insecure_port(listen_addr)
                self.logger.info(f"Starting insecure gRPC server on {listen_addr}")
            
            # Start server
            await self.server.start()
            self.logger.info("gRPC server started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start gRPC server: {str(e)}")
            raise
    
    async def stop(self, grace_period: float = 30.0) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            self.logger.info("Stopping gRPC server...")
            await self.server.stop(grace_period)
            self.logger.info("gRPC server stopped")
    
    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()


# Utility functions for creating and running the server
async def create_grpc_server(container: Container, host: str = "0.0.0.0", port: int = 50051) -> GrpcServer:
    """Create and configure a gRPC server."""
    server = GrpcServer(container, host, port)
    await server.start()
    return server


async def run_grpc_server(container: Container, host: str = "0.0.0.0", port: int = 50051) -> None:
    """Run the gRPC server until terminated."""
    server = await create_grpc_server(container, host, port)
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    # Example of running the server standalone
    from src.core.dependency_injection import Container
    
    async def main():
        # Create container and register dependencies
        container = Container()
        # ... register all dependencies ...
        
        # Run server
        await run_grpc_server(container)
    
    asyncio.run(main())
