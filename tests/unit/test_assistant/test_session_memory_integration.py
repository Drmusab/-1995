"""Tests for session-memory integration."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import uuid

from src.assistant.core import EnhancedSessionManager, SessionConfiguration, SessionType
from src.assistant.session_memory_integrator import SessionMemoryIntegrator, SessionMemoryContext
from src.core.dependency_injection import Container
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import SessionStarted, SessionEnded, MessageProcessed


@pytest.fixture
def mock_container():
    """Create a mock dependency injection container."""
    container = Mock(spec=Container)
    
    # Mock config loader
    config_loader = Mock(spec=ConfigLoader)
    config_loader.get.return_value = {
        'memory.integration': {
            'enable_episodic': True,
            'enable_semantic': True,
            'enable_working': True,
            'max_context_items': 10,
            'memory_retention_days': 30
        },
        'session_manager': {
            'storage_type': 'memory',
            'enable_memory': True
        }
    }
    
    # Mock event bus
    event_bus = Mock(spec=EventBus)
    event_bus.emit = AsyncMock()
    event_bus.subscribe = Mock()
    
    container.get.side_effect = lambda cls: {
        ConfigLoader: config_loader,
        EventBus: event_bus
    }.get(cls, Mock())
    
    return container


@pytest.fixture
async def session_memory_integrator(mock_container):
    """Create a session memory integrator for testing."""
    integrator = SessionMemoryIntegrator(mock_container)
    
    # Mock memory components
    integrator.memory_manager = Mock()
    integrator.working_memory = Mock()
    integrator.episodic_memory = Mock()
    integrator.semantic_memory = Mock()
    
    # Setup async mocks for memory operations
    integrator.episodic_memory.store = AsyncMock(return_value="memory-123")
    integrator.episodic_memory.retrieve_by_session = AsyncMock(return_value=[
        {'id': 'mem1', 'content': 'previous conversation', 'timestamp': '2025-01-01T00:00:00Z'}
    ])
    integrator.episodic_memory.retrieve_by_user = AsyncMock(return_value=[])
    
    integrator.semantic_memory.store = AsyncMock(return_value="semantic-123")
    integrator.semantic_memory.retrieve_by_user = AsyncMock(return_value=[])
    
    await integrator.initialize()
    return integrator


@pytest.fixture
async def session_manager(mock_container):
    """Create a session manager for testing."""
    manager = EnhancedSessionManager(mock_container)
    manager.memory_integrator = Mock()
    manager.memory_integrator.retrieve_session_memories = AsyncMock(return_value=[])
    await manager.initialize()
    return manager


class TestSessionMemoryIntegration:
    """Test suite for session-memory integration."""
    
    @pytest.mark.asyncio
    async def test_session_memory_integrator_initialization(self, session_memory_integrator):
        """Test that session memory integrator initializes correctly."""
        integrator = session_memory_integrator
        
        assert integrator is not None
        assert integrator.enable_episodic is True
        assert integrator.enable_semantic is True
        assert integrator.enable_working is True
        assert integrator.max_context_items == 10
    
    @pytest.mark.asyncio
    async def test_session_started_event_handling(self, session_memory_integrator):
        """Test handling of session started events."""
        integrator = session_memory_integrator
        
        # Create session started event
        event = SessionStarted(
            session_id="test-session-123",
            user_id="user-456",
            session_type="interactive"
        )
        
        # Handle the event
        await integrator._handle_session_started(event)
        
        # Verify session context was created
        assert "test-session-123" in integrator._session_contexts
        context = integrator._session_contexts["test-session-123"]
        assert context.session_id == "test-session-123"
        assert context.user_id == "user-456"
        assert isinstance(context.memories, list)
    
    @pytest.mark.asyncio
    async def test_session_ended_memory_consolidation(self, session_memory_integrator):
        """Test memory consolidation when session ends."""
        integrator = session_memory_integrator
        
        # Create session context with facts
        context = SessionMemoryContext(
            session_id="test-session-123",
            user_id="user-456",
            facts=[
                {"fact": "User likes coffee", "importance": 0.8}
            ]
        )
        integrator._session_contexts["test-session-123"] = context
        
        # Create session ended event
        event = SessionEnded(
            session_id="test-session-123",
            user_id="user-456",
            duration=300.0,
            summary={"interactions": 5}
        )
        
        # Handle the event
        await integrator._handle_session_ended(event)
        
        # Verify session was cleaned up
        assert "test-session-123" not in integrator._session_contexts
        
        # Verify semantic memory store was called for consolidation
        integrator.semantic_memory.store.assert_called()
    
    @pytest.mark.asyncio
    async def test_message_processing_and_storage(self, session_memory_integrator):
        """Test that processed messages are stored in episodic memory."""
        integrator = session_memory_integrator
        
        # Create session context
        context = SessionMemoryContext(session_id="test-session-123", user_id="user-456")
        integrator._session_contexts["test-session-123"] = context
        
        # Create message processed event
        event = MessageProcessed(
            session_id="test-session-123",
            user_id="user-456",
            message="Hello, how are you?",
            response="I'm doing well, thank you!",
            metadata={"confidence": 0.9}
        )
        
        # Handle the event
        await integrator._handle_message_processed(event)
        
        # Verify episodic memory store was called
        integrator.episodic_memory.store.assert_called_once()
        
        # Verify context was updated
        assert context.memory_storage_count == 1
        assert len(context.context_elements) > 0
        assert context.context_elements[-1]['type'] == 'assistant_response'
    
    @pytest.mark.asyncio
    async def test_session_context_retrieval(self, session_memory_integrator):
        """Test retrieval of session memory context."""
        integrator = session_memory_integrator
        
        # Create session context with data
        context = SessionMemoryContext(
            session_id="test-session-123",
            user_id="user-456",
            memories=[{"type": "episodic", "content": "previous chat"}],
            facts=[{"fact": "User prefers formal tone", "importance": 0.7}]
        )
        integrator._session_contexts["test-session-123"] = context
        
        # Get session context
        result = await integrator.get_session_context("test-session-123")
        
        # Verify context data
        assert result['session_id'] == "test-session-123"
        assert result['user_id'] == "user-456"
        assert len(result['memories']) == 1
        assert len(result['facts']) == 1
        assert 'stats' in result
        assert result['stats']['retrieval_count'] == 1
    
    @pytest.mark.asyncio
    async def test_session_memory_retrieval(self, session_memory_integrator):
        """Test retrieval of memories for a session."""
        integrator = session_memory_integrator
        
        # Create session context
        context = SessionMemoryContext(session_id="test-session-123")
        integrator._session_contexts["test-session-123"] = context
        
        # Retrieve session memories
        memories = await integrator.retrieve_session_memories(
            session_id="test-session-123",
            limit=5
        )
        
        # Verify memories were retrieved
        assert isinstance(memories, list)
        assert len(memories) == 1  # Based on mock setup
        assert memories[0]['type'] == 'episodic'
        assert 'memory_id' in memories[0]
        
        # Verify episodic memory was called
        integrator.episodic_memory.retrieve_by_session.assert_called_once_with(
            session_id="test-session-123",
            limit=5
        )
    
    @pytest.mark.asyncio
    async def test_session_fact_storage(self, session_memory_integrator):
        """Test storage of important facts from sessions."""
        integrator = session_memory_integrator
        
        # Create session context
        context = SessionMemoryContext(session_id="test-session-123", user_id="user-456")
        integrator._session_contexts["test-session-123"] = context
        
        # Store a fact
        memory_id = await integrator.store_session_fact(
            session_id="test-session-123",
            user_id="user-456",
            fact="User is a software engineer",
            importance=0.8,
            tags={"profession", "work"}
        )
        
        # Verify fact was stored
        assert memory_id == "semantic-123"  # Based on mock setup
        integrator.semantic_memory.store.assert_called_once()
        
        # Verify session context was updated
        assert context.memory_storage_count == 1
        assert len(context.facts) == 1
        assert context.facts[0]['fact'] == "User is a software engineer"
    
    @pytest.mark.asyncio
    async def test_context_size_management(self, session_memory_integrator):
        """Test that context size is managed properly."""
        integrator = session_memory_integrator
        
        # Create session context
        context = SessionMemoryContext(session_id="test-session-123")
        integrator._session_contexts["test-session-123"] = context
        
        # Add more context elements than the limit
        for i in range(15):  # More than max_context_items (10)
            context.context_elements.append({
                'type': 'user_message',
                'content': f'Message {i}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Simulate message received event to trigger size management
        from src.core.events.event_types import MessageReceived
        event = MessageReceived(
            session_id="test-session-123",
            message="New message"
        )
        
        await integrator._handle_message_received(event)
        
        # Verify context size was trimmed
        assert len(context.context_elements) <= integrator.max_context_items
    
    @pytest.mark.asyncio 
    async def test_memory_integration_statistics(self, session_memory_integrator):
        """Test memory integration statistics."""
        integrator = session_memory_integrator
        
        # Create multiple session contexts
        for i in range(3):
            context = SessionMemoryContext(
                session_id=f"session-{i}",
                memories=[{"type": "episodic"}],
                facts=[{"fact": f"fact-{i}"}]
            )
            context.memory_retrieval_count = i + 1
            context.memory_storage_count = i * 2
            integrator._session_contexts[f"session-{i}"] = context
        
        # Get statistics
        stats = integrator.get_session_statistics()
        
        # Verify statistics
        assert stats['active_sessions'] == 3
        assert stats['total_memories'] == 3
        assert stats['total_facts'] == 3
        assert stats['total_retrievals'] == 6  # 1 + 2 + 3
        assert stats['total_storage_operations'] == 6  # 0 + 2 + 4
        assert stats['memory_types_enabled']['episodic'] is True
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, session_memory_integrator):
        """Test cleanup of session memory context."""
        integrator = session_memory_integrator
        
        # Create session context
        context = SessionMemoryContext(session_id="test-session-123")
        integrator._session_contexts["test-session-123"] = context
        
        # Cleanup session
        await integrator.cleanup_session("test-session-123")
        
        # Verify session was removed
        assert "test-session-123" not in integrator._session_contexts
    
    @pytest.mark.asyncio
    async def test_error_handling_in_memory_operations(self, session_memory_integrator):
        """Test error handling during memory operations."""
        integrator = session_memory_integrator
        
        # Make episodic memory fail
        integrator.episodic_memory.store.side_effect = Exception("Memory store failed")
        
        # Create session context
        context = SessionMemoryContext(session_id="test-session-123")
        integrator._session_contexts["test-session-123"] = context
        
        # Try to process a message (should not crash)
        event = MessageProcessed(
            session_id="test-session-123",
            user_id="user-456",
            message="Hello",
            response="Hi there!",
            metadata={}
        )
        
        # Should handle error gracefully
        await integrator._handle_message_processed(event)
        
        # Session context should still exist
        assert "test-session-123" in integrator._session_contexts
    
    @pytest.mark.asyncio
    async def test_session_manager_memory_integration(self, session_manager):
        """Test that session manager integrates with memory systems."""
        manager = session_manager
        
        # Create a session
        session_id = await manager.create_session(
            user_id="user-123",
            session_type=SessionType.INTERACTIVE
        )
        
        # Verify session was created
        assert session_id is not None
        session = await manager.get_session(session_id)
        assert session is not None
        assert session.context.user_id == "user-123"
    
    @pytest.mark.asyncio
    async def test_session_interaction_with_memory_context(self, session_manager):
        """Test session interactions preserve memory context."""
        manager = session_manager
        
        # Create session
        session_id = await manager.create_session(user_id="user-123")
        
        # Process multiple messages
        messages = [
            "My name is Alice",
            "I work as a teacher", 
            "What did I tell you about my job?"
        ]
        
        for msg in messages:
            await manager.add_interaction(
                session_id=session_id,
                interaction_type='user_message',
                data={'message': msg}
            )
        
        # Get session to verify context preservation
        session = await manager.get_session(session_id)
        assert len(session.context.interaction_history) == 3
        
        # Verify memory integration was called
        if manager.memory_integrator:
            assert manager.memory_integrator.retrieve_session_memories.called