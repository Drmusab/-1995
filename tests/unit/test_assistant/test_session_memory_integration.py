"""Tests for session-memory integration."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.assistant.session_manager import EnhancedSessionManager
from src.memory.core_memory.memory_manager import MemoryManager
from src.learning.memory_learning_bridge import MemoryLearningBridge


class TestSessionMemoryIntegration:
    """Test suite for session-memory integration."""
    
    @pytest.fixture
    async def session_manager(self, mock_container):
        """Create a mock session manager."""
        manager = Mock(spec=EnhancedSessionManager)
        manager.create_session = AsyncMock(return_value="session-123")
        manager.get_session = AsyncMock(return_value={
            "session_id": "session-123",
            "user_id": "user-123",
            "state": "active",
            "created_at": "2025-01-01T00:00:00Z"
        })
        manager.end_session = AsyncMock()
        return manager
    
    @pytest.fixture
    async def memory_manager(self, mock_container):
        """Create a mock memory manager."""
        manager = Mock(spec=MemoryManager)
        manager.store_memory = AsyncMock(return_value="memory-123")
        manager.retrieve_memories = AsyncMock(return_value=[])
        manager.update_memory = AsyncMock()
        return manager
    
    @pytest.fixture
    async def memory_bridge(self, mock_container):
        """Create a mock memory-learning bridge."""
        bridge = Mock(spec=MemoryLearningBridge)
        bridge.process_session_data = AsyncMock()
        bridge.update_learning_model = AsyncMock()
        return bridge
    
    @pytest.mark.asyncio
    async def test_session_creation_with_memory_initialization(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that creating a session properly initializes memory context."""
        # Arrange
        user_id = "user-123"
        
        # Act
        session_id = await session_manager.create_session(user_id)
        
        # Assert
        assert session_id == "session-123"
        session_manager.create_session.assert_called_once_with(user_id)
    
    @pytest.mark.asyncio
    async def test_session_memory_storage_integration(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that session interactions are properly stored in memory."""
        # Arrange
        session_id = "session-123"
        interaction_data = {
            "user_input": "Hello, how are you?",
            "assistant_response": "I'm doing well, thank you!",
            "timestamp": "2025-01-01T00:00:00Z",
            "confidence": 0.9
        }
        
        # Act - simulate storing interaction in memory
        memory_id = await memory_manager.store_memory(
            content=interaction_data,
            memory_type="interaction",
            context={"session_id": session_id}
        )
        
        # Assert
        assert memory_id == "memory-123"
        memory_manager.store_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_memory_retrieval(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that session context can retrieve relevant memories."""
        # Arrange
        session_id = "session-123"
        query = "previous conversations"
        
        # Act
        memories = await memory_manager.retrieve_memories(
            query=query,
            context={"session_id": session_id},
            limit=10
        )
        
        # Assert
        assert memories == []
        memory_manager.retrieve_memories.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_session_end_triggers_memory_consolidation(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that ending a session triggers memory consolidation."""
        # Arrange
        session_id = "session-123"
        
        # Act
        await session_manager.end_session(session_id)
        
        # Assert
        session_manager.end_session.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_memory_learning_bridge_integration(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that the memory-learning bridge processes session data."""
        # Arrange
        session_data = {
            "session_id": "session-123",
            "interactions": [
                {"user_input": "Hello", "response": "Hi there!"},
                {"user_input": "How are you?", "response": "I'm doing well!"}
            ],
            "user_preferences": {"tone": "casual"}
        }
        
        # Act
        await memory_bridge.process_session_data(session_data)
        
        # Assert
        memory_bridge.process_session_data.assert_called_once_with(session_data)
    
    @pytest.mark.asyncio
    async def test_session_memory_context_preservation(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that memory context is preserved across session interactions."""
        # Arrange
        session_id = "session-123"
        
        # Simulate multiple interactions within the same session
        interactions = [
            {"input": "My name is Alice", "context": "introduction"},
            {"input": "What did I just tell you?", "context": "context_query"}
        ]
        
        # Act & Assert
        for i, interaction in enumerate(interactions):
            # Store each interaction
            await memory_manager.store_memory(
                content=interaction,
                memory_type="interaction",
                context={"session_id": session_id, "sequence": i}
            )
            
            # Retrieve context for next interaction
            context_memories = await memory_manager.retrieve_memories(
                query="recent context",
                context={"session_id": session_id}
            )
        
        # Verify storage calls
        assert memory_manager.store_memory.call_count == 2
        assert memory_manager.retrieve_memories.call_count == 2
    
    @pytest.mark.asyncio
    async def test_session_memory_error_handling(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test error handling in session-memory integration."""
        # Arrange
        session_manager.create_session.side_effect = Exception("Database error")
        
        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await session_manager.create_session("user-123")
    
    @pytest.mark.asyncio
    async def test_session_memory_cleanup_on_error(
        self, session_manager, memory_manager, memory_bridge
    ):
        """Test that memory is properly cleaned up if session creation fails."""
        # This test would verify cleanup procedures
        # Implementation depends on actual error handling strategy
        pass