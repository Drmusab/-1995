"""Test configuration and fixtures."""

import pytest
import asyncio
from typing import Generator
from unittest.mock import Mock

from src.main import AIAssistant
from src.core.dependency_injection import Container


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_assistant():
    """Create a mock AI Assistant for testing."""
    assistant = Mock(spec=AIAssistant)
    assistant.get_status.return_value = {
        "status": "running",
        "version": "1.0.0",
        "uptime_seconds": 0,
        "components_status": "healthy",
        "api_servers_running": 0
    }
    assistant.process_text_input.return_value = {
        "response_text": "Test response",
        "response_id": "test-123",
        "session_id": "session-123",
        "interaction_id": "interaction-123",
        "processing_time": 0.1,
        "confidence": 0.9,
        "suggested_follow_ups": []
    }
    return assistant


@pytest.fixture
def mock_container():
    """Create a mock dependency injection container."""
    return Mock(spec=Container)