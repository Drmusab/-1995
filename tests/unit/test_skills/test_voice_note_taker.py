"""
Unit tests for Voice Note Taker Skill
"""

import json
import os
import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.skills.custom.voice_note_taker import (
    VoiceNoteTakerSkill,
    VoiceNote,
    NoteLanguage,
    NoteCategory,
    AudioFormat
)


@pytest.fixture
def mock_container():
    """Create a mock container with required dependencies."""
    container = MagicMock()
    
    # Mock speech components
    container.get.return_value = AsyncMock()
    
    # Provide specific mocks for known components
    speech_to_text = AsyncMock()
    speech_to_text.transcribe = AsyncMock(return_value={
        "text": "Test note content",
        "detected_language": "en",
        "confidence": 0.9,
        "duration_seconds": 5.0
    })
    
    bilingual_processor = AsyncMock()
    bilingual_processor.process = AsyncMock(return_value={
        "response": "Processed text response"
    })
    bilingual_processor.detect_language = AsyncMock(return_value={
        "language": "en"
    })
    
    entity_extractor = AsyncMock()
    entity_extractor.extract_entities = AsyncMock(return_value={
        "entities": [
            {"text": "Meeting", "type": "EVENT"},
            {"text": "John", "type": "PERSON"}
        ]
    })
    
    sentiment_analyzer = AsyncMock()
    sentiment_analyzer.analyze = AsyncMock(return_value={
        "sentiment": "neutral"
    })
    
    # Configure container to return different mocks based on type
    def get_side_effect(component_type):
        if component_type == "SpeechToText":
            return speech_to_text
        elif component_type == "TextToSpeech":
            return AsyncMock()
        elif component_type.__name__ == "BilingualProcessor":
            return bilingual_processor
        elif component_type.__name__ == "EntityExtractor":
            return entity_extractor
        elif component_type.__name__ == "SentimentAnalyzer":
            return sentiment_analyzer
        return AsyncMock()
    
    container.get.side_effect = get_side_effect
    
    return container


@pytest.fixture
def skill(mock_container, tmp_path):
    """Create a VoiceNoteTakerSkill instance with mocked dependencies."""
    with patch('src.skills.custom.voice_note_taker.get_logger'):
        skill = VoiceNoteTakerSkill(mock_container)
        
        # Override paths for testing
        skill.notes_dir = tmp_path / "notes"
        skill.audio_dir = tmp_path / "audio"
        skill._ensure_directories()
        
        return skill


@pytest.mark.asyncio
async def test_create_note_from_text(skill):
    """Test creating a note from text input."""
    # Arrange
    params = {
        "text": "This is a test note for testing purposes.",
        "title": "Test Note",
        "language": "en"
    }
    
    # Act
    result = await skill._execute(**params)
    
    # Assert
    assert result.success is True
    assert "note_id" in result.result
    assert result.result["title"] == "Test Note"
    assert result.result["content"] == "This is a test note for testing purposes."
    assert result.result["language"] == "en"
    
    # Check that the note was saved
    note_path = skill.notes_dir / f"{result.result['note_id']}.json"
    assert note_path.exists()


@pytest.mark.asyncio
async def test_create_note_from_audio(skill):
    """Test creating a note from audio input."""
    # Arrange
    audio_data = np.zeros(1000)  # Mock audio data
    params = {
        "audio_data": audio_data,
        "language": "en",
        "save_audio": True
    }
    
    # Mock the save_audio function
    with patch('src.skills.custom.voice_note_taker.save_audio_file') as mock_save:
        # Act
        result = await skill._execute(**params)
        
        # Assert
        assert result.success is True
        assert mock_save.called
        assert result.result["audio_saved"] is True
        assert "note_id" in result.result


@pytest.mark.asyncio
async def test_retrieve_note(skill):
    """Test retrieving a note by ID."""
    # Arrange - First create a note
    create_params = {
        "text": "Note to be retrieved",
        "title": "Retrieval Test"
    }
    create_result = await skill._execute(**create_params)
    note_id = create_result.result["note_id"]
    
    # Act - Retrieve the note
    retrieve_params = {
        "action": "retrieve",
        "note_id": note_id
    }
    result = await skill._execute(**retrieve_params)
    
    # Assert
    assert result.success is True
    assert result.result["note_id"] == note_id
    assert result.result["title"] == "Retrieval Test"
    assert result.result["content"] == "Note to be retrieved"


@pytest.mark.asyncio
async def test_update_note(skill):
    """Test updating an existing note."""
    # Arrange - First create a note
    create_params = {
        "text": "Original content",
        "title": "Original Title"
    }
    create_result = await skill._execute(**create_params)
    note_id = create_result.result["note_id"]
    
    # Act - Update the note
    update_params = {
        "action": "update",
        "note_id": note_id,
        "title": "Updated Title",
        "text": "Updated content"
    }
    result = await skill._execute(**update_params)
    
    # Assert
    assert result.success is True
    assert result.result["title"] == "Updated Title"
    assert result.result["content"] == "Updated content"
    
    # Verify by retrieving
    retrieve_params = {
        "action": "retrieve",
        "note_id": note_id
    }
    retrieve_result = await skill._execute(**retrieve_params)
    assert retrieve_result.result["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_delete_note(skill):
    """Test deleting a note."""
    # Arrange - First create a note
    create_params = {
        "text": "Note to be deleted",
        "title": "Delete Test"
    }
    create_result = await skill._execute(**create_params)
    note_id = create_result.result["note_id"]
    
    # Act - Delete the note
    delete_params = {
        "action": "delete",
        "note_id": note_id
    }
    result = await skill._execute(**delete_params)
    
    # Assert
    assert result.success is True
    assert result.result["success"] is True
    
    # Verify note file is gone
    note_path = skill.notes_dir / f"{note_id}.json"
    assert not note_path.exists()


@pytest.mark.asyncio
async def test_list_notes(skill):
    """Test listing notes with filtering."""
    # Arrange - Create multiple notes
    await skill._execute(text="Note 1", title="First Note", tags=["test", "important"])
    await skill._execute(text="Note 2", title="Second Note", tags=["test"])
    await skill._execute(text="Note 3", title="Third Note", tags=["archive"])
    
    # Act - List all notes
    list_result = await skill._execute(action="list")
    
    # Assert
    assert list_result.success is True
    assert list_result.result["total_count"] == 3
    
    # Test filtering by tag
    tag_filter_result = await skill._execute(action="list", tags=["important"])
    assert tag_filter_result.result["total_count"] == 1
    assert tag_filter_result.result["notes"][0]["title"] == "First Note"


@pytest.mark.asyncio
async def test_export_note(skill):
    """Test exporting a note in different formats."""
    # Arrange - Create a note
    create_params = {
        "text": "This is a test note with *markdown* content.\n\nIt has multiple paragraphs.",
        "title": "Export Test",
        "tags": ["export", "test"]
    }
    create_result = await skill._execute(**create_params)
    note_id = create_result.result["note_id"]
    
    # Act - Export as markdown
    md_export = await skill._execute(action="export", note_id=note_id, export_format="markdown")
    
    # Assert
    assert md_export.success is True
    assert "# Export Test" in md_export.result["content"]
    assert md_export.result["format"] == "markdown"
    
    # Test JSON export
    json_export = await skill._execute(action="export", note_id=note_id, export_format="json")
    assert json_export.success is True
    assert json_export.result["format"] == "json"
    # Verify it's valid JSON
    json_content = json.loads(json_export.result["content"])
    assert json_content["title"] == "Export Test"


@pytest.mark.asyncio
async def test_note_to_markdown(skill):
    """Test conversion of note to markdown format."""
    # Arrange
    note = VoiceNote(
        note_id="test123",
        title="Markdown Test",
        content="This is the main content.",
        summary="This is a summary.",
        language=NoteLanguage.ENGLISH,
        categories={NoteCategory.MEETING, NoteCategory.IDEA},
        tags=["markdown", "test"],
        created_at=datetime.now(timezone.utc),
        bullet_points=["Point 1", "Point 2"],
        action_items=["Action 1", "Action 2"]
    )
    
    # Act
    markdown = skill._note_to_markdown(note)
    
    # Assert
    assert "# Markdown Test" in markdown
    assert "## Content" in markdown
    assert "## Summary" in markdown
    assert "## Key Points" in markdown
    assert "* Point 1" in markdown
    assert "## Action Items" in markdown
    assert "- [ ] Action 1" in markdown
