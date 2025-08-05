"""
Tests for Note Taking Skill
Author: Drmusab
Last Modified: 2025-01-08

Comprehensive tests for the voice-based note taking skill.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.memory.core_memory.memory_manager import MemoryManager
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.speech.speech_to_text import (
    EnhancedWhisperTranscriber,
    TranscriptionResult,
    TranscriptionRequest
)
from src.skills.builtin.note_taker.note_taker_skill import (
    NoteTakerSkill,
    NoteCategory,
    NoteStatus,
    Note,
    NoteMetadata
)
from src.skills.builtin.note_taker.content_analyzer import ContentAnalyzer
from src.skills.builtin.note_taker.summarizer import NoteSummarizer
from src.skills.builtin.note_taker.export_manager import ExportManager


@pytest.fixture
def mock_container():
    """Create a mock container with required dependencies."""
    container = MagicMock(spec=Container)
    
    # Mock config
    config = MagicMock(spec=ConfigLoader)
    config.get.return_value = {
        "audio": {"format": "wav", "quality": "high"},
        "processing": {"auto_summarize": True, "detect_action_items": True},
        "export": {"default_format": "markdown"},
        "storage": {"notes_directory": "/tmp/test_notes"}
    }
    
    # Mock dependencies
    memory_manager = AsyncMock(spec=MemoryManager)
    bilingual_manager = MagicMock(spec=BilingualManager)
    speech_transcriber = AsyncMock(spec=EnhancedWhisperTranscriber)
    
    container.get.side_effect = lambda cls: {
        ConfigLoader: config,
        MemoryManager: memory_manager,
        BilingualManager: bilingual_manager,
        EnhancedWhisperTranscriber: speech_transcriber
    }.get(cls)
    
    container.get_optional.return_value = None
    
    return container


@pytest.fixture
def note_taker_skill(mock_container):
    """Create a NoteTakerSkill instance for testing."""
    return NoteTakerSkill(mock_container)


@pytest.fixture
def sample_transcription_result():
    """Create a sample transcription result for testing."""
    return TranscriptionResult(
        text="This is a test note about machine learning concepts. We need to review the neural network architecture.",
        confidence=0.95,
        language="en",
        segments=[],
        processing_time=2.5,
        quality_metrics={"word_count": 16},
        entities=[
            {"text": "machine learning", "label": "CONCEPT"},
            {"text": "neural network", "label": "TECHNOLOGY"}
        ],
        sentiment={"positive": 0.8, "neutral": 0.2, "negative": 0.0}
    )


class TestNoteTakerSkill:
    """Test cases for NoteTakerSkill."""
    
    @pytest.mark.asyncio
    async def test_start_recording(self, note_taker_skill):
        """Test starting a recording session."""
        result = await note_taker_skill.start_recording(
            user_id="test_user",
            session_id="test_session"
        )
        
        assert "note_id" in result
        assert result["status"] == "recording"
        assert "Recording started" in result["message"]
        assert result["note_id"] in note_taker_skill.active_sessions
        assert result["note_id"] in note_taker_skill.notes
        
        # Check note metadata
        note = note_taker_skill.notes[result["note_id"]]
        assert note.metadata.status == NoteStatus.RECORDING
        assert note.metadata.category == NoteCategory.GENERAL
    
    @pytest.mark.asyncio
    async def test_stop_recording(self, note_taker_skill, sample_transcription_result):
        """Test stopping a recording and processing the note."""
        # Start recording first
        start_result = await note_taker_skill.start_recording()
        note_id = start_result["note_id"]
        
        # Mock the transcription process
        with patch.object(note_taker_skill, '_transcribe_audio', return_value=sample_transcription_result):
            with patch.object(note_taker_skill.audio_processor, 'save_recording_session', return_value=Path("/tmp/test.wav")):
                result = await note_taker_skill.stop_recording(note_id)
        
        assert result["note_id"] == note_id
        assert result["status"] == "completed"
        assert "summary" in result
        assert "content" in result
        
        # Check note was processed
        note = note_taker_skill.notes[note_id]
        assert note.metadata.status == NoteStatus.COMPLETED
        assert note.transcription == sample_transcription_result.text
        assert note.metadata.confidence == sample_transcription_result.confidence
        
        # Check session was cleaned up
        assert note_id not in note_taker_skill.active_sessions
    
    @pytest.mark.asyncio
    async def test_stop_recording_invalid_note(self, note_taker_skill):
        """Test stopping recording with invalid note ID."""
        with pytest.raises(ValueError, match="No active recording session"):
            await note_taker_skill.stop_recording("invalid_note_id")
    
    @pytest.mark.asyncio
    async def test_get_note(self, note_taker_skill):
        """Test retrieving a note."""
        # Create a test note
        note_metadata = NoteMetadata(
            id="test_note",
            title="Test Note",
            category=NoteCategory.IDEAS,
            language="en",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status=NoteStatus.COMPLETED
        )
        note = Note(
            metadata=note_metadata,
            transcription="Test transcription",
            summary="Test summary"
        )
        note_taker_skill.notes["test_note"] = note
        
        result = await note_taker_skill.get_note("test_note")
        
        assert result["metadata"]["id"] == "test_note"
        assert result["content"]["transcription"] == "Test transcription"
        assert result["content"]["summary"] == "Test summary"
    
    @pytest.mark.asyncio
    async def test_get_note_not_found(self, note_taker_skill):
        """Test retrieving a non-existent note."""
        with pytest.raises(ValueError, match="Note .* not found"):
            await note_taker_skill.get_note("non_existent_note")
    
    @pytest.mark.asyncio
    async def test_search_notes(self, note_taker_skill):
        """Test searching notes."""
        # Create test notes
        note1 = Note(
            metadata=NoteMetadata(
                id="note1",
                title="ML Concepts",
                category=NoteCategory.STUDY_NOTES,
                language="en",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=["ml", "ai"]
            ),
            transcription="Machine learning is fascinating"
        )
        note2 = Note(
            metadata=NoteMetadata(
                id="note2",
                title="Meeting Notes",
                category=NoteCategory.MEETING_NOTES,
                language="ar",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=["meeting", "work"]
            ),
            transcription="اجتماع مهم حول المشروع"
        )
        
        note_taker_skill.notes["note1"] = note1
        note_taker_skill.notes["note2"] = note2
        
        # Test search by query
        results = await note_taker_skill.search_notes(query="machine")
        assert len(results) == 1
        assert results[0]["note_id"] == "note1"
        
        # Test search by category
        results = await note_taker_skill.search_notes(category="meeting_notes")
        assert len(results) == 1
        assert results[0]["note_id"] == "note2"
        
        # Test search by language
        results = await note_taker_skill.search_notes(language="ar")
        assert len(results) == 1
        assert results[0]["note_id"] == "note2"
        
        # Test search by tags
        results = await note_taker_skill.search_notes(tags=["ml"])
        assert len(results) == 1
        assert results[0]["note_id"] == "note1"
    
    @pytest.mark.asyncio
    async def test_export_note(self, note_taker_skill):
        """Test exporting a note."""
        # Create a test note
        note = Note(
            metadata=NoteMetadata(
                id="test_note",
                title="Test Note",
                category=NoteCategory.IDEAS,
                language="en",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ),
            transcription="Test transcription",
            summary="Test summary"
        )
        note_taker_skill.notes["test_note"] = note
        
        with patch.object(note_taker_skill.export_manager, 'export_note') as mock_export:
            mock_export.return_value = {
                "note_id": "test_note",
                "export_path": "/tmp/test_note.md",
                "format": "markdown"
            }
            
            result = await note_taker_skill.export_note("test_note", format="markdown")
            
            assert result["note_id"] == "test_note"
            assert result["format"] == "markdown"
            mock_export.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_categories(self, note_taker_skill):
        """Test getting available categories."""
        categories = await note_taker_skill.get_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        # Check structure
        for category in categories:
            assert "value" in category
            assert "label" in category
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, note_taker_skill):
        """Test getting note statistics."""
        # Add some test notes
        note1 = Note(
            metadata=NoteMetadata(
                id="note1",
                title="Note 1",
                category=NoteCategory.IDEAS,
                language="en",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                status=NoteStatus.COMPLETED,
                word_count=50,
                duration=60.0,
                confidence=0.9
            )
        )
        note2 = Note(
            metadata=NoteMetadata(
                id="note2",
                title="Note 2",
                category=NoteCategory.TASKS,
                language="ar",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                status=NoteStatus.COMPLETED,
                word_count=30,
                duration=45.0,
                confidence=0.8
            )
        )
        
        note_taker_skill.notes["note1"] = note1
        note_taker_skill.notes["note2"] = note2
        
        stats = await note_taker_skill.get_statistics()
        
        assert stats["total_notes"] == 2
        assert stats["completed_notes"] == 2
        assert stats["total_words"] == 80
        assert stats["total_duration_minutes"] == 1.75  # (60+45)/60
        assert "category_distribution" in stats
        assert "language_distribution" in stats


class TestContentAnalyzer:
    """Test cases for ContentAnalyzer."""
    
    @pytest.fixture
    def content_analyzer(self, mock_container):
        """Create a ContentAnalyzer instance for testing."""
        return ContentAnalyzer(mock_container)
    
    @pytest.mark.asyncio
    async def test_categorize_content_tasks(self, content_analyzer):
        """Test categorizing task-related content."""
        text = "I need to complete the project report and submit it by Friday. Also todo: review the presentation."
        
        category = await content_analyzer.categorize_content(text)
        assert category == NoteCategory.TASKS
    
    @pytest.mark.asyncio
    async def test_categorize_content_ideas(self, content_analyzer):
        """Test categorizing idea-related content."""
        text = "I have an innovative idea for improving the user interface. This concept could revolutionize the way we interact with the system."
        
        category = await content_analyzer.categorize_content(text)
        assert category == NoteCategory.IDEAS
    
    @pytest.mark.asyncio
    async def test_categorize_content_arabic(self, content_analyzer):
        """Test categorizing Arabic content."""
        text = "لدي فكرة مبتكرة لتحسين النظام. هذا المفهوم يمكن أن يحدث ثورة في طريقة تفاعلنا مع التكنولوجيا."
        
        category = await content_analyzer.categorize_content(text)
        assert category == NoteCategory.IDEAS
    
    @pytest.mark.asyncio
    async def test_extract_action_items(self, content_analyzer):
        """Test extracting action items."""
        text = "We need to review the code. I should call the client tomorrow. Follow up with the team about the deadline."
        
        actions = await content_analyzer.extract_action_items(text)
        
        assert len(actions) > 0
        assert any("review" in action.lower() for action in actions)
        assert any("call" in action.lower() for action in actions)
    
    @pytest.mark.asyncio
    async def test_extract_definitions(self, content_analyzer):
        """Test extracting definitions."""
        text = "Machine Learning is a subset of artificial intelligence. API refers to Application Programming Interface."
        
        definitions = await content_analyzer.extract_definitions(text)
        
        assert len(definitions) > 0
        assert "Machine Learning" in definitions or "API" in definitions
    
    @pytest.mark.asyncio
    async def test_extract_tags(self, content_analyzer):
        """Test extracting tags."""
        text = "This is about #machinelearning and #ai. We discussed neural networks and deep learning."
        
        tags = await content_analyzer.extract_tags(text)
        
        assert len(tags) > 0
        assert any("machinelearning" in tag for tag in tags)
    
    @pytest.mark.asyncio
    async def test_generate_title(self, content_analyzer):
        """Test generating title from content."""
        text = "Today we discussed the new project requirements. The team agreed on the technical approach and timeline."
        
        title = await content_analyzer.generate_title(text)
        
        assert len(title) > 0
        assert len(title) <= 50  # Default max length
        assert "project" in title.lower() or "discuss" in title.lower()


class TestNoteSummarizer:
    """Test cases for NoteSummarizer."""
    
    @pytest.fixture
    def note_summarizer(self, mock_container):
        """Create a NoteSummarizer instance for testing."""
        return NoteSummarizer(mock_container)
    
    @pytest.mark.asyncio
    async def test_create_summary(self, note_summarizer):
        """Test creating a summary."""
        text = """
        Machine learning is a powerful subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze data and make predictions or decisions. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Each type has its own applications and methodologies. The field has grown significantly in recent years and is now used in various industries including healthcare, finance, and technology.
        """
        
        summary = await note_summarizer.create_summary(text, max_length=100)
        
        assert len(summary) > 0
        assert len(summary) <= 120  # Allow some flexibility
        assert "machine learning" in summary.lower()
    
    @pytest.mark.asyncio
    async def test_extract_key_points(self, note_summarizer):
        """Test extracting key points."""
        text = """
        The meeting covered several important topics. First, we discussed the project timeline and agreed on key milestones. Second, the budget allocation was reviewed and approved. Third, we identified potential risks and mitigation strategies. Finally, the team assignments were finalized for the next phase.
        """
        
        key_points = await note_summarizer.extract_key_points(text, max_points=3)
        
        assert len(key_points) <= 3
        assert len(key_points) > 0
        for point in key_points:
            assert len(point) > 10  # Should be meaningful points
    
    @pytest.mark.asyncio
    async def test_create_bullet_points(self, note_summarizer):
        """Test creating bullet points."""
        text = "The project has three phases: planning, development, and testing. Each phase has specific deliverables."
        
        bullet_points = await note_summarizer.create_bullet_points(text)
        
        assert len(bullet_points) > 0
        for point in bullet_points:
            assert point.startswith("• ")
    
    @pytest.mark.asyncio
    async def test_create_structured_outline(self, note_summarizer):
        """Test creating structured outline."""
        text = """
        Project Overview
        The new software project aims to improve user experience.
        
        Technical Requirements
        The system must support multiple platforms and handle high traffic.
        
        Timeline
        Development will take 6 months with testing in the final month.
        """
        
        outline = await note_summarizer.create_structured_outline(text)
        
        assert "summary" in outline
        assert "key_points" in outline
        assert "bullet_points" in outline
        assert "sections" in outline
        assert len(outline["sections"]) > 0


class TestExportManager:
    """Test cases for ExportManager."""
    
    @pytest.fixture
    def export_manager(self, mock_container):
        """Create an ExportManager instance for testing."""
        return ExportManager(mock_container)
    
    @pytest.fixture
    def sample_note(self):
        """Create a sample note for testing."""
        return Note(
            metadata=NoteMetadata(
                id="test_note",
                title="Test Note",
                category=NoteCategory.IDEAS,
                language="en",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=["test", "sample"]
            ),
            transcription="This is a test transcription.",
            summary="Test summary",
            key_points=["Point 1", "Point 2"],
            action_items=["Action 1", "Action 2"]
        )
    
    @pytest.mark.asyncio
    async def test_export_to_markdown(self, export_manager, sample_note, tmp_path):
        """Test exporting to markdown format."""
        # Set export directory to temp path
        export_manager.export_directory = tmp_path
        
        result = await export_manager.export_note(sample_note, format="markdown")
        
        assert result["format"] == "markdown"
        assert result["note_id"] == "test_note"
        
        # Check file was created
        export_path = Path(result["export_path"])
        assert export_path.exists()
        
        # Check content
        content = export_path.read_text(encoding='utf-8')
        assert "# Test Note" in content
        assert "## Summary" in content
        assert "Test summary" in content
        assert "## Key Points" in content
        assert "## Action Items" in content
    
    @pytest.mark.asyncio
    async def test_export_to_json(self, export_manager, sample_note, tmp_path):
        """Test exporting to JSON format."""
        export_manager.export_directory = tmp_path
        
        result = await export_manager.export_note(sample_note, format="json")
        
        assert result["format"] == "json"
        
        # Check file was created and is valid JSON
        export_path = Path(result["export_path"])
        assert export_path.exists()
        
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data["metadata"]["id"] == "test_note"
        assert data["content"]["transcription"] == "This is a test transcription."
        assert "export_info" in data
    
    @pytest.mark.asyncio
    async def test_export_to_text(self, export_manager, sample_note, tmp_path):
        """Test exporting to text format."""
        export_manager.export_directory = tmp_path
        
        result = await export_manager.export_note(sample_note, format="text")
        
        assert result["format"] == "text"
        
        # Check file content
        export_path = Path(result["export_path"])
        content = export_path.read_text(encoding='utf-8')
        
        assert "Test Note" in content
        assert "SUMMARY" in content
        assert "KEY POINTS" in content
        assert "ACTION ITEMS" in content
        assert "FULL TRANSCRIPTION" in content
    
    @pytest.mark.asyncio
    async def test_export_invalid_format(self, export_manager, sample_note):
        """Test exporting with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            await export_manager.export_note(sample_note, format="invalid")
    
    @pytest.mark.asyncio
    async def test_sanitize_filename(self, export_manager):
        """Test filename sanitization."""
        # Test with invalid characters
        result = export_manager._sanitize_filename("Test/Note<>:File")
        assert "/" not in result
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        
        # Test with spaces
        result = export_manager._sanitize_filename("Test Note File")
        assert "Test_Note_File" == result
        
        # Test with very long name
        long_name = "a" * 100
        result = export_manager._sanitize_filename(long_name)
        assert len(result) <= 50


if __name__ == "__main__":
    pytest.main([__file__])