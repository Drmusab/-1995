"""
Voice-Based Note Taking Skill
Author: Drmusab
Last Modified: 2025-01-08

Main skill implementation for voice-based note taking with Arabic/English support,
automatic transcription, summarization, categorization, and export capabilities.
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import SkillExecutionCompleted, SkillExecutionStarted
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.speech.speech_to_text import (
    EnhancedWhisperTranscriber,
    TranscriptionRequest,
    TranscriptionResult,
    AudioSource,
    TranscriptionQuality,
)

from .audio_processor import AudioProcessor as NoteAudioProcessor
from .content_analyzer import ContentAnalyzer
from .export_manager import ExportManager
from .summarizer import NoteSummarizer


class NoteCategory(Enum):
    """Note categories for automatic classification."""
    TASKS = "tasks"
    IDEAS = "ideas"
    QUOTES = "quotes"
    MEMORY = "memory"
    EXPERIENCE = "experience"
    MEETING_NOTES = "meeting_notes"
    STUDY_NOTES = "study_notes"
    PERSONAL = "personal"
    TECHNICAL = "technical"
    GENERAL = "general"


class NoteStatus(Enum):
    """Note processing status."""
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPORTED = "exported"


@dataclass
class NoteMetadata:
    """Metadata for a note."""
    id: str
    title: str
    category: NoteCategory
    language: str
    created_at: datetime
    updated_at: datetime
    duration: float = 0.0
    word_count: int = 0
    confidence: float = 0.0
    tags: List[str] = None
    status: NoteStatus = NoteStatus.RECORDING
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class Note:
    """Complete note with content and metadata."""
    metadata: NoteMetadata
    original_audio_path: Optional[str] = None
    transcription: str = ""
    summary: str = ""
    key_points: List[str] = None
    action_items: List[str] = None
    definitions: Dict[str, str] = None
    entities: List[Dict[str, Any]] = None
    sentiment: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.key_points is None:
            self.key_points = []
        if self.action_items is None:
            self.action_items = []
        if self.definitions is None:
            self.definitions = {}
        if self.entities is None:
            self.entities = []
        if self.sentiment is None:
            self.sentiment = {}


class NoteTakerSkill:
    """
    Advanced voice-based note taking skill with bilingual support.
    
    Features:
    - Real-time voice recording and transcription
    - Arabic/English bilingual support
    - Automatic content categorization and summarization
    - Action item detection and key concept extraction
    - Multiple export formats (Text, JSON, Markdown, PDF)
    - Integration with existing memory and session systems
    """
    
    def __init__(self, container: Container):
        """Initialize the note taking skill."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core dependencies
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.memory_manager = container.get(MemoryManager)
        
        # Processing components
        self.speech_transcriber = container.get(EnhancedWhisperTranscriber)
        self.bilingual_manager = container.get(BilingualManager)
        
        # Note-specific components
        self.audio_processor = NoteAudioProcessor(container)
        self.content_analyzer = ContentAnalyzer(container)
        self.summarizer = NoteSummarizer(container)
        self.export_manager = ExportManager(container)
        
        # Configuration
        self._setup_configuration()
        
        # Active recording sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Notes storage (in production would use database)
        self.notes: Dict[str, Note] = {}
        
        self.logger.info("NoteTakerSkill initialized successfully")
    
    def _setup_configuration(self) -> None:
        """Setup note taker configuration."""
        note_config = self.config.get("note_taker", {})
        
        # Audio settings
        self.audio_format = note_config.get("audio", {}).get("format", "wav")
        self.audio_quality = note_config.get("audio", {}).get("quality", "high")
        self.auto_enhance = note_config.get("audio", {}).get("auto_enhance", True)
        
        # Processing settings
        self.auto_summarize = note_config.get("processing", {}).get("auto_summarize", True)
        self.detect_action_items = note_config.get("processing", {}).get("detect_action_items", True)
        self.extract_definitions = note_config.get("processing", {}).get("extract_definitions", True)
        self.categorize_content = note_config.get("processing", {}).get("categorize_content", True)
        
        # Export settings
        self.default_export_format = note_config.get("export", {}).get("default_format", "markdown")
        self.include_audio_links = note_config.get("export", {}).get("include_audio_links", True)
        
        # Storage settings
        self.notes_directory = Path(note_config.get("storage", {}).get("notes_directory", "data/notes"))
        self.audio_directory = Path(note_config.get("storage", {}).get("audio_directory", "data/notes/audio"))
        
        # Ensure directories exist
        self.notes_directory.mkdir(parents=True, exist_ok=True)
        self.audio_directory.mkdir(parents=True, exist_ok=True)
    
    @handle_exceptions
    async def start_recording(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Start a new voice recording session.
        
        Args:
            user_id: User ID for the session
            session_id: Optional session ID
            metadata: Additional metadata for the note
            
        Returns:
            Recording session information
        """
        # Generate unique note ID
        note_id = str(uuid.uuid4())
        
        # Create note metadata
        note_metadata = NoteMetadata(
            id=note_id,
            title="Untitled Note",
            category=NoteCategory.GENERAL,
            language="ar",  # Default to Arabic as primary
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status=NoteStatus.RECORDING
        )
        
        # Create note
        note = Note(metadata=note_metadata)
        self.notes[note_id] = note
        
        # Setup recording session
        recording_session = {
            "note_id": note_id,
            "user_id": user_id,
            "session_id": session_id,
            "start_time": datetime.now(timezone.utc),
            "audio_buffer": [],
            "is_recording": True,
            "metadata": metadata or {}
        }
        
        self.active_sessions[note_id] = recording_session
        
        # Emit event
        await self.event_bus.emit(
            SkillExecutionStarted(
                skill_name="note_taker",
                session_id=session_id or note_id,
                user_id=user_id,
                parameters={"action": "start_recording", "note_id": note_id}
            )
        )
        
        self.logger.info(f"Started recording session for note {note_id}")
        
        return {
            "note_id": note_id,
            "status": "recording",
            "message": "Recording started. Speak naturally in Arabic or English.",
            "session_info": {
                "start_time": recording_session["start_time"].isoformat(),
                "audio_format": self.audio_format,
                "bilingual_support": True
            }
        }
    
    @handle_exceptions
    async def stop_recording(self, note_id: str) -> Dict[str, Any]:
        """
        Stop recording and process the captured audio.
        
        Args:
            note_id: ID of the note to stop recording
            
        Returns:
            Processing results
        """
        if note_id not in self.active_sessions:
            raise ValueError(f"No active recording session found for note {note_id}")
        
        session = self.active_sessions[note_id]
        session["is_recording"] = False
        
        note = self.notes[note_id]
        note.metadata.status = NoteStatus.PROCESSING
        note.metadata.updated_at = datetime.now(timezone.utc)
        
        try:
            # Calculate duration
            duration = (datetime.now(timezone.utc) - session["start_time"]).total_seconds()
            note.metadata.duration = duration
            
            # Process the audio (simulated - in real implementation would use actual audio)
            audio_path = await self.audio_processor.save_recording_session(session)
            note.original_audio_path = str(audio_path)
            
            # Transcribe audio
            transcription_result = await self._transcribe_audio(audio_path, note_id)
            note.transcription = transcription_result.text
            note.metadata.confidence = transcription_result.confidence
            note.metadata.language = transcription_result.language
            note.metadata.word_count = len(transcription_result.text.split())
            
            # Analyze and process content
            if self.categorize_content:
                category = await self.content_analyzer.categorize_content(transcription_result.text)
                note.metadata.category = category
            
            if self.auto_summarize:
                note.summary = await self.summarizer.create_summary(transcription_result.text)
                note.key_points = await self.summarizer.extract_key_points(transcription_result.text)
            
            if self.detect_action_items:
                note.action_items = await self.content_analyzer.extract_action_items(transcription_result.text)
            
            if self.extract_definitions:
                note.definitions = await self.content_analyzer.extract_definitions(transcription_result.text)
            
            # Extract entities and sentiment from transcription result
            note.entities = transcription_result.entities or []
            note.sentiment = transcription_result.sentiment or {}
            
            # Generate title
            note.metadata.title = await self.content_analyzer.generate_title(transcription_result.text)
            
            # Extract tags
            note.metadata.tags = await self.content_analyzer.extract_tags(transcription_result.text)
            
            note.metadata.status = NoteStatus.COMPLETED
            note.metadata.updated_at = datetime.now(timezone.utc)
            
            # Store in memory system
            await self._store_note_in_memory(note)
            
            # Clean up session
            del self.active_sessions[note_id]
            
            # Emit completion event
            await self.event_bus.emit(
                SkillExecutionCompleted(
                    skill_name="note_taker",
                    session_id=session.get("session_id", note_id),
                    user_id=session.get("user_id"),
                    result={"note_id": note_id, "status": "completed"},
                    processing_time=duration
                )
            )
            
            self.logger.info(f"Completed processing note {note_id}")
            
            return {
                "note_id": note_id,
                "status": "completed",
                "summary": {
                    "title": note.metadata.title,
                    "category": note.metadata.category.value,
                    "language": note.metadata.language,
                    "duration": duration,
                    "word_count": note.metadata.word_count,
                    "confidence": note.metadata.confidence,
                    "action_items_count": len(note.action_items),
                    "key_points_count": len(note.key_points),
                    "tags": note.metadata.tags
                },
                "content": {
                    "transcription": note.transcription,
                    "summary": note.summary,
                    "key_points": note.key_points,
                    "action_items": note.action_items
                }
            }
            
        except Exception as e:
            note.metadata.status = NoteStatus.FAILED
            note.metadata.updated_at = datetime.now(timezone.utc)
            
            # Clean up session
            if note_id in self.active_sessions:
                del self.active_sessions[note_id]
            
            self.logger.error(f"Failed to process note {note_id}: {str(e)}")
            raise
    
    async def _transcribe_audio(self, audio_path: Path, note_id: str) -> TranscriptionResult:
        """Transcribe audio using the existing speech pipeline."""
        request = TranscriptionRequest(
            audio_source=AudioSource.FILE,
            language=None,  # Auto-detect
            quality=TranscriptionQuality.HIGH_QUALITY,
            enable_preprocessing=True,
            enable_emotion_detection=True,
            enable_speaker_identification=False,
            context_aware=True,
            cache_result=True,
            session_id=note_id
        )
        
        return await self.speech_transcriber.transcribe(audio_path, request)
    
    async def _store_note_in_memory(self, note: Note) -> None:
        """Store note in the memory system for future retrieval."""
        try:
            memory_data = {
                "type": "note",
                "note_id": note.metadata.id,
                "title": note.metadata.title,
                "category": note.metadata.category.value,
                "language": note.metadata.language,
                "transcription": note.transcription,
                "summary": note.summary,
                "key_points": note.key_points,
                "action_items": note.action_items,
                "tags": note.metadata.tags,
                "created_at": note.metadata.created_at.isoformat(),
                "word_count": note.metadata.word_count,
                "confidence": note.metadata.confidence
            }
            
            # Store in memory manager
            await self.memory_manager.store_memory(
                memory_type="semantic",
                content=memory_data,
                metadata={
                    "skill": "note_taker",
                    "category": note.metadata.category.value,
                    "language": note.metadata.language,
                    "tags": note.metadata.tags
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to store note {note.metadata.id} in memory: {str(e)}")
    
    async def get_note(self, note_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific note by ID.
        
        Args:
            note_id: ID of the note to retrieve
            
        Returns:
            Note data
        """
        if note_id not in self.notes:
            raise ValueError(f"Note {note_id} not found")
        
        note = self.notes[note_id]
        
        return {
            "metadata": asdict(note.metadata),
            "content": {
                "transcription": note.transcription,
                "summary": note.summary,
                "key_points": note.key_points,
                "action_items": note.action_items,
                "definitions": note.definitions,
                "entities": note.entities,
                "sentiment": note.sentiment
            },
            "audio_path": note.original_audio_path
        }
    
    async def search_notes(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search notes by content, category, tags, or language.
        
        Args:
            query: Text query to search in note content
            category: Note category filter
            tags: Tags to filter by
            language: Language filter
            limit: Maximum number of results
            
        Returns:
            List of matching notes
        """
        matching_notes = []
        
        for note in self.notes.values():
            matches = True
            
            # Category filter
            if category and note.metadata.category.value != category:
                matches = False
            
            # Language filter
            if language and note.metadata.language != language:
                matches = False
            
            # Tags filter
            if tags and not any(tag in note.metadata.tags for tag in tags):
                matches = False
            
            # Text query filter
            if query and query.lower() not in note.transcription.lower():
                matches = False
            
            if matches:
                matching_notes.append({
                    "note_id": note.metadata.id,
                    "title": note.metadata.title,
                    "category": note.metadata.category.value,
                    "language": note.metadata.language,
                    "created_at": note.metadata.created_at.isoformat(),
                    "summary": note.summary,
                    "tags": note.metadata.tags,
                    "word_count": note.metadata.word_count,
                    "confidence": note.metadata.confidence
                })
        
        # Sort by creation time (most recent first)
        matching_notes.sort(key=lambda x: x["created_at"], reverse=True)
        
        return matching_notes[:limit]
    
    async def export_note(
        self,
        note_id: str,
        format: str = "markdown",
        include_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Export a note in the specified format.
        
        Args:
            note_id: ID of the note to export
            format: Export format (markdown, json, text, pdf)
            include_audio: Whether to include audio file reference
            
        Returns:
            Export result with file path and content
        """
        if note_id not in self.notes:
            raise ValueError(f"Note {note_id} not found")
        
        note = self.notes[note_id]
        
        # Export the note
        export_result = await self.export_manager.export_note(
            note,
            format=format,
            include_audio=include_audio and self.include_audio_links
        )
        
        # Update note status
        note.metadata.status = NoteStatus.EXPORTED
        note.metadata.updated_at = datetime.now(timezone.utc)
        
        return export_result
    
    async def get_categories(self) -> List[Dict[str, str]]:
        """Get available note categories."""
        return [
            {"value": category.value, "label": category.value.replace("_", " ").title()}
            for category in NoteCategory
        ]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get note-taking statistics."""
        total_notes = len(self.notes)
        completed_notes = sum(1 for note in self.notes.values() if note.metadata.status == NoteStatus.COMPLETED)
        
        # Category distribution
        category_counts = {}
        for note in self.notes.values():
            category = note.metadata.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Language distribution
        language_counts = {}
        for note in self.notes.values():
            lang = note.metadata.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Total words and duration
        total_words = sum(note.metadata.word_count for note in self.notes.values())
        total_duration = sum(note.metadata.duration for note in self.notes.values())
        
        return {
            "total_notes": total_notes,
            "completed_notes": completed_notes,
            "active_sessions": len(self.active_sessions),
            "category_distribution": category_counts,
            "language_distribution": language_counts,
            "total_words": total_words,
            "total_duration_minutes": total_duration / 60,
            "average_confidence": sum(note.metadata.confidence for note in self.notes.values()) / max(total_notes, 1)
        }