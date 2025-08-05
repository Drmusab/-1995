"""
Voice Note Skill Template

This template provides a starting point for creating custom voice note taking skills.
Developers can extend or modify this template for specialized use cases.

Author: Drmusab
Created: 2025-08-05
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from dataclasses import dataclass, field

from src.core.dependency_injection import Container
from src.skills.builtin.core_skills import BaseSkill, SkillMetadata, SkillResult, SkillCategory
from src.observability.logging.config import get_logger


@dataclass
class CustomVoiceNote:
    """Custom voice note data structure."""
    note_id: str
    title: str
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audio_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CustomVoiceNoteSkill(BaseSkill):
    """
    Template for custom voice note taking skills.
    
    This template provides basic functionality that can be extended for specific use cases:
    - Recording and transcribing voice notes
    - Saving notes with metadata
    - Basic retrieval and management functions
    
    Developers should customize this template by:
    1. Adding specialized processing for their domain
    2. Extending the note data structure with domain-specific fields
    3. Implementing custom analysis or export functions
    """

    def __init__(self, container: Container):
        super().__init__(container)
        self.logger = get_logger(__name__)
        
        # Core dependencies - customize as needed
        try:
            self.speech_to_text = self._get_component("SpeechToText")
        except Exception as e:
            self.logger.error(f"Failed to initialize dependencies: {e}")
            raise
            
        # Storage paths - customize for your implementation
        self.notes_dir = Path("data/user_data/custom_voice_notes")
        self.audio_dir = Path("data/user_data/custom_voice_notes/audio")
        self._ensure_directories()
        
        # In-memory cache
        self._notes_cache: Dict[str, CustomVoiceNote] = {}

    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata - customize for your implementation."""
        return SkillMetadata(
            skill_id="custom_voice_note_skill",  # Change to your unique ID
            name="Custom Voice Note Skill",      # Change to your skill name
            description="Template for custom voice note taking skills.",
            version="1.0.0",
            category=SkillCategory.PRODUCTIVITY,
            parameters={
                "audio_data": {
                    "type": "numpy.ndarray",
                    "description": "Raw audio data for voice note",
                    "required": False
                },
                "text": {
                    "type": "str",
                    "description": "Text input (if audio not provided)",
                    "required": False
                },
                "title": {
                    "type": "str", 
                    "description": "Note title",
                    "required": False
                },
                "action": {
                    "type": "str",
                    "description": "Action to perform (create, retrieve, list)",
                    "default": "create",
                    "required": False
                },
                "note_id": {
                    "type": "str",
                    "description": "Note ID for retrieve operations",
                    "required": False
                }
            },
            author="Your Name",
            tags={"voice", "notes", "template"},
            is_stateful=True,
            timeout_seconds=30.0
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute the custom voice note skill - extend with your logic."""
        action = params.get("action", "create")
        
        try:
            if action == "create":
                result = await self._create_note(**params)
            elif action == "retrieve":
                result = await self._retrieve_note(params.get("note_id"))
            elif action == "list":
                result = await self._list_notes()
            else:
                return SkillResult(
                    success=False,
                    result={"error": f"Unknown action: {action}"},
                    errors=[f"Unknown action: {action}"]
                )
                
            return SkillResult(
                success=True,
                result=result
            )
            
        except Exception as e:
            self.logger.error(f"Error in custom voice note skill: {e}")
            return SkillResult(
                success=False,
                result={"error": str(e)},
                errors=[str(e)]
            )

    async def _create_note(self, **params) -> Dict[str, Any]:
        """Create a new voice note - customize with your logic."""
        # Get input content (from audio or text)
        audio_data = params.get("audio_data")
        text_input = params.get("text")
        
        if audio_data is not None:
            # Process audio data to text
            transcription_result = await self.speech_to_text.transcribe(audio_data)
            text = transcription_result.get("text", "")
        elif text_input:
            # Use provided text
            text = text_input
        else:
            raise ValueError("Either audio_data or text must be provided")
            
        # Generate note ID and title
        note_id = str(uuid.uuid4())
        title = params.get("title") or f"Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Create note object
        note = CustomVoiceNote(
            note_id=note_id,
            title=title,
            content=text,
            created_at=datetime.now(timezone.utc),
            tags=params.get("tags", []),
            metadata={
                "source": "audio" if audio_data is not None else "text",
            }
        )
        
        # Save note to storage
        await self._save_note(note)
        
        # Add to cache
        self._notes_cache[note_id] = note
        
        # Prepare result
        return {
            "note_id": note_id,
            "title": title,
            "content": text,
            "created_at": note.created_at.isoformat(),
            "tags": note.tags
        }

    async def _retrieve_note(self, note_id: str) -> Dict[str, Any]:
        """Retrieve a note by ID - customize with your logic."""
        if not note_id:
            raise ValueError("Note ID is required")
            
        # Check cache first
        if note_id in self._notes_cache:
            note = self._notes_cache[note_id]
        else:
            # Load from storage
            note_path = self.notes_dir / f"{note_id}.json"
            if not note_path.exists():
                raise ValueError(f"Note with ID {note_id} not found")
                
            with open(note_path, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
                
            # Convert dictionary to note
            note = CustomVoiceNote(
                note_id=note_data["note_id"],
                title=note_data["title"],
                content=note_data["content"],
                created_at=datetime.fromisoformat(note_data["created_at"]),
                audio_path=note_data.get("audio_path"),
                tags=note_data.get("tags", []),
                metadata=note_data.get("metadata", {})
            )
            
            # Add to cache
            self._notes_cache[note_id] = note
            
        # Return note data
        return {
            "note_id": note.note_id,
            "title": note.title,
            "content": note.content,
            "created_at": note.created_at.isoformat(),
            "tags": note.tags,
            "has_audio": bool(note.audio_path)
        }

    async def _list_notes(self) -> Dict[str, Any]:
        """List all notes - customize with your logic."""
        notes = []
        
        # Load all notes
        note_files = list(self.notes_dir.glob("*.json"))
        
        # Process each note file
        for note_file in note_files:
            try:
                with open(note_file, 'r', encoding='utf-8') as f:
                    note_data = json.load(f)
                    
                notes.append({
                    "note_id": note_data["note_id"],
                    "title": note_data["title"],
                    "created_at": note_data["created_at"],
                    "tags": note_data.get("tags", [])
                })
                
            except Exception as e:
                self.logger.error(f"Error processing note file {note_file}: {e}")
        
        # Sort by creation date (newest first)
        notes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "total_count": len(notes),
            "notes": notes
        }

    async def _save_note(self, note: CustomVoiceNote) -> None:
        """Save note to storage - customize with your logic."""
        note_path = self.notes_dir / f"{note.note_id}.json"
        
        # Convert note to dictionary
        note_data = {
            "note_id": note.note_id,
            "title": note.title,
            "content": note.content,
            "created_at": note.created_at.isoformat(),
            "audio_path": note.audio_path,
            "tags": note.tags,
            "metadata": note.metadata
        }
        
        try:
            with open(note_path, 'w', encoding='utf-8') as f:
                json.dump(note_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving note: {e}")
            raise

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        os.makedirs(self.notes_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

    # ADD YOUR CUSTOM METHODS HERE
    # Examples:
    # - Custom analysis functions
    # - Domain-specific processing
    # - Specialized export capabilities
    # - Integration with other components
