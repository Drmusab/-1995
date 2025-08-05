"""
Voice-Based Note Taking & Summarizer Skill
Supports Arabic (primary) and English (secondary)

Author: Drmusab
Created: 2025-08-05

Take meeting notes, class summaries, or personal thoughts from voice input
Structure notes by topic or timestamp
Export as text, JSON, or Markdown
Summarize content into bullet points or action items
Save notes with date and time
Store voice recordings in WAV or FLAC format
Automatically tag/categorize content (tasks, ideas, quotes, memory, experience)
"""

import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from dataclasses import dataclass, field

from src.core.dependency_injection import Container
from src.skills.builtin.core_skills import BaseSkill, SkillMetadata, SkillResult, SkillCategory
from src.processing.speech.audio_utils import save_audio_file
from src.processing.natural_language.bilingual_manager import BilingualProcessor
from src.processing.natural_language.entity_extractor import EntityExtractor
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.observability.logging.config import get_logger


class NoteLanguage(Enum):
    """Supported languages for note taking."""
    ARABIC = "ar"
    ENGLISH = "en"
    AUTO = "auto"  # Auto-detect


class NoteCategory(Enum):
    """Categories for auto-tagging notes."""
    TASK = "task"
    IDEA = "idea"
    QUOTE = "quote"
    MEMORY = "memory"
    EXPERIENCE = "experience"
    MEETING = "meeting"
    CLASS = "class"
    GENERAL = "general"


class NoteFormat(Enum):
    """Export formats for notes."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class AudioFormat(Enum):
    """Supported audio formats for storing recordings."""
    WAV = "wav"
    FLAC = "flac"


@dataclass
class VoiceNote:
    """Voice note data structure."""
    note_id: str
    title: str
    content: str
    summary: Optional[str] = None
    language: NoteLanguage = NoteLanguage.AUTO
    categories: Set[NoteCategory] = field(default_factory=set)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audio_path: Optional[str] = None
    audio_format: AudioFormat = AudioFormat.WAV
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    bullet_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    entities: List[Dict[str, Any]] = field(default_factory=list)


class VoiceNoteTakerSkill(BaseSkill):
    """
    Voice-based note taking and summarization skill with bilingual support.
    
    Features:
    - Records and transcribes voice notes in Arabic (primary) and English (secondary)
    - Automatically categorizes and tags content
    - Summarizes content into bullet points and action items
    - Saves notes with timestamps and voice recordings
    - Exports in multiple formats
    """

    def __init__(self, container: Container):
        super().__init__(container)
        self.logger = get_logger(__name__)
        
        # Core dependencies
        try:
            self.speech_to_text = self._get_component("SpeechToText")
            self.text_to_speech = self._get_component("TextToSpeech")
            self.bilingual_processor = self._get_component(BilingualProcessor)
            self.entity_extractor = self._get_component(EntityExtractor)
            self.sentiment_analyzer = self._get_component(SentimentAnalyzer)
        except Exception as e:
            self.logger.error(f"Failed to initialize dependencies: {e}")
            raise
            
        # Storage paths
        self.notes_dir = Path("data/user_data/voice_notes")
        self.audio_dir = Path("data/user_data/voice_notes/audio")
        self._ensure_directories()
        
        # In-memory cache for recent notes
        self._notes_cache: Dict[str, VoiceNote] = {}
        self.max_cache_size = 100

    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return SkillMetadata(
            skill_id="voice_note_taker",
            name="Voice Note Taker & Summarizer",
            description="Takes and manages voice notes in Arabic and English, with summarization, categorization, and export capabilities.",
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
                "language": {
                    "type": "str",
                    "description": "Language code (ar, en, auto)",
                    "default": "auto",
                    "required": False
                },
                "title": {
                    "type": "str",
                    "description": "Note title (if not provided, will be auto-generated)",
                    "required": False
                },
                "save_audio": {
                    "type": "bool",
                    "description": "Whether to save the audio recording",
                    "default": True,
                    "required": False
                },
                "audio_format": {
                    "type": "str",
                    "description": "Audio format for saving (wav, flac)",
                    "default": "wav",
                    "required": False
                },
                "generate_summary": {
                    "type": "bool",
                    "description": "Generate a summary of the note",
                    "default": True,
                    "required": False
                },
                "export_format": {
                    "type": "str",
                    "description": "Export format (text, json, markdown)",
                    "default": "markdown",
                    "required": False
                },
                "action": {
                    "type": "str",
                    "description": "Action to perform (create, retrieve, update, delete, list, export)",
                    "default": "create",
                    "required": False
                },
                "note_id": {
                    "type": "str",
                    "description": "Note ID for retrieve, update, delete operations",
                    "required": False
                }
            },
            examples=[
                {
                    "description": "Create a voice note from audio",
                    "parameters": {
                        "audio_data": "<audio_data>",
                        "language": "ar",
                        "save_audio": True
                    }
                },
                {
                    "description": "Create a note from text",
                    "parameters": {
                        "text": "Meeting with team about the new project launch",
                        "title": "Team Meeting",
                        "language": "en"
                    }
                },
                {
                    "description": "Retrieve a specific note",
                    "parameters": {
                        "action": "retrieve",
                        "note_id": "note_12345"
                    }
                }
            ],
            author="Drmusab",
            tags={"productivity", "notes", "voice", "bilingual", "arabic", "english"},
            is_stateful=True,
            timeout_seconds=60.0,
            cache_ttl_seconds=3600
        )

    async def _execute(self, **params) -> SkillResult:
        """Execute the voice note taker skill."""
        action = params.get("action", "create")
        
        try:
            if action == "create":
                result = await self._create_note(**params)
            elif action == "retrieve":
                result = await self._retrieve_note(params.get("note_id"))
            elif action == "update":
                result = await self._update_note(params.get("note_id"), **params)
            elif action == "delete":
                result = await self._delete_note(params.get("note_id"))
            elif action == "list":
                result = await self._list_notes(
                    category=params.get("category"),
                    date_from=params.get("date_from"),
                    date_to=params.get("date_to"),
                    tags=params.get("tags")
                )
            elif action == "export":
                result = await self._export_note(
                    params.get("note_id"),
                    params.get("export_format", "markdown")
                )
            else:
                return SkillResult(
                    success=False,
                    result={"error": f"Unknown action: {action}"},
                    errors=[f"Unknown action: {action}"]
                )
                
            return SkillResult(
                success=True,
                result=result,
                execution_time=result.get("execution_time", 0.0),
                confidence=result.get("confidence", 0.9)
            )
            
        except Exception as e:
            self.logger.error(f"Error in voice note taker: {e}")
            return SkillResult(
                success=False,
                result={"error": str(e)},
                errors=[str(e)]
            )

    async def _create_note(self, **params) -> Dict[str, Any]:
        """Create a new voice note from audio or text."""
        start_time = datetime.now(timezone.utc)
        
        # Get input content (from audio or text)
        audio_data = params.get("audio_data")
        text_input = params.get("text")
        
        if audio_data is not None:
            # Process audio data to text
            language_hint = self._parse_language(params.get("language", "auto"))
            transcription_result = await self.speech_to_text.transcribe(
                audio_data, language=language_hint.value
            )
            
            text = transcription_result.get("text", "")
            detected_language = transcription_result.get("detected_language", language_hint.value)
            duration = transcription_result.get("duration_seconds", 0)
            
            confidence = transcription_result.get("confidence", 0.0)
        elif text_input:
            # Use provided text
            text = text_input
            detected_language = await self._detect_language(text)
            duration = 0
            confidence = 1.0
        else:
            raise ValueError("Either audio_data or text must be provided")
            
        # Generate note ID and title
        note_id = str(uuid.uuid4())
        title = params.get("title") or await self._generate_title(text, detected_language)
        
        # Process text for additional insights
        language = NoteLanguage(detected_language)
        
        # Generate summary if requested
        summary = None
        bullet_points = []
        action_items = []
        
        if params.get("generate_summary", True) and text:
            summary_results = await self._generate_summary(text, language)
            summary = summary_results.get("summary")
            bullet_points = summary_results.get("bullet_points", [])
            action_items = summary_results.get("action_items", [])
        
        # Extract entities and sentiment
        entities = await self._extract_entities(text, language)
        sentiment = await self._analyze_sentiment(text, language)
        
        # Auto-categorize content
        categories = await self._categorize_content(text, entities, language)
        tags = await self._generate_tags(text, entities, language)
        
        # Save audio if requested
        audio_path = None
        audio_format = AudioFormat(params.get("audio_format", "wav"))
        
        if audio_data is not None and params.get("save_audio", True):
            audio_path = await self._save_audio(audio_data, note_id, audio_format)
        
        # Create note object
        note = VoiceNote(
            note_id=note_id,
            title=title,
            content=text,
            summary=summary,
            language=language,
            categories=categories,
            tags=tags,
            created_at=datetime.now(timezone.utc),
            audio_path=audio_path,
            audio_format=audio_format,
            duration_seconds=duration,
            bullet_points=bullet_points,
            action_items=action_items,
            sentiment=sentiment,
            entities=entities,
            metadata={
                "confidence": confidence,
                "source": "audio" if audio_data is not None else "text",
            }
        )
        
        # Save note to storage
        await self._save_note(note)
        
        # Add to cache
        self._add_to_cache(note)
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Prepare result
        return {
            "note_id": note_id,
            "title": title,
            "content": text,
            "summary": summary,
            "bullet_points": bullet_points,
            "action_items": action_items,
            "language": language.value,
            "categories": [cat.value for cat in categories],
            "tags": tags,
            "created_at": note.created_at.isoformat(),
            "audio_saved": audio_path is not None,
            "audio_format": audio_format.value if audio_path else None,
            "duration_seconds": duration,
            "sentiment": sentiment,
            "entities": entities,
            "execution_time": execution_time,
            "confidence": confidence
        }

    async def _retrieve_note(self, note_id: str) -> Dict[str, Any]:
        """Retrieve a note by ID."""
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
                
            note = await self._load_note(note_id)
            self._add_to_cache(note)
            
        return self._note_to_dict(note)

    async def _update_note(self, note_id: str, **params) -> Dict[str, Any]:
        """Update an existing note."""
        if not note_id:
            raise ValueError("Note ID is required")
            
        # Retrieve existing note
        note = await self._load_note(note_id)
        
        # Update fields
        if "title" in params:
            note.title = params["title"]
            
        if "text" in params:
            note.content = params["text"]
            
            # Regenerate insights if content changed
            if params.get("generate_summary", True):
                summary_results = await self._generate_summary(note.content, note.language)
                note.summary = summary_results.get("summary")
                note.bullet_points = summary_results.get("bullet_points", [])
                note.action_items = summary_results.get("action_items", [])
                
            note.entities = await self._extract_entities(note.content, note.language)
            note.sentiment = await self._analyze_sentiment(note.content, note.language)
            note.categories = await self._categorize_content(note.content, note.entities, note.language)
            
            if params.get("regenerate_tags", False):
                new_tags = await self._generate_tags(note.content, note.entities, note.language)
                note.tags = new_tags
                
        if "tags" in params:
            # Append or replace tags
            if params.get("append_tags", False):
                note.tags.extend(params["tags"])
                # Remove duplicates while preserving order
                note.tags = list(dict.fromkeys(note.tags))
            else:
                note.tags = params["tags"]
                
        # Save updated note
        await self._save_note(note)
        self._add_to_cache(note)
        
        return self._note_to_dict(note)

    async def _delete_note(self, note_id: str) -> Dict[str, Any]:
        """Delete a note and its associated audio file."""
        if not note_id:
            raise ValueError("Note ID is required")
            
        # Check if note exists
        note_path = self.notes_dir / f"{note_id}.json"
        if not note_path.exists():
            raise ValueError(f"Note with ID {note_id} not found")
            
        # Load note to get audio path
        note = await self._load_note(note_id)
        
        # Delete audio file if exists
        if note.audio_path:
            audio_path = Path(note.audio_path)
            if audio_path.exists():
                audio_path.unlink()
                
        # Delete note file
        note_path.unlink()
        
        # Remove from cache
        if note_id in self._notes_cache:
            del self._notes_cache[note_id]
            
        return {
            "success": True,
            "note_id": note_id,
            "message": f"Note {note_id} successfully deleted"
        }

    async def _list_notes(
        self, 
        category: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """List notes with optional filtering."""
        notes = []
        
        # Load all notes
        note_files = list(self.notes_dir.glob("*.json"))
        
        # Parse date filters
        from_date = None
        to_date = None
        
        if date_from:
            try:
                from_date = datetime.fromisoformat(date_from)
            except ValueError:
                self.logger.warning(f"Invalid date_from format: {date_from}")
                
        if date_to:
            try:
                to_date = datetime.fromisoformat(date_to)
            except ValueError:
                self.logger.warning(f"Invalid date_to format: {date_to}")
        
        # Process each note file
        for note_file in note_files:
            try:
                with open(note_file, 'r', encoding='utf-8') as f:
                    note_data = json.load(f)
                    
                # Apply filters
                # Filter by category
                if category and category not in [cat.lower() for cat in note_data.get("categories", [])]:
                    continue
                    
                # Filter by date range
                if from_date or to_date:
                    try:
                        note_date = datetime.fromisoformat(note_data.get("created_at"))
                        if from_date and note_date < from_date:
                            continue
                        if to_date and note_date > to_date:
                            continue
                    except (ValueError, TypeError):
                        # Skip notes with invalid dates
                        continue
                
                # Filter by tags
                if tags:
                    note_tags = set(t.lower() for t in note_data.get("tags", []))
                    if not any(tag.lower() in note_tags for tag in tags):
                        continue
                
                # Add summary data
                notes.append({
                    "note_id": note_data.get("note_id"),
                    "title": note_data.get("title"),
                    "created_at": note_data.get("created_at"),
                    "language": note_data.get("language"),
                    "categories": note_data.get("categories", []),
                    "tags": note_data.get("tags", []),
                    "has_audio": bool(note_data.get("audio_path")),
                    "duration_seconds": note_data.get("duration_seconds", 0)
                })
                
            except Exception as e:
                self.logger.error(f"Error processing note file {note_file}: {e}")
        
        # Sort by creation date (newest first)
        notes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return {
            "total_count": len(notes),
            "notes": notes
        }

    async def _export_note(self, note_id: str, format_type: str = "markdown") -> Dict[str, Any]:
        """Export a note in the specified format."""
        if not note_id:
            raise ValueError("Note ID is required")
            
        # Get the note
        note = await self._load_note(note_id)
        
        # Convert to specified format
        format_type = format_type.lower()
        
        if format_type == "json":
            # Already in JSON format
            content = json.dumps(self._note_to_dict(note), ensure_ascii=False, indent=2)
            mime_type = "application/json"
        elif format_type == "markdown":
            content = self._note_to_markdown(note)
            mime_type = "text/markdown"
        elif format_type == "text":
            content = self._note_to_text(note)
            mime_type = "text/plain"
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        # Return formatted content
        return {
            "note_id": note_id,
            "format": format_type,
            "content": content,
            "mime_type": mime_type,
            "filename": f"{note.title.replace(' ', '_')}_{note_id}.{format_type}"
        }

    async def _save_audio(
        self, audio_data: np.ndarray, note_id: str, format_type: AudioFormat
    ) -> str:
        """Save audio data to file."""
        filename = f"{note_id}.{format_type.value}"
        filepath = self.audio_dir / filename
        
        # Save audio file
        save_audio_file(audio_data, str(filepath), format_type.value)
        
        return str(filepath)

    async def _generate_title(self, text: str, language: str) -> str:
        """Generate a title from the note text."""
        # Use bilingual processor to generate title
        max_title_length = 50
        
        if not text:
            return "Untitled Note"
            
        # Basic approach: use first sentence or first few words
        sentences = text.split('.')
        first_sentence = sentences[0].strip()
        
        if len(first_sentence) <= max_title_length:
            title = first_sentence
        else:
            # Use first few words
            words = first_sentence.split()
            title = " ".join(words[:5]) + "..."
            
        if not title:
            title = "Untitled Note"
            
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return f"{title} - {timestamp}"

    async def _generate_summary(self, text: str, language: NoteLanguage) -> Dict[str, Any]:
        """Generate summary, bullet points, and action items."""
        # Use bilingual processor for language-specific processing
        lang_code = language.value
        
        try:
            # Get summary
            summary_prompt = f"Summarize the following text in the same language ({lang_code}):\n\n{text}"
            summary_result = await self.bilingual_processor.process(summary_prompt, lang_code)
            summary = summary_result.get("response", "")
            
            # Generate bullet points
            bullets_prompt = f"Extract 3-5 key points from this text as bullet points in the same language ({lang_code}):\n\n{text}"
            bullets_result = await self.bilingual_processor.process(bullets_prompt, lang_code)
            
            # Parse bullet points - handle various formats
            bullet_text = bullets_result.get("response", "")
            bullet_points = []
            
            for line in bullet_text.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    bullet_points.append(line[1:].strip())
                elif line and not any(line.startswith(c) for c in ['#', '>', '=']):
                    bullet_points.append(line)
            
            # Extract action items
            actions_prompt = f"Extract action items or tasks from this text in the same language ({lang_code}):\n\n{text}"
            actions_result = await self.bilingual_processor.process(actions_prompt, lang_code)
            
            # Parse action items
            action_text = actions_result.get("response", "")
            action_items = []
            
            for line in action_text.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    action_items.append(line[1:].strip())
                elif line and not any(line.startswith(c) for c in ['#', '>', '=']):
                    action_items.append(line)
            
            return {
                "summary": summary,
                "bullet_points": bullet_points,
                "action_items": action_items
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return {
                "summary": "",
                "bullet_points": [],
                "action_items": []
            }

    async def _extract_entities(self, text: str, language: NoteLanguage) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        try:
            entities_result = await self.entity_extractor.extract_entities(
                text, language=language.value
            )
            return entities_result.get("entities", [])
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []

    async def _analyze_sentiment(self, text: str, language: NoteLanguage) -> Optional[str]:
        """Analyze sentiment of the text."""
        try:
            sentiment_result = await self.sentiment_analyzer.analyze(
                text, language=language.value
            )
            return sentiment_result.get("sentiment")
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return None

    async def _categorize_content(
        self, text: str, entities: List[Dict[str, Any]], language: NoteLanguage
    ) -> Set[NoteCategory]:
        """Auto-categorize content based on text analysis."""
        categories = set()
        lang_code = language.value
        
        try:
            # Use bilingual processor for categorization
            category_prompt = f"""
            Categorize the following text into one or more of these categories:
            - TASK: Contains action items, to-dos, or assignments
            - IDEA: Contains creative concepts, innovations, or proposals
            - QUOTE: Contains direct quotations from people
            - MEMORY: Contains personal memories or recollections
            - EXPERIENCE: Contains descriptions of experiences or events
            - MEETING: Contains meeting notes or discussion points
            - CLASS: Contains educational or learning content
            - GENERAL: General information that doesn't fit other categories
            
            Text: {text}
            
            Return only the category names, comma-separated.
            """
            
            result = await self.bilingual_processor.process(category_prompt, lang_code)
            category_text = result.get("response", "GENERAL")
            
            # Parse categories
            for cat_str in category_text.split(','):
                cat_str = cat_str.strip().upper()
                try:
                    # Match with category enum
                    category = NoteCategory[cat_str]
                    categories.add(category)
                except (KeyError, ValueError):
                    # No matching category, skip
                    pass
            
            # Ensure at least one category
            if not categories:
                categories.add(NoteCategory.GENERAL)
                
        except Exception as e:
            self.logger.error(f"Error categorizing content: {e}")
            categories.add(NoteCategory.GENERAL)
            
        return categories

    async def _generate_tags(
        self, text: str, entities: List[Dict[str, Any]], language: NoteLanguage
    ) -> List[str]:
        """Generate tags based on content and entities."""
        tags = []
        lang_code = language.value
        
        try:
            # Extract tags from entities
            for entity in entities:
                if entity.get("type") in ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]:
                    tags.append(entity.get("text"))
                    
            # Use bilingual processor for additional tags
            tags_prompt = f"Generate 3-5 relevant tags or keywords for this text in the same language ({lang_code}):\n\n{text}"
            result = await self.bilingual_processor.process(tags_prompt, lang_code)
            
            # Parse tags
            tags_text = result.get("response", "")
            for line in tags_text.split('\n'):
                line = line.strip()
                if line.startswith('#'):
                    # Handle hashtag format
                    tags.append(line[1:].strip())
                elif ',' in line:
                    # Handle comma-separated format
                    for tag in line.split(','):
                        tag = tag.strip()
                        if tag and len(tag) > 1:
                            tags.append(tag)
                elif line and not any(line.startswith(c) for c in ['•', '-', '*']):
                    tags.append(line)
                    
            # Remove duplicates while preserving order
            unique_tags = []
            seen = set()
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower not in seen and tag:
                    seen.add(tag_lower)
                    unique_tags.append(tag)
                    
            return unique_tags[:10]  # Limit to 10 tags
            
        except Exception as e:
            self.logger.error(f"Error generating tags: {e}")
            return []

    async def _detect_language(self, text: str) -> str:
        """Detect language of the text."""
        try:
            # Use bilingual processor to detect language
            detection_result = await self.bilingual_processor.detect_language(text)
            detected_lang = detection_result.get("language", "en")
            
            # Map to supported languages
            if detected_lang.startswith("ar"):
                return NoteLanguage.ARABIC.value
            else:
                return NoteLanguage.ENGLISH.value
                
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return NoteLanguage.ENGLISH.value

    async def _save_note(self, note: VoiceNote) -> None:
        """Save note to storage."""
        note_path = self.notes_dir / f"{note.note_id}.json"
        
        try:
            with open(note_path, 'w', encoding='utf-8') as f:
                json.dump(self._note_to_dict(note), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving note: {e}")
            raise

    async def _load_note(self, note_id: str) -> VoiceNote:
        """Load note from storage."""
        note_path = self.notes_dir / f"{note_id}.json"
        
        if not note_path.exists():
            raise ValueError(f"Note with ID {note_id} not found")
            
        try:
            with open(note_path, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
                
            # Convert dictionary back to VoiceNote
            return self._dict_to_note(note_data)
            
        except Exception as e:
            self.logger.error(f"Error loading note: {e}")
            raise

    def _note_to_dict(self, note: VoiceNote) -> Dict[str, Any]:
        """Convert VoiceNote to dictionary."""
        return {
            "note_id": note.note_id,
            "title": note.title,
            "content": note.content,
            "summary": note.summary,
            "language": note.language.value,
            "categories": [cat.value for cat in note.categories],
            "tags": note.tags,
            "created_at": note.created_at.isoformat(),
            "audio_path": note.audio_path,
            "audio_format": note.audio_format.value,
            "duration_seconds": note.duration_seconds,
            "metadata": note.metadata,
            "bullet_points": note.bullet_points,
            "action_items": note.action_items,
            "sentiment": note.sentiment,
            "entities": note.entities
        }

    def _dict_to_note(self, data: Dict[str, Any]) -> VoiceNote:
        """Convert dictionary to VoiceNote."""
        try:
            # Parse created_at
            if isinstance(data.get("created_at"), str):
                created_at = datetime.fromisoformat(data["created_at"])
            else:
                created_at = datetime.now(timezone.utc)
                
            # Parse language
            try:
                language = NoteLanguage(data.get("language", "en"))
            except ValueError:
                language = NoteLanguage.ENGLISH
                
            # Parse audio format
            try:
                audio_format = AudioFormat(data.get("audio_format", "wav"))
            except ValueError:
                audio_format = AudioFormat.WAV
                
            # Parse categories
            categories = set()
            for cat_str in data.get("categories", []):
                try:
                    categories.add(NoteCategory(cat_str))
                except ValueError:
                    pass
                    
            if not categories:
                categories.add(NoteCategory.GENERAL)
                
            return VoiceNote(
                note_id=data.get("note_id", ""),
                title=data.get("title", ""),
                content=data.get("content", ""),
                summary=data.get("summary"),
                language=language,
                categories=categories,
                tags=data.get("tags", []),
                created_at=created_at,
                audio_path=data.get("audio_path"),
                audio_format=audio_format,
                duration_seconds=data.get("duration_seconds", 0.0),
                metadata=data.get("metadata", {}),
                bullet_points=data.get("bullet_points", []),
                action_items=data.get("action_items", []),
                sentiment=data.get("sentiment"),
                entities=data.get("entities", [])
            )
            
        except Exception as e:
            self.logger.error(f"Error converting dict to note: {e}")
            raise

    def _note_to_markdown(self, note: VoiceNote) -> str:
        """Convert note to Markdown format."""
        md_content = []
        
        # Title and metadata
        md_content.append(f"# {note.title}")
        md_content.append(f"**Date:** {note.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append(f"**Language:** {note.language.value}")
        md_content.append(f"**Categories:** {', '.join(cat.value for cat in note.categories)}")
        if note.tags:
            md_content.append(f"**Tags:** {', '.join(note.tags)}")
        if note.audio_path:
            md_content.append(f"**Audio:** [{note.audio_format.value.upper()} Recording]({note.audio_path})")
            md_content.append(f"**Duration:** {note.duration_seconds:.1f} seconds")
        if note.sentiment:
            md_content.append(f"**Sentiment:** {note.sentiment}")
            
        # Content
        md_content.append("\n## Content")
        md_content.append(note.content)
        
        # Summary
        if note.summary:
            md_content.append("\n## Summary")
            md_content.append(note.summary)
            
        # Bullet Points
        if note.bullet_points:
            md_content.append("\n## Key Points")
            for point in note.bullet_points:
                md_content.append(f"* {point}")
                
        # Action Items
        if note.action_items:
            md_content.append("\n## Action Items")
            for item in note.action_items:
                md_content.append(f"- [ ] {item}")
                
        # Entities
        if note.entities:
            md_content.append("\n## Entities")
            for entity in note.entities:
                md_content.append(f"* **{entity.get('text')}** ({entity.get('type')})")
                
        return "\n\n".join(md_content)

    def _note_to_text(self, note: VoiceNote) -> str:
        """Convert note to plain text format."""
        text_content = []
        
        # Title and metadata
        text_content.append(f"{note.title}")
        text_content.append(f"Date: {note.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        text_content.append(f"Language: {note.language.value}")
        text_content.append(f"Categories: {', '.join(cat.value for cat in note.categories)}")
        if note.tags:
            text_content.append(f"Tags: {', '.join(note.tags)}")
        if note.audio_path:
            text_content.append(f"Audio: {note.audio_path} ({note.duration_seconds:.1f}s)")
            
        # Content
        text_content.append("\nCONTENT:")
        text_content.append(note.content)
        
        # Summary
        if note.summary:
            text_content.append("\nSUMMARY:")
            text_content.append(note.summary)
            
        # Bullet Points
        if note.bullet_points:
            text_content.append("\nKEY POINTS:")
            for i, point in enumerate(note.bullet_points, 1):
                text_content.append(f"{i}. {point}")
                
        # Action Items
        if note.action_items:
            text_content.append("\nACTION ITEMS:")
            for i, item in enumerate(note.action_items, 1):
                text_content.append(f"{i}. {item}")
                
        return "\n".join(text_content)

    def _add_to_cache(self, note: VoiceNote) -> None:
        """Add note to in-memory cache."""
        self._notes_cache[note.note_id] = note
        
        # Maintain cache size
        if len(self._notes_cache) > self.max_cache_size:
            # Remove oldest note (based on creation time)
            oldest_id = min(
                self._notes_cache.keys(),
                key=lambda k: self._notes_cache[k].created_at
            )
            del self._notes_cache[oldest_id]

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        os.makedirs(self.notes_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

    def _parse_language(self, language_str: str) -> NoteLanguage:
        """Parse language string to NoteLanguage enum."""
        if not language_str:
            return NoteLanguage.AUTO
            
        language_str = language_str.lower()
        
        if language_str in ("ar", "arabic", "العربية"):
            return NoteLanguage.ARABIC
        elif language_str in ("en", "english", "الإنجليزية"):
            return NoteLanguage.ENGLISH
        else:
            return NoteLanguage.AUTO
