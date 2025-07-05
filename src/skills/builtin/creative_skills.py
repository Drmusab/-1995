"""
Advanced Creative Skills for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:21:47 UTC

This module provides comprehensive creative capabilities for the AI assistant,
including story generation, poetry writing, creative problem solving, 
character development, visual art guidance, musical composition assistance,
and other creative content generation features.
"""

from typing import Dict, Any, List, Optional, Union, Tuple, Set, Callable
import asyncio
import uuid
import random
from datetime import datetime, timezone
from pathlib import Path
import logging
import json
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SkillExecutionStarted, SkillExecutionCompleted, SkillExecutionFailed,
    LearningEventOccurred, UserFeedbackReceived
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container

# Memory and context
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.memory.core_memory.memory_types import WorkingMemory, EpisodicMemory, SemanticMemory

# Processing components
from src.processing.natural_language.language_chain import LanguageChain
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
from src.processing.natural_language.entity_extractor import EntityExtractor

# Reasoning
from src.reasoning.knowledge_graph import KnowledgeGraph
from src.reasoning.logic_engine import LogicEngine

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner

# Integrations
from src.integrations.llm.model_router import ModelRouter

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class CreativeFormat(Enum):
    """Creative content formats."""
    STORY = "story"
    POEM = "poem"
    SCRIPT = "script"
    SONG = "song"
    DIALOGUE = "dialogue"
    ESSAY = "essay"
    CHARACTER_PROFILE = "character_profile"
    SETTING_DESCRIPTION = "setting_description"
    PLOT_OUTLINE = "plot_outline"
    METAPHOR = "metaphor"
    CONCEPT_ART = "concept_art"
    MUSICAL_COMPOSITION = "musical_composition"
    JOKE = "joke"
    RIDDLE = "riddle"


class CreativeGenre(Enum):
    """Creative genres for content generation."""
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    HORROR = "horror"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    THRILLER = "thriller"
    HISTORICAL = "historical"
    COMEDY = "comedy"
    DRAMA = "drama"
    ADVENTURE = "adventure"
    CYBERPUNK = "cyberpunk"
    STEAMPUNK = "steampunk"
    MAGICAL_REALISM = "magical_realism"
    DYSTOPIAN = "dystopian"
    UTOPIAN = "utopian"
    SURREALISM = "surrealism"
    FOLKLORE = "folklore"
    MYTHOLOGY = "mythology"


class ToneStyle(Enum):
    """Writing tones and styles."""
    FORMAL = "formal"
    INFORMAL = "informal"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    POETIC = "poetic"
    TECHNICAL = "technical"
    DRAMATIC = "dramatic"
    INSPIRATIONAL = "inspirational"
    MELANCHOLIC = "melancholic"
    WHIMSICAL = "whimsical"
    SUSPENSEFUL = "suspenseful"
    ROMANTIC = "romantic"
    EDUCATIONAL = "educational"
    SATIRICAL = "satirical"
    PHILOSOPHICAL = "philosophical"


class CreativeSkill(ABC):
    """Abstract base class for all creative skills."""
    
    def __init__(self, container: Container):
        """
        Initialize the creative skill.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Core components
        self.model_router = container.get(ModelRouter)
        self.language_chain = container.get(LanguageChain)
        self.sentiment_analyzer = container.get(SentimentAnalyzer)
        self.entity_extractor = container.get(EntityExtractor)
        
        # Memory and context
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)
        self.working_memory = container.get(WorkingMemory)
        self.episodic_memory = container.get(EpisodicMemory)
        self.semantic_memory = container.get(SemanticMemory)
        
        # Reasoning components
        self.knowledge_graph = container.get(KnowledgeGraph)
        self.logic_engine = container.get(LogicEngine)
        
        # Learning components
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Register metrics
        if self.metrics:
            self.metrics.register_counter(f"skill_{self.get_skill_id()}_executions_total")
            self.metrics.register_histogram(f"skill_{self.get_skill_id()}_execution_time_seconds")
            self.metrics.register_counter(f"skill_{self.get_skill_id()}_errors_total")
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the creative skill."""
        pass
    
    @abstractmethod
    def get_skill_id(self) -> str:
        """Get the unique skill identifier."""
        pass
    
    @abstractmethod
    def get_skill_description(self) -> str:
        """Get the skill description."""
        pass
    
    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the skill parameters schema."""
        return {}
    
    def get_skill_examples(self) -> List[Dict[str, Any]]:
        """Get examples of skill usage."""
        return []
    
    def get_skill_category(self) -> str:
        """Get the skill category."""
        return "creative"
    
    async def _track_execution(self, execution_id: str, **kwargs) -> None:
        """Track skill execution in memory for learning."""
        try:
            # Store execution data
            execution_data = {
                "skill_id": self.get_skill_id(),
                "execution_id": execution_id,
                "timestamp": datetime.now(timezone.utc),
                "parameters": kwargs,
                "category": self.get_skill_category()
            }
            
            await self.episodic_memory.store(execution_data)
            
            # Update learning system
            await self.continual_learner.learn_from_skill_execution(execution_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to track skill execution: {str(e)}")


class StoryGenerator(CreativeSkill):
    """Advanced story generation skill."""
    
    def get_skill_id(self) -> str:
        return "story_generator"
    
    def get_skill_description(self) -> str:
        return "Generates creative stories in various genres and formats with customizable parameters."
    
    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "theme": {
                "type": "string",
                "description": "Main theme or topic of the story",
                "required": True
            },
            "genre": {
                "type": "string",
                "description": "Genre of the story",
                "enum": [g.value for g in CreativeGenre],
                "default": CreativeGenre.FANTASY.value
            },
            "tone": {
                "type": "string",
                "description": "Tone and style of the story",
                "enum": [t.value for t in ToneStyle],
                "default": ToneStyle.DRAMATIC.value
            },
            "length": {
                "type": "string",
                "description": "Length of the story",
                "enum": ["short", "medium", "long"],
                "default": "medium"
            },
            "characters": {
                "type": "array",
                "description": "Character descriptions or names to include",
                "items": {"type": "string"},
                "default": []
            },
            "setting": {
                "type": "string",
                "description": "Setting or world description",
                "required": False
            },
            "include_dialogue": {
                "type": "boolean",
                "description": "Whether to include dialogue",
                "default": True
            },
            "pov": {
                "type": "string",
                "description": "Point of view",
                "enum": ["first_person", "second_person", "third_person"],
                "default": "third_person"
            }
        }
    
    def get_skill_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "parameters": {
                    "theme": "time travel paradox",
                    "genre": "science_fiction",
                    "tone": "philosophical",
                    "length": "medium",
                    "characters": ["Dr. Eleanor Voss, a quantum physicist", "Marcus, her younger self"]
                },
                "description": "Generate a philosophical sci-fi story about time travel paradoxes"
            },
            {
                "parameters": {
                    "theme": "unexpected friendship",
                    "genre": "fantasy",
                    "tone": "whimsical",
                    "length": "short",
                    "setting": "floating islands connected by rainbow bridges"
                },
                "description": "Create a whimsical fantasy short story set in a world of floating islands"
            }
        ]
    
    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a creative story based on provided parameters.
        
        Args:
            theme: Main theme or topic
            genre: Story genre
            tone: Writing tone and style
            length: Desired story length
            characters: Character descriptions
            setting: Story setting description
            include_dialogue: Whether to include dialogue
            pov: Point of view
            
        Returns:
            Generated story and metadata
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Track metrics
        if self.metrics:
            self.metrics.increment(f"skill_{self.get_skill_id()}_executions_total")
        
        # Emit execution started event
        await self.event_bus.emit(SkillExecutionStarted(
            skill_id=self.get_skill_id(),
            execution_id=execution_id,
            parameters=kwargs
        ))
        
        try:
            with self.tracer.trace("story_generation") if self.tracer else None:
                # Extract parameters with defaults
                theme = kwargs.get("theme", "adventure")
                genre = kwargs.get("genre", CreativeGenre.FANTASY.value)
                tone = kwargs.get("tone", ToneStyle.DRAMATIC.value)
                length = kwargs.get("length", "medium")
                characters = kwargs.get("characters", [])
                setting = kwargs.get("setting", "")
                include_dialogue = kwargs.get("include_dialogue", True)
                pov = kwargs.get("pov", "third_person")
                
                # Determine target word count based on length
                word_count = {
                    "short": 500,
                    "medium": 1500,
                    "long": 3000
                }.get(length, 1500)
                
                # Retrieve relevant memory and knowledge
                relevant_memories = await self._retrieve_relevant_memories(theme, genre)
                knowledge_elements = await self._retrieve_knowledge(theme, genre)
                
                # Generate plot structure
                plot_structure = await self._generate_plot_structure(
                    theme, genre, tone, characters, setting
                )
                
                # Construct prompt for story generation
                prompt = self._construct_story_prompt(
                    theme, genre, tone, word_count, characters, 
                    setting, include_dialogue, pov, plot_structure
                )
                
                # Generate story
                model_parameters = {
                    "temperature": 0.85,
                    "max_tokens": min(4000, word_count * 3),
                    "top_p": 0.92,
                    "frequency_penalty": 0.5,
                    "presence_penalty": 0.5
                }
                
                story = await self.model_router.generate_text(
                    prompt, model="creative", parameters=model_parameters
                )
                
                # Process story to ensure it matches requirements
                processed_story = await self._process_generated_story(
                    story, theme, genre, tone, word_count
                )
                
                # Extract story elements for return
                story_elements = await self._extract_story_elements(processed_story)
                
                # Calculate execution time
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Track execution for learning
                await self._track_execution(
                    execution_id,
                    theme=theme,
                    genre=genre,
                    tone=tone,
                    length=length,
                    execution_time=execution_time
                )
                
                # Emit completion event
                await self.event_bus.emit(SkillExecutionCompleted(
                    skill_id=self.get_skill_id(),
                    execution_id=execution_id,
                    execution_time=execution_time
                ))
                
                if self.metrics:
                    self.metrics.record(f"skill_{self.get_skill_id()}_execution_time_seconds", execution_time)
                
                # Return results
                return {
                    "story": processed_story,
                    "title": story_elements.get("title", f"A {genre} Tale"),
                    "word_count": len(processed_story.split()),
                    "elements": story_elements,
                    "plot_structure": plot_structure,
                    "genre": genre,
                    "tone": tone,
                    "execution_time": execution_time,
                    "execution_id": execution_id
                }
                
        except Exception as e:
            # Handle errors
            if self.metrics:
                self.metrics.increment(f"skill_{self.get_skill_id()}_errors_total")
            
            await self.event_bus.emit(SkillExecutionFailed(
                skill_id=self.get_skill_id(),
                execution_id=execution_id,
                error_message=str(e)
            ))
            
            self.logger.error(f"Story generation failed: {str(e)}")
            raise
    
    async def _retrieve_relevant_memories(self, theme: str, genre: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for story generation."""
        # Construct query for semantic memory
        query = f"story {theme} {genre}"
        
        # Retrieve memories
        memories = await self.semantic_memory.retrieve_relevant(query, limit=5)
        return memories
    
    async def _retrieve_knowledge(self, theme: str, genre: str) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge for story elements."""
        # Query knowledge graph
        elements = await self.knowledge_graph.query(
            [theme, genre], relationship_types=["related_to", "part_of"]
        )
        return elements
    
    async def _generate_plot_structure(
        self, theme: str, genre: str, tone: str, 
        characters: List[str], setting: str
    ) -> Dict[str, Any]:
        """Generate plot structure for the story."""
        # Basic plot structure template
        plot_structure = {
            "exposition": "",
            "rising_action": [],
            "climax": "",
            "falling_action": [],
            "resolution": ""
        }
        
        # Construct prompt for plot generation
        prompt = (
            f"Generate a plot structure for a {tone} {genre} story about {theme}. "
            f"Include these characters: {', '.join(characters) if characters else 'new characters'} "
            f"Setting: {setting if setting else 'appropriate setting for the genre'}. "
            f"Include exposition, rising action events, climax, falling action, and resolution."
        )
        
        # Generate plot
        plot_text = await self.model_router.generate_text(
            prompt, model="creative", parameters={"temperature": 0.7}
        )
        
        # Parse plot sections
        sections = plot_text.split("\n\n")
        if len(sections) >= 5:
            plot_structure["exposition"] = sections[0].replace("Exposition: ", "").strip()
            plot_structure["rising_action"] = [
                event.strip() for event in sections[1].replace("Rising Action: ", "").split(". ") if event.strip()
            ]
            plot_structure["climax"] = sections[2].replace("Climax: ", "").strip()
            plot_structure["falling_action"] = [
                event.strip() for event in sections[3].replace("Falling Action: ", "").split(". ") if event.strip()
            ]
            plot_structure["resolution"] = sections[4].replace("Resolution: ", "").strip()
        else:
            # Simple parsing fallback
            for section in sections:
                if "exposition" in section.lower():
                    plot_structure["exposition"] = section.split(":", 1)[1].strip() if ":" in section else section
                elif "rising action" in section.lower():
                    events = section.split(":", 1)[1].strip() if ":" in section else section
                    plot_structure["rising_action"] = [e.strip() for e in events.split(". ") if e.strip()]
                elif "climax" in section.lower():
                    plot_structure["climax"] = section.split(":", 1)[1].strip() if ":" in section else section
                elif "falling action" in section.lower():
                    events = section.split(":", 1)[1].strip() if ":" in section else section
                    plot_structure["falling_action"] = [e.strip() for e in events.split(". ") if e.strip()]
                elif "resolution" in section.lower():
                    plot_structure["resolution"] = section.split(":", 1)[1].strip() if ":" in section else section
        
        return plot_structure
    
    def _construct_story_prompt(
        self, theme: str, genre: str, tone: str, word_count: int,
        characters: List[str], setting: str, include_dialogue: bool, 
        pov: str, plot_structure: Dict[str, Any]
    ) -> str:
        """Construct detailed prompt for story generation."""
        # Format plot structure for inclusion in prompt
        plot_summary = (
            f"Exposition: {plot_structure['exposition']}\n"
            f"Rising Action: {'. '.join(plot_structure['rising_action'])}\n"
            f"Climax: {plot_structure['climax']}\n"
            f"Falling Action: {'. '.join(plot_structure['falling_action'])}\n"
            f"Resolution: {plot_structure['resolution']}"
        )
        
        # Detailed prompt construction
        prompt = (
            f"Write a compelling {genre} story with a {tone} tone about {theme}. "
            f"Target length: approximately {word_count} words.\n\n"
            
            f"Story setting: {setting if setting else 'an appropriate setting for the theme and genre'}\n\n"
            
            f"Characters: {', '.join(characters) if characters else 'Create interesting characters appropriate for the story'}\n\n"
            
            f"Point of view: {pov.replace('_', ' ')}\n\n"
            
            f"{'Include natural, meaningful dialogue that develops characters and advances the plot.' if include_dialogue else 'Minimize dialogue and focus on descriptive narrative.'}\n\n"
            
            f"Follow this plot structure:\n{plot_summary}\n\n"
            
            f"Ensure the story has a compelling beginning that hooks the reader, character development, "
            f"vivid descriptions, thematic depth, and a satisfying ending. "
            f"Be creative, evocative, and authentic to the {genre} genre with a distinct {tone} voice."
        )
        
        return prompt
    
    async def _process_generated_story(
        self, story: str, theme: str, genre: str, tone: str, target_word_count: int
    ) -> str:
        """Process and refine the generated story."""
        current_word_count = len(story.split())
        
        # Check if length needs significant adjustment
        if abs(current_word_count - target_word_count) / target_word_count > 0.3:
            # If too long or too short, request adjustment
            direction = "expand" if current_word_count < target_word_count else "condense"
            
            adjustment_prompt = (
                f"Please {direction} the following {genre} story to approximately {target_word_count} words "
                f"while maintaining the {tone} tone and theme of {theme}. "
                f"Preserve the essential plot elements and character development.\n\n"
                f"Story: {story}"
            )
            
            # Regenerate with adjusted length
            story = await self.model_router.generate_text(
                adjustment_prompt, model="creative", parameters={"temperature": 0.7}
            )
        
        # Clean up formatting
        story = story.strip()
        
        # Ensure proper title formatting if included
        if story.startswith("#"):
            title_line, rest = story.split("\n", 1)
            story = title_line + "\n\n" + rest.strip()
        
        return story
    
    async def _extract_story_elements(self, story: str) -> Dict[str, Any]:
        """Extract key elements from the generated story for metadata."""
        # Extract title if present
        title = "Untitled Story"
        if story.startswith("#"):
            title_line = story.split("\n", 1)[0]
            title = title_line.replace("#", "").strip()
        
        # Extract other elements
        extraction_prompt = (
            f"Extract the following elements from this story:\n"
            f"1. Main characters (with brief descriptions)\n"
            f"2. Setting\n"
            f"3. Main conflict\n"
            f"4. Key themes\n"
            f"5. Narrative style\n\n"
            f"Story: {story[:2000]}..."  # Use beginning for efficiency
        )
        
        extraction_text = await self.model_router.generate_text(
            extraction_prompt, model="analysis", parameters={"temperature": 0.3}
        )
        
        # Parse extraction results
        elements = {
            "title": title,
            "characters": [],
            "setting": "",
            "conflict": "",
            "themes": [],
            "style": ""
        }
        
        current_section = None
        for line in extraction_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "character" in line.lower() and ":" in line:
                current_section = "characters"
                continue
            elif "setting" in line.lower() and ":" in line:
                current_section = "setting"
                continue
            elif "conflict" in line.lower() and ":" in line:
                current_section = "conflict"
                continue
            elif "theme" in line.lower() and ":" in line:
                current_section = "themes"
                continue
            elif "style" in line.lower() and ":" in line or "narrative" in line.lower() and ":" in line:
                current_section = "style"
                continue
                
            if current_section == "characters":
                if line.startswith("-") or line.startswith("*"):
                    elements["characters"].append(line[1:].strip())
            elif current_section == "setting":
                elements["setting"] += line + " "
            elif current_section == "conflict":
                elements["conflict"] += line + " "
            elif current_section == "themes":
                if line.startswith("-") or line.startswith("*"):
                    elements["themes"].append(line[1:].strip())
                else:
                    themes = [t.strip() for t in line.split(",")]
                    elements["themes"].extend(themes)
            elif current_section == "style":
                elements["style"] += line + " "
        
        # Clean up
        elements["setting"] = elements["setting"].strip()
        elements["conflict"] = elements["conflict"].strip()
        elements["style"] = elements["style"].strip()
        
        return elements


class PoetryGenerator(CreativeSkill):
    """Advanced poetry generation skill."""
    
    def get_skill_id(self) -> str:
        return "poetry_generator"
    
    def get_skill_description(self) -> str:
        return "Generates evocative poetry in various styles, forms, and themes."
    
    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "theme": {
                "type": "string",
                "description": "Theme or subject of the poem",
                "required": True
            },
            "form": {
                "type": "string",
                "description": "Poetic form",
                "enum": ["free_verse", "sonnet", "haiku", "limerick", "ballad", "villanelle", "acrostic", "blank_verse"],
                "default": "free_verse"
            },
            "tone": {
                "type": "string",
                "description": "Emotional tone of the poem",
                "enum": [t.value for t in ToneStyle if t in [ToneStyle.MELANCHOLIC, ToneStyle.INSPIRATIONAL, 
                                                            ToneStyle.ROMANTIC, ToneStyle.PHILOSOPHICAL,
                                                            ToneStyle.WHIMSICAL, ToneStyle.DRAMATIC]],
                "default": ToneStyle.POETIC.value
            },
            "style": {
                "type": "string",
                "description": "Stylistic approach",
                "enum": ["modern", "classical", "experimental", "minimalist", "ornate", "concrete"],
                "default": "modern"
            },
            "acrostic_word": {
                "type": "string",
                "description": "Word for acrostic poem (only needed for acrostic form)",
                "required": False
            },
            "length": {
                "type": "string",
                "description": "Approximate length of the poem",
                "enum": ["short", "medium", "long"],
                "default": "medium"
            },
            "include_title": {
                "type": "boolean",
                "description": "Whether to generate a title for the poem",
                "default": True
            }
        }
    
    def get_skill_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "parameters": {
                    "theme": "autumn leaves",
                    "form": "haiku",
                    "tone": "melancholic",
                    "style": "classical"
                },
                "description": "Generate a classical haiku about autumn leaves with a melancholic tone"
            },
            {
                "parameters": {
                    "theme": "digital connection",
                    "form": "free_verse",
                    "tone": "philosophical",
                    "style": "experimental",
                    "length": "medium"
                },
                "description": "Create an experimental free verse poem about digital connection with a philosophical tone"
            },
            {
                "parameters": {
                    "theme": "courage",
                    "form": "acrostic",
                    "acrostic_word": "BRAVE",
                    "tone": "inspirational"
                },
                "description": "Generate an inspirational acrostic poem using the word 'BRAVE' about courage"
            }
        ]
    
    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Generate poetry based on provided parameters.
        
        Args:
            theme: Theme or subject of the poem
            form: Poetic form (free_verse, sonnet, haiku, etc.)
            tone: Emotional tone
            style: Stylistic approach
            acrostic_word: Word for acrostic poem
            length: Approximate length
            include_title: Whether to generate a title
            
        Returns:
            Generated poem and metadata
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Track metrics
        if self.metrics:
            self.metrics.increment(f"skill_{self.get_skill_id()}_executions_total")
        
        # Emit execution started event
        await self.event_bus.emit(SkillExecutionStarted(
            skill_id=self.get_skill_id(),
            execution_id=execution_id,
            parameters=kwargs
        ))
        
        try:
            with self.tracer.trace("poetry_generation") if self.tracer else None:
                # Extract parameters with defaults
                theme = kwargs.get("theme", "nature")
                form = kwargs.get("form", "free_verse")
                tone = kwargs.get("tone", ToneStyle.POETIC.value)
                style = kwargs.get("style", "modern")
                acrostic_word = kwargs.get("acrostic_word", "")
                length = kwargs.get("length", "medium")
                include_title = kwargs.get("include_title", True)
                
                # Validate acrostic word if needed
                if form == "acrostic" and not acrostic_word:
                    acrostic_word = theme.split()[0].upper() if theme else "POEM"
                
                # Get form-specific guidelines
                form_guidelines = self._get_form_guidelines(form, acrostic_word)
                
                # Determine target length
                target_length = self._determine_poem_length(form, length)
                
                # Construct poetry prompt
                prompt = self._construct_poetry_prompt(
                    theme, form, tone, style, form_guidelines, target_length, include_title
                )
                
                # Generate poem
                model_parameters = {
                    "temperature": 0.9,  # Higher creativity for poetry
                    "max_tokens": 2000,
                    "top_p": 0.95,
                    "frequency_penalty": 0.6,
                    "presence_penalty": 0.7
                }
                
                poem = await self.model_router.generate_text(
                    prompt, model="creative", parameters=model_parameters
                )
                
                # Process the poem
                processed_poem, title = await self._process_poem(poem, form, include_title)
                
                # Analyze poetic elements
                poetic_elements = await self._analyze_poetic_elements(processed_poem, form)
                
                # Calculate execution time
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Track execution for learning
                await self._track_execution(
                    execution_id,
                    theme=theme,
                    form=form,
                    tone=tone,
                    style=style,
                    execution_time=execution_time
                )
                
                # Emit completion event
                await self.event_bus.emit(SkillExecutionCompleted(
                    skill_id=self.get_skill_id(),
                    execution_id=execution_id,
                    execution_time=execution_time
                ))
                
                if self.metrics:
                    self.metrics.record(f"skill_{self.get_skill_id()}_execution_time_seconds", execution_time)
                
                # Return results
                return {
                    "poem": processed_poem,
                    "title": title,
                    "form": form,
                    "theme": theme,
                    "tone": tone,
                    "style": style,
                    "poetic_elements": poetic_elements,
                    "execution_time": execution_time,
                    "execution_id": execution_id
                }
                
        except Exception as e:
            # Handle errors
            if self.metrics:
                self.metrics.increment(f"skill_{self.get_skill_id()}_errors_total")
            
            await self.event_bus.emit(SkillExecutionFailed(
                skill_id=self.get_skill_id(),
                execution_id=execution_id,
                error_message=str(e)
            ))
            
            self.logger.error(f"Poetry generation failed: {str(e)}")
            raise
    
    def _get_form_guidelines(self, form: str, acrostic_word: str = "") -> str:
        """Get specific guidelines for different poetic forms."""
        guidelines = {
            "sonnet": (
                "14 lines with a specific rhyme scheme. "
                "Traditional English sonnets use iambic pentameter with an ABABCDCDEFEFGG rhyme scheme. "
                "Develop a single theme, with a turn (volta) between the octave and sestet."
            ),
            "haiku": (
                "Three lines with a 5-7-5 syllable pattern. "
                "Focus on nature imagery and a seasonal reference. "
                "Create a moment of insight or awareness."
            ),
            "limerick": (
                "Five lines with an AABBA rhyme scheme. "
                "Lines 1, 2, and 5 have 7-10 syllables with three stresses. "
                "Lines 3 and 4 have 5-7 syllables with two stresses. "
                "Typically humorous or nonsensical."
            ),
            "ballad": (
                "Narrative poem with four-line stanzas (quatrains). "
                "Typically has an ABCB rhyme scheme. "
                "Tell a story with dramatic elements."
            ),
            "villanelle": (
                "19 lines with five tercets and a concluding quatrain. "
                "Two repeating rhymes and two refrains. "
                "The first and third lines of the first tercet repeat alternately at the end of each subsequent tercet and together at the end of the quatrain."
            ),
            "acrostic": (
                f"Poem where the first letter of each line spells out the word '{acrostic_word}'. "
                f"Each line should begin with the corresponding letter of '{acrostic_word}'."
            ),
            "blank_verse": (
                "Unrhymed iambic pentameter. "
                "Each line has 10 syllables with a stress pattern of unstressed-stressed. "
                "Focus on rhythm and meter rather than rhyme."
            ),
            "free_verse": (
                "No fixed meter, rhyme scheme, or pattern. "
                "Focus on natural speech rhythms and creative expression. "
                "Use line breaks and stanza divisions for emphasis and meaning."
            )
        }
        
        return guidelines.get(form, "Create a poem with appropriate structure and style.")
    
    def _determine_poem_length(self, form: str, length_preference: str) -> str:
        """Determine target length based on form and preference."""
        # Fixed-length forms
        if form == "haiku":
            return "3 lines with 5-7-5 syllable pattern"
        elif form == "sonnet":
            return "14 lines"
        elif form == "limerick":
            return "5 lines"
        elif form == "villanelle":
            return "19 lines (5 tercets and 1 quatrain)"
        elif form == "acrostic":
            return "Variable based on acrostic word"
        
        # Variable length forms
        length_guide = {
            "short": "4-8 lines",
            "medium": "12-20 lines",
            "long": "25-40 lines"
        }
        
        return length_guide.get(length_preference, "12-20 lines")
    
    def _construct_poetry_prompt(
        self, theme: str, form: str, tone: str, style: str, 
        form_guidelines: str, target_length: str, include_title: bool
    ) -> str:
        """Construct detailed prompt for poetry generation."""
        # Detailed prompt construction
        prompt = (
            f"Write a {style} {form} poem with a {tone} tone about {theme}. "
            f"Target length: {target_length}.\n\n"
            
            f"Form guidelines: {form_guidelines}\n\n"
            
            f"{'Include an evocative title that captures the essence of the poem.' if include_title else 'Do not include a title.'}\n\n"
            
            f"Style notes: Use {style} poetic techniques and language. "
            f"Maintain a consistent {tone} emotional quality throughout. "
            f"Use vivid imagery, thoughtful word choice, and appropriate poetic devices "
            f"(metaphor, simile, alliteration, assonance, etc.) to convey the theme.\n\n"
            
            f"Create a poem that resonates emotionally and captures the essence of {theme} "
            f"in a way that's authentic to the {form} tradition while incorporating {style} sensibilities."
        )
        
        return prompt
    
    async def _process_poem(self, poem: str, form: str, include_title: bool) -> Tuple[str, str]:
        """Process the generated poem and extract title if present."""
        poem = poem.strip()
        title = ""
        
        # Extract title if present and requested
        if include_title and "\n" in poem:
            first_line = poem.split("\n", 1)[0].strip()
            
            # Check if first line looks like a title (no ending punctuation, relatively short)
            if (len(first_line) < 60 and 
                not first_line.endswith((".", "!", "?", ","", ";", ":"")) and
                not first_line.startswith(("#", "*", "-"))):
                
                title = first_line
                poem = poem.split("\n", 1)[1].strip()
        
        # For haiku, ensure proper formatting
        if form == "haiku":
            poem_lines = [line for line in poem.split("\n") if line.strip()]
            if len(poem_lines) >= 3:
                poem = "\n".join(poem_lines[:3])  # Take only the first three lines
        
        return poem, title
    
    async def _analyze_poetic_elements(self, poem: str, form: str) -> Dict[str, Any]:
        """Analyze poetic elements and techniques used in the poem."""
        # Prompt for analysis
        analysis_prompt = (
            f"Analyze this {form} poem for the following elements:\n"
            f"1. Main imagery and symbols\n"
            f"2. Poetic devices used (metaphor, simile, alliteration, etc.)\n"
            f"3. Thematic elements\n"
            f"4. Emotional tone\n"
            f"5. Structural elements specific to this form\n\n"
            f"Poem:\n{poem}"
        )
        
        analysis_text = await self.model_router.generate_text(
            analysis_prompt, model="analysis", parameters={"temperature": 0.3}
        )
        
        # Parse the analysis
        elements = {
            "imagery": [],
            "devices": [],
            "themes": [],
            "tone": "",
            "structure": {}
        }
        
        current_section = None
        for line in analysis_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if "imagery" in line.lower() and ":" in line or "symbol" in line.lower() and ":" in line:
                current_section = "imagery"
                continue
            elif "device" in line.lower() and ":" in line:
                current_section = "devices"
                continue
            elif "theme" in line.lower() and ":" in line:
                current_section = "themes"
                continue
            elif "tone" in line.lower() and ":" in line or "emotion" in line.lower() and ":" in line:
                current_section = "tone"
                continue
            elif "structure" in line.lower() and ":" in line:
                current_section = "structure"
                continue
                
            if current_section == "imagery":
                if line.startswith("-") or line.startswith("*"):
                    elements["imagery"].append(line[1:].strip())
                elif "," in line:
                    items = [item.strip() for item in line.split(",") if item.strip()]
                    elements["imagery"].extend(items)
            elif current_section == "devices":
                if line.startswith("-") or line.startswith("*"):
                    elements["devices"].append(line[1:].strip())
                elif "," in line:
                    items = [item.strip() for item in line.split(",") if item.strip()]
                    elements["devices"].extend(items)
            elif current_section == "themes":
                if line.startswith("-") or line.startswith("*"):
                    elements["themes"].append(line[1:].strip())
                elif "," in line:
                    items = [item.strip() for item in line.split(",") if item.strip()]
                    elements["themes"].extend(items)
            elif current_section == "tone":
                elements["tone"] += line + " "
            elif current_section == "structure":
                if "rhyme scheme" in line.lower():
                    elements["structure"]["rhyme_scheme"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "meter" in line.lower():
                    elements["structure"]["meter"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "stanza" in line.lower():
                    elements["structure"]["stanza_pattern"] = line.split(":", 1)[1].strip() if ":" in line else line
                else:
                    key = "general"
                    if ":" in line:
                        key_part, value = line.split(":", 1)
                        key = key_part.lower().strip().replace(" ", "_")
                        elements["structure"][key] = value.strip()
                    else:
                        elements["structure"][key] = line.strip()
        
        # Clean up
        elements["tone"] = elements["tone"].strip()
        
        # Additional form-specific analysis
        if form == "haiku":
            # Count syllables
            lines = [line for line in poem.split("\n") if line.strip()]
            if len(lines) >= 3:
                elements["structure"]["syllable_count"] = [
                    self._estimate_syllables(lines[0]),
                    self._estimate_syllables(lines[1]),
                    self._estimate_syllables(lines[2])
                ]
        
        return elements
    
    def _estimate_syllables(self, line: str) -> int:
        """Estimate syllable count in a line of text."""
        # Simple syllable estimation - would be more accurate with a dedicated library
        line = line.lower().strip()
        if not line:
            return 0
            
        # Count vowel groups
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False
        
        for char in line:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        
        # Adjust for common patterns
        if line.endswith('e'):
            count -= 1
        if line.endswith('le'):
            count += 1
        if count == 0:
            count = 1
            
        return count


class CharacterCreator(CreativeSkill):
    """Advanced character creation and development skill."""
    
    def get_skill_id(self) -> str:
        return "character_creator"
    
    def get_skill_description(self) -> str:
        return "Creates richly detailed fictional characters with backstories, motivations, and personality traits."
    
    def get_skill_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "name": {
                "type": "string",
                "description": "Character name (or leave blank to generate one)",
                "required": False
            },
            "genre": {
                "type": "string",
                "description": "Genre setting for the character",
                "enum": [g.value for g in CreativeGenre],
                "default": CreativeGenre.FANTASY.value
            },
            "archetype": {
                "type": "string",
                "description": "Character archetype",
                "enum": ["hero", "mentor", "ally", "trickster", "guardian", "shadow", 
                         "anti_hero", "lover", "rebel", "caregiver", "explorer", "creator", "sage"],
                "default": "hero"
            },
            "complexity": {
                "type": "string",
                "description": "Character complexity level",
                "enum": ["simple", "moderate", "complex"],
                "default": "moderate"
            },
            "age": {
                "type": "string",
                "description": "Character age",
                "required": False
            },
            "occupation": {
                "type": "string",
                "description": "Character occupation or role",
                "required": False
            },
            "moral_alignment": {
                "type": "string",
                "description": "Character's moral alignment",
                "enum": ["lawful_good", "neutral_good", "chaotic_good", "lawful_neutral", 
                         "true_neutral", "chaotic_neutral", "lawful_evil", "neutral_evil", "chaotic_evil"],
                "required": False
            },
            "include_visual_description": {
                "type": "boolean",
                "description": "Whether to include detailed visual description",
                "default": True
            },
            "include_dialogue_example": {
                "type": "boolean",
                "description": "Whether to include dialogue examples",
                "default": True
            }
        }
    
    def get_skill_examples(self) -> List[Dict[str, Any]]:
        return [
            {
                "parameters": {
                    "genre": "science_fiction",
                    "archetype": "anti_hero",
                    "complexity": "complex",
                    "occupation": "rogue AI researcher",
                    "moral_alignment": "chaotic_neutral"
                },
                "description": "Create a complex anti-hero character who is a rogue AI researcher in a sci-fi setting"
            },
            {
                "parameters": {
                    "name": "Elara Moonwhisper",
                    "genre": "fantasy",
                    "archetype": "mentor",
                    "age": "ancient",
                    "include_dialogue_example": True
                },
                "description": "Create a fantasy mentor character named Elara Moonwhisper who is ancient, including dialogue examples"
            }
        ]
    
    @handle_exceptions
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Create a detailed fictional character.
        
        Args:
            name: Character name (optional)
            genre: Genre setting
            archetype: Character archetype
            complexity: Character complexity level
            age: Character age (optional)
            occupation: Character occupation (optional)
            moral_alignment: Character's moral alignment (optional)
            include_visual_description: Whether to include detailed visual description
            include_dialogue_example: Whether to include dialogue examples
            
        Returns:
            Character profile and metadata
        """
        execution_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Track metrics
        if self.metrics:
            self.metrics.increment(f"skill_{self.get_skill_id()}_executions_total")
        
        # Emit execution started event
        await self.event_bus.emit(SkillExecutionStarted(
            skill_id=self.get_skill_id(),
            execution_id=execution_id,
            parameters=kwargs
        ))
        
        try:
            with self.tracer.trace("character_creation") if self.tracer else None:
                # Extract parameters with defaults
                name = kwargs.get("name", "")
                genre = kwargs.get("genre", CreativeGenre.FANTASY.value)
                archetype = kwargs.get("archetype", "hero")
                complexity = kwargs.get("complexity", "moderate")
                age = kwargs.get("age", "")
                occupation = kwargs.get("occupation", "")
                moral_alignment = kwargs.get("moral_alignment", "")
                include_visual = kwargs.get("include_visual_description", True)
                include_dialogue = kwargs.get("include_dialogue_example", True)
                
                # Generate a name if not provided
                if not name:
                    name = await self._generate_name(genre, archetype)
                
                # Build character creation prompt
                prompt = self._construct_character_prompt(
                    name, genre, archetype, complexity, age, occupation, 
                    moral_alignment, include_visual, include_dialogue
                )
                
                # Generate character profile
                model_parameters = {
                    "temperature": 0.8,
                    "max_tokens": 3000,
                    "top_p": 0.9,
                    "frequency_penalty": 0.4,
                    "presence_penalty": 0.4
                }
                
                character_text = await self.model_router.generate_text(
                    prompt, model="creative", parameters=model_parameters
                )
                
                # Extract structured character data
                character_data = await self._extract_character_data(character_text)
                
                # Generate additional character elements
                character_data["potential_arcs"] = await self._generate_character_arcs(
                    character_data, genre, archetype
                )
                
                # Calculate execution time
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                # Track execution for learning
                await self._track_execution(
                    execution_id,
                    genre=genre,
                    archetype=archetype,
                    complexity=complexity,
                    execution_time=execution_time
                )
                
                # Emit completion event
                await self.event_bus.emit(SkillExecutionCompleted(
                    skill_id=self.get_skill_id(),
                    execution_id=execution_id,
                    execution_time=execution_time
                ))
                
                if self.metrics:
                    self.metrics.record(f"skill_{self.get_skill_id()}_execution_time_seconds", execution_time)
                
                # Return results
                return {
                    "character": character_data,
                    "raw_profile": character_text,
                    "execution_time": execution_time,
                    "execution_id": execution_id
                }
                
        except Exception as e:
            # Handle errors
            if self.metrics:
                self.metrics.increment(f"skill_{self.get_skill_id()}_errors_total")
            
            await self.event_bus.emit(SkillExecutionFailed(
                skill_id=self.get_skill_id(),
                execution_id=execution_id,
                error_message=str(e)
            ))
            
            self.logger.error(f"Character creation failed: {str(e)}")
            raise
    
    async def _generate_name(self, genre: str, archetype: str) -> str:
        """Generate a fitting character name based on genre and archetype."""
        name_prompt = (
            f"Generate a single evocative name (first and last) for a {archetype} character in a {genre} setting. "
            f"The name should fit the genre conventions while being distinctive and memorable. "
            f"Only provide the name without explanation."
        )
        
        name = await self.model_router.generate_text(
            name_prompt, model="creative", parameters={"temperature": 0.9, "max_tokens": 30}
        )
        
        # Clean up
        name = name.strip().split("\n")[0]
        if ":" in name:
            name = name.split(":", 1)[1].strip()
        
        return name
    
    def _construct_character_prompt(
        self, name: str, genre: str, archetype: str, complexity: str, 
        age: str, occupation: str, moral_alignment: str, 
        include_visual: bool, include_dialogue: bool
    ) -> str:
        """Construct detailed prompt for character creation."""
        # Additional details based on provided parameters
        specifics = []
        if age:
            specifics.append(f"Age: {age}")
        if occupation:
            specifics.append(f"Occupation: {occupation}")
        if moral_alignment:
            specifics.append(f"Moral alignment: {moral_alignment.replace('_', ' ')}")
        
        specifics_text = ". ".join(specifics) if specifics else "Create appropriate details based on the genre and archetype."
        
        # Complexity details
        complexity_guide = {
            "simple": "Create a straightforward character with clear motivations and a relatively uncomplicated background.",
            "moderate": "Develop a character with some nuance, including inner conflicts and a moderately detailed background.",
            "complex": "Craft a multilayered character with significant depth, internal contradictions, moral complexity, and a richly detailed background."
        }
        
        complexity_text = complexity_guide.get(complexity, complexity_guide["moderate"])
        
        # Detailed prompt
        prompt = (
            f"Create a detailed character profile for {name}, a {archetype} character in a {genre} setting.\n\n"
            
            f"Character specifications: {specifics_text}\n\n"
            
            f"Character complexity: {complexity_text}\n\n"
            
            f"Include the following sections in the character profile:\n"
            f"1. Basic Information (name, age, role/occupation, etc.)\n"
            f"2. Background & History\n"
            f"3. Personality Traits\n"
            f"4. Motivations & Goals\n"
            f"5. Strengths & Weaknesses\n"
            f"6. Relationships & Connections\n"
            f"7. Secrets & Hidden Depths\n"
            f"8. Fears & Vulnerabilities\n"
        )
        
        if include_visual:
            prompt += f"9. Visual Description (appearance, clothing style, distinctive features)\n"
        
        if include_dialogue:
            prompt += f"{'10' if include_visual else '9'}. Dialogue Examples (2-3 characteristic quotes)\n"
        
        prompt += (
            f"\nMake the character distinctive, memorable, and fitting for a {genre} setting while embodying the {archetype} "
            f"archetype in a unique way. Ensure the character has depth, clear motivations, and interesting contradictions "
            f"that drive their behavior. The character should feel authentic and three-dimensional."
        )
        
        return prompt
    
    async def _extract_character_data(self, character_text: str) -> Dict[str, Any]:
        """Extract structured data from character profile text."""
        # Extract structured data using pattern matching and LLM assistance
        structured_data = {
            "basic_info": {},
            "background": "",
            "personality": [],
            "motivations": [],
            "strengths": [],
            "weaknesses": [],
            "relationships": [],
            "secrets": [],
            "fears": [],
            "visual_description": "",
            "dialogue_examples": []
        }
        
        # Use LLM to help extract structured data
        extraction_prompt = (
            f"Extract the following elements from this character profile into a structured format:\n"
            f"1. Basic Information (name, age, role/occupation as key-value pairs)\n"
            f"2. Background (summarized)\n"
            f"3. Personality Traits (as a list)\n"
            f"4. Motivations & Goals (as a list)\n"
            f"5. Strengths (as a list)\n"
            f"6. Weaknesses (as a list)\n"
            f"7. Relationships (as a list)\n"
            f"8. Secrets (as a list)\n"
            f"9. Fears (as a list)\n"
            f"10. Visual Description (summarized)\n"
            f"11. Dialogue Examples (as a list of quotes)\n\n"
            f"Character Profile:\n{character_text}"
        )
        
        extraction_result = await self.model_router.generate_text(
            extraction_prompt, model="analysis", parameters={"temperature": 0.3}
        )
        
        # Parse the extraction result
        current_section = None
        basic_info_collecting = False
        
        for line in extraction_result.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Identify sections
            if "basic information" in line.lower() or "basic info" in line.lower():
                current_section = "basic_info"
                basic_info_collecting = True
                continue
            elif "background" in line.lower():
                current_section = "background"
                basic_info_collecting = False
                continue
            elif "personality" in line.lower():
                current_section = "personality"
                basic_info_collecting = False
                continue
            elif "motivations" in line.lower() or "goals" in line.lower():
                current_section = "motivations"
                basic_info_collecting = False
                continue
            elif "strengths" in line.lower() and "weaknesses" not in line.lower():
                current_section = "strengths"
                basic_info_collecting = False
                continue
            elif "weaknesses" in line.lower():
                current_section = "weaknesses"
                basic_info_collecting = False
                continue
            elif "relationships" in line.lower() or "connections" in line.lower():
                current_section = "relationships"
                basic_info_collecting = False
                continue
            elif "secrets" in line.lower() or "hidden depths" in line.lower():
                current_section = "secrets"
                basic_info_collecting = False
                continue
            elif "fears" in line.lower() or "vulnerabilities" in line.lower():
                current_section = "fears"
                basic_info_collecting = False
                continue
            elif "visual" in line.lower() or "appearance" in line.lower() or "description" in line.lower():
                current_section = "visual_description"
                basic_info_collecting = False
                continue
            elif "dialogue" in line.lower() or "quotes" in line.lower():
                current_section = "dialogue_examples"
                basic_info_collecting = False
                continue
            
            # Process content based on current section
            if current_section == "basic_info" and basic_info_collecting:
                if ":" in line:
                    key, value = line.split(":", 1)
                    structured_data["basic_info"][key.strip().lower().replace(" ", "_")] = value.strip()
            elif current_section == "background":
                structured_data["background"] += line + " "
            elif current_section in ["personality", "motivations", "strengths", "weaknesses", 
                                    "relationships", "secrets", "fears"]:
                if line.startswith("-") or line.startswith("*"):
                    structured_data[current_section].append(line[1:].strip())
                elif line[0].isdigit() and ". " in line:
                    structured_data[current_section].append(line.split(". ", 1)[1].strip())
            elif current_section == "visual_description":
                structured_data["visual_description"] += line + " "
            elif current_section == "dialogue_examples":
                if line.startswith("-") or line.startswith("*") or line.startswith('"') or line.startswith('"'):
                    # Clean up quotes
                    line = line[1:].strip() if line.startswith("-") or line.startswith("*") else line.strip()
                    if not line.startswith('"') and not line.startswith('"') and not line.startswith("'"):
                        line = f'"{line}"'
                    structured_data["dialogue_examples"].append(line)
                elif line[0].isdigit() and ". " in line:
                    line = line.split(". ", 1)[1].strip()
                    if not line.startswith('"') and not line.startswith('"') and not line.startswith("'"):
                        line = f'"{line}"'
                    structured_data["dialogue_examples"].append(line)
        
        # Clean up
        structured_data["background"] = structured_data["background"].strip()
        structured_data["visual_description"] = structured_data["visual_description"].strip()
        
        # Ensure we have a name
        if "name" not in structured_data["basic_info"]:
            # Try to extract from the first line of the profile
            first_line = character_text.split("\n", 1)[0].strip()
            if ":" in first_line:
                potential_name = first_line.split(":", 1)[1].strip()
                structured_data["basic_info"]["name"] = potential_name
            else:
                structured_data["basic_info"]["name"] = "Unnamed Character"
        
        return structured_data
    
    async def _generate_character_arcs(
        self, character_data: Dict[str, Any], genre: str, archetype: str
    ) -> List[Dict[str, str]]:
        """Generate potential character arcs based on the character profile."""
        # Create prompt for character arc generation
        character_summary = (
            f"Character: {character_data['basic_info'].get('name', 'Unnamed')}\n"
            f"Genre: {genre}\n"
            f"Archetype: {archetype}\n"
            f"Background: {character_data['background']}\n"
            f"Motivations: {', '.join(character_data['motivations'])}\n"
            f"Strengths: {', '.join(character_data['strengths'])}\n"
            f"Weaknesses: {', '.join(character_data['weaknesses'])}\n"
            f"Fears: {', '.join(character_data['fears'])}\n"
            f"Secrets: {', '.join(character_data['secrets'])}"
        )
        
        arc_prompt = (
            f"Based on this character profile, generate 3 potential character arcs that would be compelling "
            f"for this character in a
