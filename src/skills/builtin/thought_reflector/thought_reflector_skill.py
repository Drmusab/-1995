"""
Meta-Cognition / Thought Reflection Skill
Author: Drmusab
Last Modified: 2025-01-20

Advanced meta-cognitive skill for analyzing thought patterns, generating reflective insights,
and encouraging deeper self-awareness through journaling prompts and affirmations.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
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
from src.skills.skill_registry import SkillInterface, SkillMetadata, SkillType, SkillCapability
from .types import (
    ReflectionType, ThoughtTheme, ThoughtPattern, ProblemSolvingStyle, ReflectionResult
)


class ThoughtReflectorSkill(SkillInterface):
    """
    Meta-Cognition / Thought Reflection Skill for deep self-awareness.
    
    Features:
    - Weekly thought pattern summaries
    - Problem-solving style analysis
    - Personalized journaling prompts
    - Reflective questions and affirmations
    - Pattern detection across time periods
    - Integration with memory systems for historical analysis
    """
    
    def __init__(self):
        """Initialize the thought reflector skill."""
        self.logger = get_logger(__name__)
        self.initialized = False
        
        # Will be set during initialization
        self.container: Optional[Container] = None
        self.config: Optional[ConfigLoader] = None
        self.event_bus: Optional[EventBus] = None
        self.error_handler: Optional[ErrorHandler] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # Skill components
        self.thought_analyzer: Optional[ThoughtAnalyzer] = None
        self.reflection_generator: Optional[ReflectionGenerator] = None
        self.pattern_detector: Optional[PatternDetector] = None
        self.weekly_summarizer: Optional[WeeklySummarizer] = None
        
        # Configuration
        self.analysis_window_days = 7
        self.min_interactions_for_analysis = 5
        self.confidence_threshold = 0.6
        self.storage_directory = Path("data/thought_reflections")
        
        # Cache for patterns and analyses
        self.pattern_cache: Dict[str, List[ThoughtPattern]] = {}
        self.style_cache: Dict[str, ProblemSolvingStyle] = {}
        
        self.logger.info("ThoughtReflectorSkill initialized")
    
    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return SkillMetadata(
            skill_id="builtin.thought_reflector",
            name="Thought Reflector",
            version="1.0.0",
            description="Meta-cognitive skill for analyzing thought patterns and generating reflective insights",
            author="Drmusab",
            skill_type=SkillType.BUILTIN,
            capabilities=[
                SkillCapability(
                    name="weekly_summary",
                    description="Generate weekly summaries of thought patterns",
                    input_types=["text", "dict"],
                    output_types=["dict", "text"]
                ),
                SkillCapability(
                    name="problem_solving_analysis",
                    description="Analyze and describe problem-solving style",
                    input_types=["text", "dict"],
                    output_types=["dict", "text"]
                ),
                SkillCapability(
                    name="journaling_prompts",
                    description="Generate personalized journaling prompts",
                    input_types=["text", "dict"],
                    output_types=["dict", "text"]
                ),
                SkillCapability(
                    name="affirmations",
                    description="Create personalized affirmations based on patterns",
                    input_types=["text", "dict"],
                    output_types=["dict", "text"]
                ),
                SkillCapability(
                    name="reframing_exercises",
                    description="Suggest reframing exercises for challenging thoughts",
                    input_types=["text", "dict"],
                    output_types=["dict", "text"]
                )
            ],
            tags=["meta-cognition", "reflection", "self-awareness", "patterns", "journaling"],
            dependencies=["memory_manager", "event_bus", "config"],
            resource_requirements={
                "memory_mb": 128,
                "cpu_cores": 1,
                "disk_mb": 50
            }
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the skill with dependencies."""
        try:
            # Get dependencies from container
            if hasattr(self, '_dependencies') and self._dependencies:
                self.container = self._dependencies['container']
                self.config = self._dependencies['config']
                self.event_bus = self._dependencies['event_bus']
                self.error_handler = self._dependencies.get('error_handler')
                self.memory_manager = self._dependencies['memory_manager']
            
            # Setup configuration
            self._setup_configuration(config)
            
            # Initialize components
            self.thought_analyzer = ThoughtAnalyzer(self.config)
            self.reflection_generator = ReflectionGenerator(self.config)
            self.pattern_detector = PatternDetector(self.config)
            self.weekly_summarizer = WeeklySummarizer(self.config, self.memory_manager)
            
            # Ensure storage directory exists
            self.storage_directory.mkdir(parents=True, exist_ok=True)
            
            self.initialized = True
            self.logger.info("ThoughtReflectorSkill initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ThoughtReflectorSkill: {str(e)}")
            raise
    
    async def set_dependencies(self, dependencies: Dict[str, Any]) -> None:
        """Set dependencies for dependency injection."""
        self._dependencies = dependencies
    
    def _setup_configuration(self, config: Dict[str, Any]) -> None:
        """Setup skill configuration."""
        thought_config = config.get("thought_reflector", {})
        
        # Analysis settings
        self.analysis_window_days = thought_config.get("analysis_window_days", 7)
        self.min_interactions_for_analysis = thought_config.get("min_interactions", 5)
        self.confidence_threshold = thought_config.get("confidence_threshold", 0.6)
        
        # Storage settings
        storage_path = thought_config.get("storage_directory", "data/thought_reflections")
        self.storage_directory = Path(storage_path)
        
        # Feature flags
        self.enable_weekly_summaries = thought_config.get("enable_weekly_summaries", True)
        self.enable_style_analysis = thought_config.get("enable_style_analysis", True)
        self.enable_affirmations = thought_config.get("enable_affirmations", True)
        
        self.logger.debug("ThoughtReflectorSkill configuration loaded")
    
    @handle_exceptions
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """
        Execute the thought reflection skill.
        
        Args:
            input_data: Can be a dict with 'action' and parameters, or a text query
            context: Execution context with user_id, session_id, etc.
            
        Returns:
            Reflection result based on the requested action
        """
        if not self.initialized:
            raise RuntimeError("Skill not initialized")
        
        execution_id = str(uuid.uuid4())
        user_id = context.get("user_id")
        session_id = context.get("session_id")
        
        # Parse input
        if isinstance(input_data, dict):
            action = input_data.get("action", "weekly_summary")
            parameters = input_data.get("parameters", {})
        elif isinstance(input_data, str):
            # Try to parse natural language request
            action, parameters = self._parse_natural_language_request(input_data)
        else:
            raise ValueError("Input must be a dict with action/parameters or a text string")
        
        self.logger.info(f"Executing thought reflection action: {action}")
        
        # Emit execution started event
        if self.event_bus:
            await self.event_bus.emit(
                SkillExecutionStarted(
                    skill_name="thought_reflector",
                    session_id=session_id or execution_id,
                    user_id=user_id,
                    parameters={"action": action, **parameters}
                )
            )
        
        try:
            # Route to appropriate handler
            if action == "weekly_summary":
                result = await self._generate_weekly_summary(user_id, parameters)
            elif action == "problem_solving_style":
                result = await self._analyze_problem_solving_style(user_id, parameters)
            elif action == "journaling_prompt":
                result = await self._generate_journaling_prompt(user_id, parameters)
            elif action == "affirmation":
                result = await self._generate_affirmation(user_id, parameters)
            elif action == "reframing_exercise":
                result = await self._generate_reframing_exercise(user_id, parameters)
            elif action == "deeper_inquiry":
                result = await self._generate_deeper_inquiry(user_id, parameters)
            else:
                raise ValueError(f"Unknown action: {action}")
            
            # Emit execution completed event
            if self.event_bus:
                await self.event_bus.emit(
                    SkillExecutionCompleted(
                        skill_name="thought_reflector",
                        session_id=session_id or execution_id,
                        user_id=user_id,
                        result_summary=f"Generated {action} successfully"
                    )
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing thought reflection: {str(e)}")
            raise
    
    def _parse_natural_language_request(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Parse natural language request to extract action and parameters."""
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in ["weekly", "week", "summary", "summarize"]):
            return "weekly_summary", {}
        elif any(phrase in text_lower for phrase in ["problem solving", "how do i solve", "approach"]):
            return "problem_solving_style", {}
        elif any(phrase in text_lower for phrase in ["journal", "writing prompt", "reflect on"]):
            return "journaling_prompt", {}
        elif any(phrase in text_lower for phrase in ["affirmation", "positive", "encourage"]):
            return "affirmation", {}
        elif any(phrase in text_lower for phrase in ["reframe", "different perspective", "look at differently"]):
            return "reframing_exercise", {}
        elif any(phrase in text_lower for phrase in ["deeper", "explore", "investigate", "inquiry"]):
            return "deeper_inquiry", {}
        else:
            # Default to weekly summary
            return "weekly_summary", {}
    
    async def _generate_weekly_summary(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a weekly summary of thought patterns."""
        if not self.weekly_summarizer:
            raise RuntimeError("Weekly summarizer not initialized")
        
        # Get time window
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.analysis_window_days)
        
        # Generate summary
        summary = await self.weekly_summarizer.generate_summary(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Detect patterns
        patterns = await self.pattern_detector.detect_patterns(
            user_id=user_id,
            time_window_days=self.analysis_window_days
        )
        
        # Generate insights
        insights = self._generate_insights_from_patterns(patterns)
        
        result = ReflectionResult(
            reflection_type=ReflectionType.WEEKLY_SUMMARY,
            content=self._format_weekly_summary(summary, patterns, insights),
            themes=[p.theme for p in patterns],
            patterns=patterns,
            suggestions=insights,
            metadata={
                "analysis_period": f"{start_date.date()} to {end_date.date()}",
                "interaction_count": summary.get("interaction_count", 0),
                "dominant_themes": [p.theme.value for p in patterns[:3]]  # Top 3 themes
            }
        )
        
        return asdict(result)
    
    async def _analyze_problem_solving_style(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's problem-solving style."""
        if not self.thought_analyzer:
            raise RuntimeError("Thought analyzer not initialized")
        
        # Check cache first
        if user_id in self.style_cache:
            style = self.style_cache[user_id]
        else:
            # Analyze problem-solving patterns
            style = await self.thought_analyzer.analyze_problem_solving_style(
                user_id=user_id,
                time_window_days=self.analysis_window_days
            )
            self.style_cache[user_id] = style
        
        # Generate reflection content
        content = self._format_problem_solving_analysis(style)
        
        result = ReflectionResult(
            reflection_type=ReflectionType.PROBLEM_SOLVING_STYLE,
            content=content,
            themes=[],  # Problem-solving style doesn't have specific themes
            patterns=[],
            suggestions=style.suggestions,
            metadata={
                "style_name": style.style_name,
                "confidence": style.confidence,
                "analysis_date": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return asdict(result)
    
    async def _generate_journaling_prompt(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a personalized journaling prompt."""
        if not self.reflection_generator:
            raise RuntimeError("Reflection generator not initialized")
        
        # Get recent patterns
        patterns = await self.pattern_detector.detect_patterns(
            user_id=user_id,
            time_window_days=3  # More recent patterns for journaling
        )
        
        # Generate prompt
        prompt = await self.reflection_generator.generate_journaling_prompt(patterns)
        
        result = ReflectionResult(
            reflection_type=ReflectionType.JOURNALING_PROMPT,
            content=prompt,
            themes=[p.theme for p in patterns],
            patterns=patterns,
            suggestions=[
                "Set aside 10-15 minutes for writing",
                "Write freely without editing yourself",
                "Focus on your feelings and thoughts",
                "Consider multiple perspectives"
            ],
            metadata={
                "prompt_type": "personalized",
                "based_on_themes": [p.theme.value for p in patterns[:2]]
            }
        )
        
        return asdict(result)
    
    async def _generate_affirmation(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a personalized affirmation."""
        if not self.reflection_generator:
            raise RuntimeError("Reflection generator not initialized")
        
        # Get patterns to understand areas needing encouragement
        patterns = await self.pattern_detector.detect_patterns(
            user_id=user_id,
            time_window_days=self.analysis_window_days
        )
        
        # Generate affirmation
        affirmation = await self.reflection_generator.generate_affirmation(patterns)
        
        result = ReflectionResult(
            reflection_type=ReflectionType.AFFIRMATION,
            content=affirmation,
            themes=[p.theme for p in patterns],
            patterns=patterns,
            suggestions=[
                "Repeat this affirmation throughout your day",
                "Say it aloud to reinforce the message",
                "Write it down and place it somewhere visible",
                "Believe in the truth of these words"
            ],
            metadata={
                "affirmation_type": "personalized",
                "focus_areas": [p.theme.value for p in patterns if p.confidence > self.confidence_threshold]
            }
        )
        
        return asdict(result)
    
    async def _generate_reframing_exercise(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a reframing exercise for challenging thoughts."""
        if not self.reflection_generator:
            raise RuntimeError("Reflection generator not initialized")
        
        # Get patterns that might indicate challenging areas
        patterns = await self.pattern_detector.detect_patterns(
            user_id=user_id,
            time_window_days=self.analysis_window_days
        )
        
        # Generate reframing exercise
        exercise = await self.reflection_generator.generate_reframing_exercise(patterns)
        
        result = ReflectionResult(
            reflection_type=ReflectionType.REFRAMING_EXERCISE,
            content=exercise,
            themes=[p.theme for p in patterns],
            patterns=patterns,
            suggestions=[
                "Take time to work through each step",
                "Be honest about your initial thoughts",
                "Consider multiple alternative perspectives",
                "Practice this technique regularly"
            ],
            metadata={
                "exercise_type": "cognitive_reframing",
                "target_patterns": [p.theme.value for p in patterns[:2]]
            }
        )
        
        return asdict(result)
    
    async def _generate_deeper_inquiry(self, user_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deeper inquiry questions for self-exploration."""
        if not self.reflection_generator:
            raise RuntimeError("Reflection generator not initialized")
        
        # Get patterns for deeper exploration
        patterns = await self.pattern_detector.detect_patterns(
            user_id=user_id,
            time_window_days=self.analysis_window_days
        )
        
        # Generate inquiry questions
        inquiry = await self.reflection_generator.generate_deeper_inquiry(patterns)
        
        result = ReflectionResult(
            reflection_type=ReflectionType.DEEPER_INQUIRY,
            content=inquiry,
            themes=[p.theme for p in patterns],
            patterns=patterns,
            suggestions=[
                "Sit with these questions without rushing to answer",
                "Allow unexpected insights to emerge",
                "Consider discussing with a trusted friend or counselor",
                "Journal about your responses"
            ],
            metadata={
                "inquiry_type": "self_exploration",
                "exploration_themes": [p.theme.value for p in patterns]
            }
        )
        
        return asdict(result)
    
    def _generate_insights_from_patterns(self, patterns: List[ThoughtPattern]) -> List[str]:
        """Generate insights from detected patterns."""
        insights = []
        
        if not patterns:
            insights.append("Continue to engage in self-reflection to develop meaningful patterns.")
            return insights
        
        # Sort patterns by frequency and confidence
        sorted_patterns = sorted(patterns, key=lambda p: p.frequency * p.confidence, reverse=True)
        
        for i, pattern in enumerate(sorted_patterns[:3]):  # Top 3 patterns
            if pattern.theme == ThoughtTheme.TIME_MANAGEMENT:
                insights.append(f"You've been thinking a lot about time management and efficiency.")
            elif pattern.theme == ThoughtTheme.CREATIVITY:
                insights.append(f"Creativity and innovative thinking appear to be on your mind frequently.")
            elif pattern.theme == ThoughtTheme.PROBLEM_SOLVING:
                insights.append(f"You approach challenges with structured thinking and analytical methods.")
            elif pattern.theme == ThoughtTheme.PERSONAL_GROWTH:
                insights.append(f"Personal development and self-improvement are recurring themes for you.")
            elif pattern.theme == ThoughtTheme.RELATIONSHIPS:
                insights.append(f"Interpersonal connections and relationships feature prominently in your thoughts.")
            # Add more theme-specific insights...
        
        return insights
    
    def _format_weekly_summary(self, summary: Dict[str, Any], patterns: List[ThoughtPattern], insights: List[str]) -> str:
        """Format the weekly summary into readable text."""
        content = "## Weekly Thought Reflection Summary\n\n"
        
        # Overview
        interaction_count = summary.get("interaction_count", 0)
        content += f"Over the past week, you've engaged in {interaction_count} meaningful interactions. "
        
        if patterns:
            dominant_themes = [p.theme.value.replace('_', ' ').title() for p in patterns[:2]]
            content += f"Your thoughts have primarily centered around **{' and '.join(dominant_themes)}**.\n\n"
        
        # Key insights
        if insights:
            content += "### Key Insights:\n"
            for insight in insights:
                content += f"- {insight}\n"
            content += "\n"
        
        # Patterns discovered
        if patterns:
            content += "### Patterns Discovered:\n"
            for pattern in patterns[:3]:  # Top 3 patterns
                theme_name = pattern.theme.value.replace('_', ' ').title()
                content += f"- **{theme_name}**: Appeared {pattern.frequency} times with {pattern.confidence:.1%} confidence\n"
            content += "\n"
        
        # Reflection questions
        content += "### Questions for Reflection:\n"
        content += "- How do these patterns align with your current goals and values?\n"
        content += "- What insights surprise you about your thinking patterns?\n"
        content += "- Which patterns would you like to strengthen or modify?\n\n"
        
        content += "Remember, self-awareness is the first step toward intentional growth and positive change."
        
        return content
    
    def _format_problem_solving_analysis(self, style: ProblemSolvingStyle) -> str:
        """Format problem-solving style analysis."""
        content = f"## Your Problem-Solving Style: {style.style_name}\n\n"
        
        content += "### Characteristics:\n"
        for characteristic in style.characteristics:
            content += f"- {characteristic}\n"
        content += "\n"
        
        content += "### Your Strengths:\n"
        for strength in style.strengths:
            content += f"- {strength}\n"
        content += "\n"
        
        if style.suggestions:
            content += "### Suggestions for Growth:\n"
            for suggestion in style.suggestions:
                content += f"- {suggestion}\n"
            content += "\n"
        
        content += f"*Analysis confidence: {style.confidence:.1%}*\n\n"
        content += "Understanding your natural problem-solving approach can help you leverage your strengths and develop complementary skills."
        
        return content
    
    async def validate(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, dict):
            return "action" in input_data or len(input_data) == 0
        elif isinstance(input_data, str):
            return len(input_data.strip()) > 0
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check skill health."""
        health = {
            "status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized,
            "components": {
                "thought_analyzer": self.thought_analyzer is not None,
                "reflection_generator": self.reflection_generator is not None,
                "pattern_detector": self.pattern_detector is not None,
                "weekly_summarizer": self.weekly_summarizer is not None
            },
            "cache_size": {
                "patterns": len(self.pattern_cache),
                "styles": len(self.style_cache)
            }
        }
        
        return health
    
    async def cleanup(self) -> None:
        """Cleanup skill resources."""
        self.pattern_cache.clear()
        self.style_cache.clear()
        self.initialized = False
        self.logger.info("ThoughtReflectorSkill cleaned up")


# Import helper modules after class definition to avoid circular imports
from .thought_analyzer import ThoughtAnalyzer
from .reflection_generator import ReflectionGenerator
from .pattern_detector import PatternDetector
from .weekly_summarizer import WeeklySummarizer