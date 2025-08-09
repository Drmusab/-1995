"""
Shared data types and enums for the Thought Reflector skill.
Author: Drmusab
Last Modified: 2025-01-20
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any


class ReflectionType(Enum):
    """Types of reflection that can be generated."""
    WEEKLY_SUMMARY = "weekly_summary"
    PROBLEM_SOLVING_STYLE = "problem_solving_style"
    JOURNALING_PROMPT = "journaling_prompt"
    AFFIRMATION = "affirmation"
    REFRAMING_EXERCISE = "reframing_exercise"
    DEEPER_INQUIRY = "deeper_inquiry"


class ThoughtTheme(Enum):
    """Common themes identified in user thoughts."""
    TIME_MANAGEMENT = "time_management"
    CREATIVITY = "creativity"
    PROBLEM_SOLVING = "problem_solving"
    RELATIONSHIPS = "relationships"
    PRODUCTIVITY = "productivity"
    PERSONAL_GROWTH = "personal_growth"
    EMOTIONAL_AWARENESS = "emotional_awareness"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    STRESS_MANAGEMENT = "stress_management"


@dataclass
class ThoughtPattern:
    """Represents an identified thought pattern."""
    theme: ThoughtTheme
    frequency: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    first_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_detected: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProblemSolvingStyle:
    """Represents user's problem-solving approach."""
    style_name: str
    characteristics: List[str]
    strengths: List[str]
    suggestions: List[str]
    confidence: float
    examples: List[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    """Result of a reflection analysis."""
    reflection_type: ReflectionType
    content: str
    themes: List[ThoughtTheme]
    patterns: List[ThoughtPattern]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))