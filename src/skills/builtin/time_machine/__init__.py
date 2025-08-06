"""
Context Time Machine Skill Package
Author: Drmusab
Last Modified: 2025-01-08

This package implements the "Context Time Machine" feature that allows users to:
- Ask about past conversations in Arabic and English
- Replay conversation snippets or summaries
- Analyze behavioral changes and trends
- Search through past interactions using natural language

The time machine integrates with the existing memory system and provides
bilingual support for Arabic users.
"""

from .time_machine_skill import TimeMachineSkill
from .query_parser import QueryParser, QueryType, TimeRange
from .memory_replayer import MemoryReplayer, ReplayMode
from .reflection_analyzer import ReflectionAnalyzer, BehavioralMetric
from .search_engine import SearchEngine, SearchMode
from .visualization import VisualizationEngine, ChartType

__all__ = [
    'TimeMachineSkill',
    'QueryParser',
    'QueryType', 
    'TimeRange',
    'MemoryReplayer',
    'ReplayMode',
    'ReflectionAnalyzer',
    'BehavioralMetric',
    'SearchEngine',
    'SearchMode',
    'VisualizationEngine',
    'ChartType'
]

__version__ = "1.0.0"