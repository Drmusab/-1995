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
from .query_parser import QueryParser, QueryType, TimeRange, ParsedQuery
from .memory_replayer import MemoryReplayer, ReplayMode, ConversationThread, ConversationSegment
from .reflection_analyzer import ReflectionAnalyzer, BehavioralMetric, BehavioralAnalysis, BehavioralTrend
from .search_engine import SearchEngine, SearchMode, SearchResult, SearchResponse
from .visualization import VisualizationEngine, ChartType, ExportFormat, VisualizationResult

__all__ = [
    'TimeMachineSkill',
    'QueryParser',
    'QueryType', 
    'TimeRange',
    'ParsedQuery',
    'MemoryReplayer',
    'ReplayMode',
    'ConversationThread',
    'ConversationSegment',
    'ReflectionAnalyzer',
    'BehavioralMetric',
    'BehavioralAnalysis',
    'BehavioralTrend',
    'SearchEngine',
    'SearchMode',
    'SearchResult',
    'SearchResponse',
    'VisualizationEngine',
    'ChartType',
    'ExportFormat',
    'VisualizationResult'
]

__version__ = "1.0.0"