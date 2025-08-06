# Context Time Machine Skill Documentation

## Overview

The Context Time Machine is an advanced AI skill that allows users to query their past conversations and analyze behavioral patterns over time. It provides bilingual support for Arabic and English users with sophisticated natural language understanding.

## Features

### 1. Natural Language Query Processing
- **Arabic Support**: Full support for Arabic queries like "ماذا تحدثنا عنه الأسبوع الماضي؟"
- **English Support**: Complete English query processing like "What did we talk about last week?"
- **Query Types**:
  - Conversation Recall: Finding past discussions
  - Project Mentions: Tracking project-related conversations
  - Behavioral Analysis: Analyzing communication patterns
  - Mood Tracking: Emotional state over time
  - Temporal Summaries: Time-based conversation summaries

### 2. Advanced Search Engine
- **Hybrid Search**: Combines semantic and keyword search
- **Vector Search**: Uses embeddings for semantic similarity
- **Temporal Filtering**: Time-based search with natural language time references
- **Contextual Search**: Context-aware search based on query intent

### 3. Conversation Replay
- **Multiple Modes**:
  - Full: Complete conversation transcripts
  - Summary: Condensed conversation summaries
  - Highlights: Key moments and important segments
  - Timeline: Chronological view of interactions
  - Contextual: Topic-based conversation grouping

### 4. Behavioral Analysis
- **Metrics Tracked**:
  - Tone: Communication tone analysis
  - Mood: Emotional state tracking
  - Energy: Energy level indicators
  - Confidence: Confidence in communication
  - Engagement: Level of interaction
  - Complexity: Language complexity analysis

- **Trend Analysis**:
  - Improving, declining, or stable trends
  - Change magnitude calculation
  - Confidence scoring for reliability

### 5. Visualization Engine
- **Chart Types**:
  - Line charts for trend visualization
  - Bar charts for metric comparison
  - Progress bars for improvement tracking
  - ASCII charts for simple text output

- **Export Formats**:
  - JSON (Chart.js compatible)
  - CSV for data export
  - ASCII for text-based visualization

## Architecture

### Core Components

1. **QueryParser** (`query_parser.py`)
   - Natural language understanding
   - Time reference extraction
   - Intent classification
   - Bilingual pattern matching

2. **SearchEngine** (`search_engine.py`)
   - Multi-modal search capabilities
   - Relevance scoring
   - Result aggregation
   - Search suggestions

3. **MemoryReplayer** (`memory_replayer.py`)
   - Conversation reconstruction
   - Thread grouping and organization
   - Multiple replay modes
   - Context preservation

4. **ReflectionAnalyzer** (`reflection_analyzer.py`)
   - Behavioral pattern analysis
   - Trend identification
   - Insight generation
   - Recommendation system

5. **VisualizationEngine** (`visualization.py`)
   - Chart generation
   - Data visualization
   - Multiple export formats
   - Interactive charts

6. **TimeMachineSkill** (`time_machine_skill.py`)
   - Main orchestrator
   - Request processing
   - Response compilation
   - Error handling

## Usage Examples

### Arabic Queries
```
"ماذا تحدثنا عنه الأسبوع الماضي؟"
"متى كانت آخر مرة ذكرت فيها المشروع؟"
"هل تحسن أسلوبي في الاجتماعات؟"
"كيف كان مزاجي في الشهر الماضي؟"
```

### English Queries
```
"What did we talk about last week?"
"When was the last time I mentioned the project?"
"Have I improved my style in meetings?"
"How was my mood last month?"
```

## Configuration

The skill is configured via `configs/skills/time_machine.yaml`:

- **Language Settings**: Default language, bilingual mode
- **Search Configuration**: Result limits, relevance thresholds
- **Analysis Parameters**: Trend sensitivity, confidence thresholds
- **Visualization Options**: Chart types, colors, export formats
- **Pattern Definitions**: Language-specific patterns for analysis

## Integration

### Memory System Integration
- Works with episodic, semantic, and working memory
- Searches across all memory types
- Respects memory access controls

### Bilingual Manager Integration
- Automatic language detection
- Language-appropriate responses
- Pattern matching in both languages

### Event System Integration
- Emits skill execution events
- Tracks processing metrics
- Supports monitoring and observability

## Response Format

The skill returns comprehensive responses including:

```json
{
  "query_analysis": {
    "original_query": "User's original query",
    "detected_language": "ar|en",
    "query_type": "conversation_recall|behavioral_analysis|...",
    "confidence": 0.85,
    "time_range": {...},
    "topics": [...],
    "entities": [...]
  },
  "search_results": {
    "total_found": 15,
    "search_time": 0.12,
    "results": [...]
  },
  "answer": "Main response text in user's language",
  "conversations": {
    "total_threads": 5,
    "threads": [...]
  },
  "behavioral_analysis": {
    "confidence_score": 0.8,
    "trends": [...]
  },
  "insights": [...],
  "recommendations": [...],
  "visualizations": [...],
  "suggestions": [...]
}
```

## Performance Considerations

- **Asynchronous Processing**: All operations are async for scalability
- **Caching**: Results are cached to improve response times
- **Batch Processing**: Multiple searches can run concurrently
- **Memory Efficient**: Streaming results for large datasets

## Error Handling

- Graceful degradation when components fail
- Language-appropriate error messages
- Fallback modes for reduced functionality
- Comprehensive logging for debugging

## Future Enhancements

1. **Machine Learning Integration**: More sophisticated behavioral analysis
2. **Voice Analysis**: Integration with speech emotion detection
3. **Visual Timeline**: Rich visual timeline interface
4. **Collaborative Analysis**: Multi-user conversation analysis
5. **Export Integration**: Direct export to note-taking applications

## Security and Privacy

- Respects memory access controls
- User data isolation
- Configurable data retention policies
- Encryption support for sensitive data

This implementation provides a foundation for advanced conversational AI capabilities with deep introspection and analysis features, making it particularly valuable for users who want to understand and improve their communication patterns over time.