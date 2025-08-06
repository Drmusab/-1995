"""
Context Time Machine Skill - Main Implementation
Author: Drmusab
Last Modified: 2025-01-08

Main skill implementation that orchestrates the Context Time Machine functionality:
- Handles user queries in Arabic and English
- Coordinates search, replay, analysis, and visualization
- Provides bilingual responses
- Integrates with existing memory and session systems
"""

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import SkillExecutionCompleted, SkillExecutionStarted
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager, Language

from .query_parser import QueryParser, ParsedQuery, QueryType
from .search_engine import SearchEngine, SearchMode
from .memory_replayer import MemoryReplayer, ReplayMode
from .reflection_analyzer import ReflectionAnalyzer, BehavioralAnalysis
from .visualization import VisualizationEngine, ChartType, ExportFormat


class TimeMachineSkill:
    """
    Context Time Machine Skill - Advanced memory and behavioral analysis.
    
    Allows users to:
    - Query past conversations in Arabic and English
    - Replay conversation snippets and summaries
    - Analyze behavioral changes over time
    - Visualize trends and patterns
    
    Examples of supported queries:
    - "ماذا تحدثنا عنه الأسبوع الماضي؟" (What did we talk about last week?)
    - "متى كانت آخر مرة ذكرت فيها المشروع؟" (When was the last time I mentioned the project?)
    - "هل تحسن أسلوبي في الاجتماعات؟" (Have I improved my style in meetings?)
    """
    
    def __init__(self, container: Container):
        """Initialize the Time Machine skill."""
        self.container = container
        self.logger = get_logger(__name__)
        
        # Core dependencies
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.memory_manager = container.get(MemoryManager)
        self.bilingual_manager = container.get(BilingualManager)
        
        # Time Machine components
        self.query_parser = QueryParser(container)
        self.search_engine = SearchEngine(container)
        self.memory_replayer = MemoryReplayer(container)
        self.reflection_analyzer = ReflectionAnalyzer(container)
        self.visualization_engine = VisualizationEngine(container)
        
        self.logger.info("TimeMachineSkill initialized successfully")
    
    @handle_exceptions
    async def process_query(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_visualization: bool = True,
        replay_mode: ReplayMode = ReplayMode.SUMMARY,
        search_mode: SearchMode = SearchMode.HYBRID
    ) -> Dict[str, Any]:
        """
        Process a time machine query and return comprehensive results.
        
        Args:
            user_query: Natural language query from user
            user_id: User identifier
            session_id: Session identifier
            include_visualization: Whether to include chart visualizations
            replay_mode: Mode for conversation replay
            search_mode: Mode for memory search
            
        Returns:
            Comprehensive response with search results, analysis, and insights
        """
        start_time = datetime.now(timezone.utc)
        
        # Emit start event
        await self.event_bus.emit(
            SkillExecutionStarted(
                skill_name="time_machine",
                session_id=session_id or "unknown",
                user_id=user_id,
                parameters={
                    "query": user_query,
                    "replay_mode": replay_mode.value,
                    "search_mode": search_mode.value
                }
            )
        )
        
        try:
            # Step 1: Parse the query
            parsed_query = await self.query_parser.parse_query(user_query)
            
            # Step 2: Search for relevant memories
            search_response = await self.search_engine.search(
                parsed_query=parsed_query,
                mode=search_mode,
                limit=50  # Get more results for better analysis
            )
            
            # Step 3: Replay conversations based on query type
            replay_response = None
            if parsed_query.query_type in [
                QueryType.CONVERSATION_RECALL,
                QueryType.PROJECT_MENTION,
                QueryType.TEMPORAL_SUMMARY
            ]:
                replay_response = await self.memory_replayer.replay_conversations(
                    search_results=search_response.results,
                    mode=replay_mode,
                    language=parsed_query.language,
                    max_threads=10
                )
            
            # Step 4: Behavioral analysis if requested
            behavioral_analysis = None
            if parsed_query.query_type in [
                QueryType.BEHAVIORAL_ANALYSIS,
                QueryType.MOOD_TRACKING,
                QueryType.PROGRESS_TRACKING
            ]:
                if replay_response and replay_response.conversation_threads:
                    behavioral_analysis = await self.reflection_analyzer.analyze_behavioral_patterns(
                        conversation_threads=replay_response.conversation_threads,
                        language=parsed_query.language,
                        user_id=user_id
                    )
            
            # Step 5: Generate visualizations if requested
            visualizations = []
            if include_visualization and behavioral_analysis:
                try:
                    # Create trends chart
                    trends_viz = await self.visualization_engine.create_behavioral_trends_chart(
                        analysis=behavioral_analysis,
                        chart_type=ChartType.LINE_CHART,
                        export_format=ExportFormat.JSON
                    )
                    visualizations.append(trends_viz)
                    
                    # Create comparison chart
                    comparison_viz = await self.visualization_engine.create_metric_comparison_chart(
                        analysis=behavioral_analysis,
                        export_format=ExportFormat.JSON
                    )
                    visualizations.append(comparison_viz)
                    
                    # Create progress chart if there are improvements
                    progress_viz = await self.visualization_engine.create_progress_visualization(
                        analysis=behavioral_analysis,
                        export_format=ExportFormat.ASCII  # ASCII for simplicity
                    )
                    visualizations.append(progress_viz)
                    
                except Exception as viz_error:
                    self.logger.warning(f"Visualization generation failed: {str(viz_error)}")
            
            # Step 6: Compile response
            response = await self._compile_response(
                parsed_query=parsed_query,
                search_response=search_response,
                replay_response=replay_response,
                behavioral_analysis=behavioral_analysis,
                visualizations=visualizations,
                user_id=user_id
            )
            
            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Emit completion event
            await self.event_bus.emit(
                SkillExecutionCompleted(
                    skill_name="time_machine",
                    session_id=session_id or "unknown",
                    user_id=user_id,
                    result={
                        "query_type": parsed_query.query_type.value,
                        "results_count": len(search_response.results),
                        "language": parsed_query.language.value
                    },
                    processing_time=processing_time
                )
            )
            
            self.logger.info(
                f"Time machine query processed successfully: "
                f"{parsed_query.query_type.value} in {processing_time:.2f}s"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Time machine query processing failed: {str(e)}")
            
            # Return error response in appropriate language
            error_language = Language.ARABIC
            try:
                error_language = self.bilingual_manager.detect_language(user_query)
            except:
                pass
            
            error_response = await self._create_error_response(str(e), error_language)
            
            # Emit completion event with error
            await self.event_bus.emit(
                SkillExecutionCompleted(
                    skill_name="time_machine",
                    session_id=session_id or "unknown",
                    user_id=user_id,
                    result={"error": str(e)},
                    processing_time=(datetime.now(timezone.utc) - start_time).total_seconds()
                )
            )
            
            return error_response
    
    async def _compile_response(
        self,
        parsed_query: ParsedQuery,
        search_response,
        replay_response=None,
        behavioral_analysis=None,
        visualizations=None,
        user_id=None
    ) -> Dict[str, Any]:
        """Compile a comprehensive response based on all analysis results."""
        language = parsed_query.language
        
        # Base response structure
        response = {
            "query_analysis": {
                "original_query": parsed_query.original_text,
                "detected_language": language.value,
                "query_type": parsed_query.query_type.value,
                "confidence": parsed_query.intent_confidence,
                "time_range": None,
                "topics": list(parsed_query.topics),
                "entities": list(parsed_query.entities)
            },
            "search_results": {
                "total_found": search_response.total_count,
                "search_time": search_response.search_time,
                "results": []
            },
            "answer": "",
            "insights": [],
            "recommendations": [],
            "visualizations": visualizations or [],
            "metadata": {
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id
            }
        }
        
        # Add time range if present
        if parsed_query.time_range:
            response["query_analysis"]["time_range"] = {
                "start": parsed_query.time_range.start.isoformat() if parsed_query.time_range.start else None,
                "end": parsed_query.time_range.end.isoformat() if parsed_query.time_range.end else None,
                "type": parsed_query.time_range.reference_type.value
            }
        
        # Process search results
        for result in search_response.results[:10]:  # Limit displayed results
            response["search_results"]["results"].append({
                "content_snippet": result.snippet,
                "timestamp": result.timestamp.isoformat(),
                "relevance_score": result.relevance_score,
                "memory_type": result.memory_type.value,
                "highlights": result.highlights
            })
        
        # Generate main answer based on query type
        response["answer"] = await self._generate_main_answer(
            parsed_query, search_response, replay_response, behavioral_analysis
        )
        
        # Add replay information if available
        if replay_response:
            response["conversations"] = {
                "total_threads": replay_response.total_threads,
                "time_range": {
                    "start": replay_response.time_range[0].isoformat(),
                    "end": replay_response.time_range[1].isoformat()
                },
                "threads": []
            }
            
            for thread in replay_response.conversation_threads[:5]:  # Limit displayed threads
                response["conversations"]["threads"].append({
                    "title": thread.title,
                    "summary": thread.summary,
                    "start_time": thread.start_time.isoformat(),
                    "duration_minutes": thread.total_duration / 60,
                    "topics": list(thread.topics),
                    "key_insights": thread.key_insights,
                    "segments_count": len(thread.segments)
                })
            
            response["insights"].extend(replay_response.insights)
        
        # Add behavioral analysis if available
        if behavioral_analysis:
            response["behavioral_analysis"] = {
                "confidence_score": behavioral_analysis.confidence_score,
                "analysis_period": {
                    "start": behavioral_analysis.analysis_period[0].isoformat(),
                    "end": behavioral_analysis.analysis_period[1].isoformat()
                },
                "trends": []
            }
            
            for trend in behavioral_analysis.trends:
                response["behavioral_analysis"]["trends"].append({
                    "metric": trend.metric.value,
                    "direction": trend.direction.value,
                    "change_magnitude": trend.change_magnitude,
                    "confidence": trend.confidence,
                    "insights": trend.insights
                })
            
            response["insights"].extend(behavioral_analysis.overall_insights)
            response["recommendations"].extend(behavioral_analysis.recommendations)
        
        # Add search suggestions
        response["suggestions"] = search_response.suggestions
        
        return response
    
    async def _generate_main_answer(
        self,
        parsed_query: ParsedQuery,
        search_response,
        replay_response=None,
        behavioral_analysis=None
    ) -> str:
        """Generate the main answer text based on query type and results."""
        language = parsed_query.language
        query_type = parsed_query.query_type
        
        if language == Language.ARABIC:
            if query_type == QueryType.CONVERSATION_RECALL:
                if replay_response and replay_response.conversation_threads:
                    thread_count = len(replay_response.conversation_threads)
                    return f"وجدت {thread_count} محادثة في الفترة المحددة. إليك ملخص لأهم المواضيع التي تم مناقشتها."
                else:
                    return "لم أجد محادثات في الفترة المحددة. قد تحتاج إلى توسيع نطاق البحث."
            
            elif query_type == QueryType.PROJECT_MENTION:
                if search_response.results:
                    latest_result = search_response.results[0]
                    return f"آخر مرة تم ذكر هذا الموضوع كانت في {latest_result.timestamp.strftime('%Y-%m-%d')}."
                else:
                    return "لم أجد أي ذكر لهذا الموضوع في المحادثات المسجلة."
            
            elif query_type == QueryType.BEHAVIORAL_ANALYSIS:
                if behavioral_analysis:
                    improving_trends = [t for t in behavioral_analysis.trends if t.direction.value == "improving"]
                    if improving_trends:
                        return f"تحليل السلوك يظهر تحسناً في {len(improving_trends)} مؤشرات. الثقة في النتائج: {behavioral_analysis.confidence_score:.0%}."
                    else:
                        return "تحليل السلوك يظهر استقراراً عاماً في الأداء."
                else:
                    return "لا توجد بيانات كافية لتحليل السلوك في الفترة المحددة."
            
            elif query_type == QueryType.MOOD_TRACKING:
                if behavioral_analysis:
                    mood_trends = [t for t in behavioral_analysis.trends if t.metric.value == "mood"]
                    if mood_trends:
                        trend = mood_trends[0]
                        if trend.direction.value == "improving":
                            return "تحليل المزاج يظهر تحسناً تدريجياً في الحالة النفسية."
                        elif trend.direction.value == "declining":
                            return "تحليل المزاج يظهر انخفاضاً في الحالة النفسية."
                        else:
                            return "المزاج مستقر بشكل عام."
                    else:
                        return "لا توجد بيانات كافية لتحليل المزاج."
                else:
                    return "لا توجد بيانات كافية لتحليل المزاج في الفترة المحددة."
            
            elif query_type == QueryType.TEMPORAL_SUMMARY:
                if replay_response:
                    total_time = sum(thread.total_duration for thread in replay_response.conversation_threads) / 3600
                    topics = set()
                    for thread in replay_response.conversation_threads:
                        topics.update(thread.topics)
                    return f"في الفترة المحددة، أجريت محادثات لمدة إجمالية {total_time:.1f} ساعة، تناولت {len(topics)} مواضيع مختلفة."
                else:
                    return "لا توجد محادثات في الفترة المحددة للتلخيص."
            
            elif query_type == QueryType.TOPIC_SEARCH:
                if search_response.results:
                    return f"وجدت {len(search_response.results)} نتيجة متعلقة بالموضوع المطلوب."
                else:
                    return "لم أجد أي نتائج متعلقة بالموضوع المطلوب."
            
            else:
                # General search
                if search_response.results:
                    return f"وجدت {len(search_response.results)} نتيجة ذات صلة باستعلامك."
                else:
                    return "لم أجد نتائج تطابق استعلامك. حاول استخدام كلمات مختلفة."
        
        else:  # English
            if query_type == QueryType.CONVERSATION_RECALL:
                if replay_response and replay_response.conversation_threads:
                    thread_count = len(replay_response.conversation_threads)
                    return f"Found {thread_count} conversations in the specified period. Here's a summary of the main topics discussed."
                else:
                    return "No conversations found in the specified period. You may need to expand the search scope."
            
            elif query_type == QueryType.PROJECT_MENTION:
                if search_response.results:
                    latest_result = search_response.results[0]
                    return f"The last mention of this topic was on {latest_result.timestamp.strftime('%Y-%m-%d')}."
                else:
                    return "No mentions of this topic found in recorded conversations."
            
            elif query_type == QueryType.BEHAVIORAL_ANALYSIS:
                if behavioral_analysis:
                    improving_trends = [t for t in behavioral_analysis.trends if t.direction.value == "improving"]
                    if improving_trends:
                        return f"Behavioral analysis shows improvement in {len(improving_trends)} metrics. Confidence: {behavioral_analysis.confidence_score:.0%}."
                    else:
                        return "Behavioral analysis shows general stability in performance."
                else:
                    return "Insufficient data for behavioral analysis in the specified period."
            
            elif query_type == QueryType.MOOD_TRACKING:
                if behavioral_analysis:
                    mood_trends = [t for t in behavioral_analysis.trends if t.metric.value == "mood"]
                    if mood_trends:
                        trend = mood_trends[0]
                        if trend.direction.value == "improving":
                            return "Mood analysis shows gradual improvement in emotional state."
                        elif trend.direction.value == "declining":
                            return "Mood analysis shows decline in emotional state."
                        else:
                            return "Mood is generally stable."
                    else:
                        return "Insufficient data for mood analysis."
                else:
                    return "Insufficient data for mood analysis in the specified period."
            
            elif query_type == QueryType.TEMPORAL_SUMMARY:
                if replay_response:
                    total_time = sum(thread.total_duration for thread in replay_response.conversation_threads) / 3600
                    topics = set()
                    for thread in replay_response.conversation_threads:
                        topics.update(thread.topics)
                    return f"During the specified period, you had conversations totaling {total_time:.1f} hours, covering {len(topics)} different topics."
                else:
                    return "No conversations found in the specified period to summarize."
            
            elif query_type == QueryType.TOPIC_SEARCH:
                if search_response.results:
                    return f"Found {len(search_response.results)} results related to the requested topic."
                else:
                    return "No results found related to the requested topic."
            
            else:
                # General search
                if search_response.results:
                    return f"Found {len(search_response.results)} results relevant to your query."
                else:
                    return "No results found matching your query. Try using different keywords."
    
    async def _create_error_response(self, error_message: str, language: Language) -> Dict[str, Any]:
        """Create an error response in the appropriate language."""
        if language == Language.ARABIC:
            answer = f"عذراً، حدث خطأ أثناء معالجة طلبك: {error_message}"
            insights = ["يرجى المحاولة مرة أخرى أو تبسيط الاستعلام"]
        else:
            answer = f"Sorry, an error occurred while processing your request: {error_message}"
            insights = ["Please try again or simplify your query"]
        
        return {
            "query_analysis": {
                "detected_language": language.value,
                "error": error_message
            },
            "search_results": {
                "total_found": 0,
                "results": []
            },
            "answer": answer,
            "insights": insights,
            "recommendations": [],
            "visualizations": [],
            "metadata": {
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "error": True
            }
        }
    
    async def get_skill_info(self) -> Dict[str, Any]:
        """Get information about the Time Machine skill."""
        return {
            "name": "time_machine",
            "version": "1.0.0",
            "description": {
                "ar": "آلة الزمن السياقية - تحليل المحادثات والذكريات والسلوك",
                "en": "Context Time Machine - conversation, memory, and behavioral analysis"
            },
            "capabilities": [
                "conversation_recall",
                "memory_search",
                "behavioral_analysis",
                "trend_visualization",
                "bilingual_support"
            ],
            "supported_languages": ["ar", "en"],
            "example_queries": {
                "ar": [
                    "ماذا تحدثنا عنه الأسبوع الماضي؟",
                    "متى كانت آخر مرة ذكرت فيها المشروع؟",
                    "هل تحسن أسلوبي في الاجتماعات؟",
                    "كيف كان مزاجي في الشهر الماضي؟"
                ],
                "en": [
                    "What did we talk about last week?",
                    "When was the last time I mentioned the project?",
                    "Have I improved my style in meetings?",
                    "How was my mood last month?"
                ]
            },
            "query_types": [qt.value for qt in QueryType],
            "replay_modes": [rm.value for rm in ReplayMode],
            "search_modes": [sm.value for sm in SearchMode]
        }