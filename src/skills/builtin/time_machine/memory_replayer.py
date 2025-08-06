"""
Memory Replayer for Context Time Machine
Author: Drmusab
Last Modified: 2025-01-08

Reconstructs and replays past conversations and interactions:
- Conversation timeline reconstruction
- Summarization of conversation sequences
- Context preservation and linking
- Multiple replay modes and formats
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.dependency_injection import Container
from src.memory.core_memory.base_memory import MemoryItem, MemoryType
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager, Language

from .search_engine import SearchResult


class ReplayMode(Enum):
    """Different modes for conversation replay."""
    FULL = "full"                    # Complete conversation replay
    SUMMARY = "summary"              # Summarized version
    HIGHLIGHTS = "highlights"        # Key moments only
    TIMELINE = "timeline"           # Chronological timeline view
    CONTEXTUAL = "contextual"       # Context-aware reconstruction


@dataclass
class ConversationSegment:
    """A segment of conversation with metadata."""
    id: str
    content: str
    timestamp: datetime
    participants: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    sentiment: Optional[Dict[str, Any]] = None
    topics: List[str] = field(default_factory=list)
    importance: float = 0.5
    memory_ids: List[str] = field(default_factory=list)


@dataclass
class ConversationThread:
    """A complete conversation thread with multiple segments."""
    id: str
    title: str
    segments: List[ConversationSegment]
    participants: Set[str]
    start_time: datetime
    end_time: datetime
    total_duration: float
    context: Dict[str, Any] = field(default_factory=dict)
    topics: Set[str] = field(default_factory=set)
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)


@dataclass
class ReplayResponse:
    """Response containing replayed conversation data."""
    conversation_threads: List[ConversationThread]
    total_threads: int
    time_range: Tuple[datetime, datetime]
    replay_mode: ReplayMode
    language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


class MemoryReplayer:
    """
    Replays and reconstructs past conversations and interactions.
    
    Provides different modes of conversation replay, from full transcripts
    to summarized highlights, with proper context preservation and
    bilingual support.
    """
    
    def __init__(self, container: Container):
        """Initialize the memory replayer."""
        self.container = container
        self.logger = get_logger(__name__)
        self.memory_manager = container.get(MemoryManager)
        self.bilingual_manager = container.get(BilingualManager)
        
        # Configuration
        self.max_segment_gap = 300  # 5 minutes between segments
        self.min_thread_duration = 60  # 1 minute minimum thread duration
        self.max_summary_length = 500  # Maximum summary length
        
        self.logger.info("MemoryReplayer initialized")
    
    async def replay_conversations(
        self,
        search_results: List[SearchResult],
        mode: ReplayMode = ReplayMode.SUMMARY,
        language: Language = Language.ARABIC,
        max_threads: int = 10
    ) -> ReplayResponse:
        """
        Replay conversations from search results.
        
        Args:
            search_results: Results from search engine
            mode: Replay mode to use
            language: Language for response
            max_threads: Maximum conversation threads to return
            
        Returns:
            ReplayResponse with conversation threads
        """
        try:
            # Group search results into conversation threads
            threads = await self._group_into_threads(search_results)
            
            # Process threads based on mode
            processed_threads = []
            for thread in threads[:max_threads]:
                processed_thread = await self._process_thread(thread, mode, language)
                if processed_thread:
                    processed_threads.append(processed_thread)
            
            # Generate insights
            insights = await self._generate_insights(processed_threads, language)
            
            # Calculate time range
            if processed_threads:
                start_time = min(thread.start_time for thread in processed_threads)
                end_time = max(thread.end_time for thread in processed_threads)
            else:
                now = datetime.now(timezone.utc)
                start_time = end_time = now
            
            response = ReplayResponse(
                conversation_threads=processed_threads,
                total_threads=len(threads),
                time_range=(start_time, end_time),
                replay_mode=mode,
                language=language,
                metadata={
                    "total_segments": sum(len(thread.segments) for thread in processed_threads),
                    "average_duration": (
                        sum(thread.total_duration for thread in processed_threads) / 
                        len(processed_threads)
                    ) if processed_threads else 0,
                    "topics_covered": len(set().union(*(thread.topics for thread in processed_threads)))
                },
                insights=insights
            )
            
            self.logger.debug(f"Replayed {len(processed_threads)} conversation threads")
            return response
            
        except Exception as e:
            self.logger.error(f"Conversation replay failed: {str(e)}")
            return ReplayResponse(
                conversation_threads=[],
                total_threads=0,
                time_range=(datetime.now(timezone.utc), datetime.now(timezone.utc)),
                replay_mode=mode,
                language=language,
                metadata={"error": str(e)}
            )
    
    async def _group_into_threads(self, search_results: List[SearchResult]) -> List[ConversationThread]:
        """Group search results into conversation threads."""
        threads = []
        current_thread = None
        
        # Sort results by timestamp
        sorted_results = sorted(search_results, key=lambda x: x.timestamp)
        
        for result in sorted_results:
            # Create conversation segment from search result
            segment = await self._create_segment_from_result(result)
            
            if current_thread is None:
                # Start new thread
                current_thread = ConversationThread(
                    id=f"thread_{len(threads)}",
                    title="",
                    segments=[segment],
                    participants=set(),
                    start_time=segment.timestamp,
                    end_time=segment.timestamp,
                    total_duration=0,
                    topics=set(segment.topics)
                )
            else:
                # Check if this segment belongs to current thread
                time_gap = (segment.timestamp - current_thread.end_time).total_seconds()
                
                if time_gap <= self.max_segment_gap:
                    # Add to current thread
                    current_thread.segments.append(segment)
                    current_thread.end_time = segment.timestamp
                    current_thread.topics.update(segment.topics)
                else:
                    # Finalize current thread and start new one
                    if current_thread.segments:
                        current_thread = await self._finalize_thread(current_thread)
                        threads.append(current_thread)
                    
                    # Start new thread
                    current_thread = ConversationThread(
                        id=f"thread_{len(threads)}",
                        title="",
                        segments=[segment],
                        participants=set(),
                        start_time=segment.timestamp,
                        end_time=segment.timestamp,
                        total_duration=0,
                        topics=set(segment.topics)
                    )
        
        # Finalize last thread
        if current_thread and current_thread.segments:
            current_thread = await self._finalize_thread(current_thread)
            threads.append(current_thread)
        
        return threads
    
    async def _create_segment_from_result(self, result: SearchResult) -> ConversationSegment:
        """Create a conversation segment from a search result."""
        # Extract content
        content = str(result.content)
        if isinstance(result.content, dict):
            # Handle structured content
            if "transcription" in result.content:
                content = result.content["transcription"]
            elif "text" in result.content:
                content = result.content["text"]
            else:
                content = json.dumps(result.content, ensure_ascii=False, indent=2)
        
        # Extract participants (simplified)
        participants = []
        if result.metadata.get("user_id"):
            participants.append(result.metadata["user_id"])
        participants.append("assistant")  # Always include assistant
        
        # Extract topics from content and metadata
        topics = []
        if result.metadata.get("tags"):
            topics.extend(result.metadata["tags"])
        
        # Simple topic extraction from content
        if "مشروع" in content or "project" in content.lower():
            topics.append("project")
        if "اجتماع" in content or "meeting" in content.lower():
            topics.append("meeting")
        if "برمجة" in content or "programming" in content.lower():
            topics.append("programming")
        
        segment = ConversationSegment(
            id=result.memory_id,
            content=content,
            timestamp=result.timestamp,
            participants=participants,
            context={
                "memory_type": result.memory_type.value,
                "relevance_score": result.relevance_score
            },
            topics=topics,
            importance=result.metadata.get("importance", 0.5),
            memory_ids=[result.memory_id]
        )
        
        return segment
    
    async def _finalize_thread(self, thread: ConversationThread) -> ConversationThread:
        """Finalize a conversation thread by calculating metadata."""
        if not thread.segments:
            return thread
        
        # Calculate duration
        thread.total_duration = (thread.end_time - thread.start_time).total_seconds()
        
        # Extract participants
        for segment in thread.segments:
            thread.participants.update(segment.participants)
        
        # Generate title
        thread.title = await self._generate_thread_title(thread)
        
        # Generate summary if needed
        if len(thread.segments) > 1:
            thread.summary = await self._generate_thread_summary(thread)
        
        # Extract key insights
        thread.key_insights = await self._extract_key_insights(thread)
        
        return thread
    
    async def _generate_thread_title(self, thread: ConversationThread) -> str:
        """Generate a title for the conversation thread."""
        if not thread.segments:
            return "Empty Conversation"
        
        # Use topics to generate title
        if thread.topics:
            main_topic = list(thread.topics)[0]
            if main_topic == "project":
                return "مناقشة المشروع"
            elif main_topic == "meeting":
                return "ملاحظات الاجتماع"
            elif main_topic == "programming":
                return "جلسة برمجة"
            else:
                return f"محادثة حول {main_topic}"
        
        # Fallback to timestamp-based title
        date_str = thread.start_time.strftime("%Y-%m-%d")
        return f"محادثة {date_str}"
    
    async def _generate_thread_summary(self, thread: ConversationThread) -> str:
        """Generate a summary of the conversation thread."""
        if not thread.segments:
            return ""
        
        # Combine all segment content
        all_content = " ".join(segment.content for segment in thread.segments)
        
        # Simple summarization (in production would use LLM)
        if len(all_content) <= self.max_summary_length:
            return all_content
        
        # Extract key sentences
        sentences = all_content.split(".")
        key_sentences = []
        total_length = 0
        
        for sentence in sentences:
            if total_length + len(sentence) <= self.max_summary_length:
                key_sentences.append(sentence.strip())
                total_length += len(sentence)
            else:
                break
        
        summary = ". ".join(key_sentences)
        if len(summary) < len(all_content):
            summary += "..."
        
        return summary
    
    async def _extract_key_insights(self, thread: ConversationThread) -> List[str]:
        """Extract key insights from the conversation thread."""
        insights = []
        
        # Duration insight
        if thread.total_duration > 3600:  # More than 1 hour
            insights.append("محادثة مطولة - أكثر من ساعة")
        elif thread.total_duration < 300:  # Less than 5 minutes
            insights.append("محادثة قصيرة - أقل من 5 دقائق")
        
        # Topic insights
        if len(thread.topics) > 3:
            insights.append(f"محادثة متنوعة - تغطي {len(thread.topics)} مواضيع")
        elif "project" in thread.topics:
            insights.append("محادثة متعلقة بالمشروع")
        
        # Participants insight
        if len(thread.participants) > 2:
            insights.append(f"محادثة جماعية - {len(thread.participants)} مشاركين")
        
        # Importance insight
        avg_importance = sum(segment.importance for segment in thread.segments) / len(thread.segments)
        if avg_importance > 0.7:
            insights.append("محادثة مهمة")
        
        return insights
    
    async def _process_thread(
        self,
        thread: ConversationThread,
        mode: ReplayMode,
        language: Language
    ) -> Optional[ConversationThread]:
        """Process a thread based on the replay mode."""
        if mode == ReplayMode.FULL:
            return await self._process_full_mode(thread, language)
        elif mode == ReplayMode.SUMMARY:
            return await self._process_summary_mode(thread, language)
        elif mode == ReplayMode.HIGHLIGHTS:
            return await self._process_highlights_mode(thread, language)
        elif mode == ReplayMode.TIMELINE:
            return await self._process_timeline_mode(thread, language)
        elif mode == ReplayMode.CONTEXTUAL:
            return await self._process_contextual_mode(thread, language)
        else:
            return thread
    
    async def _process_full_mode(
        self,
        thread: ConversationThread,
        language: Language
    ) -> ConversationThread:
        """Process thread for full replay mode."""
        # Return thread as-is for full mode
        return thread
    
    async def _process_summary_mode(
        self,
        thread: ConversationThread,
        language: Language
    ) -> ConversationThread:
        """Process thread for summary mode."""
        # Create a single summary segment
        summary_segment = ConversationSegment(
            id=f"{thread.id}_summary",
            content=thread.summary or await self._generate_thread_summary(thread),
            timestamp=thread.start_time,
            participants=list(thread.participants),
            context={"mode": "summary", "original_segments": len(thread.segments)},
            topics=list(thread.topics),
            importance=max(segment.importance for segment in thread.segments) if thread.segments else 0.5
        )
        
        # Return thread with single summary segment
        processed_thread = ConversationThread(
            id=thread.id,
            title=thread.title,
            segments=[summary_segment],
            participants=thread.participants,
            start_time=thread.start_time,
            end_time=thread.end_time,
            total_duration=thread.total_duration,
            context=thread.context,
            topics=thread.topics,
            summary=thread.summary,
            key_insights=thread.key_insights
        )
        
        return processed_thread
    
    async def _process_highlights_mode(
        self,
        thread: ConversationThread,
        language: Language
    ) -> ConversationThread:
        """Process thread for highlights mode."""
        # Filter segments by importance
        important_segments = [
            segment for segment in thread.segments
            if segment.importance > 0.6
        ]
        
        # If no important segments, take top 3 by importance
        if not important_segments:
            important_segments = sorted(
                thread.segments,
                key=lambda x: x.importance,
                reverse=True
            )[:3]
        
        # Return thread with only important segments
        processed_thread = ConversationThread(
            id=thread.id,
            title=thread.title,
            segments=important_segments,
            participants=thread.participants,
            start_time=thread.start_time,
            end_time=thread.end_time,
            total_duration=thread.total_duration,
            context=thread.context,
            topics=thread.topics,
            summary=thread.summary,
            key_insights=thread.key_insights
        )
        
        return processed_thread
    
    async def _process_timeline_mode(
        self,
        thread: ConversationThread,
        language: Language
    ) -> ConversationThread:
        """Process thread for timeline mode."""
        # Create timeline-formatted segments
        timeline_segments = []
        
        for i, segment in enumerate(thread.segments):
            timeline_content = (
                f"{segment.timestamp.strftime('%H:%M:%S')} - "
                f"{segment.content[:100]}..."
                if len(segment.content) > 100 else segment.content
            )
            
            timeline_segment = ConversationSegment(
                id=f"{segment.id}_timeline",
                content=timeline_content,
                timestamp=segment.timestamp,
                participants=segment.participants,
                context={**segment.context, "timeline_index": i},
                topics=segment.topics,
                importance=segment.importance
            )
            timeline_segments.append(timeline_segment)
        
        # Return thread with timeline segments
        processed_thread = ConversationThread(
            id=thread.id,
            title=thread.title,
            segments=timeline_segments,
            participants=thread.participants,
            start_time=thread.start_time,
            end_time=thread.end_time,
            total_duration=thread.total_duration,
            context=thread.context,
            topics=thread.topics,
            summary=thread.summary,
            key_insights=thread.key_insights
        )
        
        return processed_thread
    
    async def _process_contextual_mode(
        self,
        thread: ConversationThread,
        language: Language
    ) -> ConversationThread:
        """Process thread for contextual mode."""
        # Group segments by context/topic
        contextual_segments = []
        topics_covered = list(thread.topics)
        
        for topic in topics_covered:
            topic_segments = [
                segment for segment in thread.segments
                if topic in segment.topics
            ]
            
            if topic_segments:
                # Create a context segment for this topic
                combined_content = f"المحادثات حول {topic}:\n"
                combined_content += "\n".join(
                    f"• {segment.content[:200]}..."
                    if len(segment.content) > 200 else f"• {segment.content}"
                    for segment in topic_segments
                )
                
                context_segment = ConversationSegment(
                    id=f"{thread.id}_{topic}_context",
                    content=combined_content,
                    timestamp=topic_segments[0].timestamp,
                    participants=list(set().union(*(set(seg.participants) for seg in topic_segments))),
                    context={"topic": topic, "source_segments": len(topic_segments)},
                    topics=[topic],
                    importance=max(seg.importance for seg in topic_segments)
                )
                contextual_segments.append(context_segment)
        
        # If no topic-based segments, fall back to original segments
        if not contextual_segments:
            contextual_segments = thread.segments
        
        # Return thread with contextual segments
        processed_thread = ConversationThread(
            id=thread.id,
            title=thread.title,
            segments=contextual_segments,
            participants=thread.participants,
            start_time=thread.start_time,
            end_time=thread.end_time,
            total_duration=thread.total_duration,
            context=thread.context,
            topics=thread.topics,
            summary=thread.summary,
            key_insights=thread.key_insights
        )
        
        return processed_thread
    
    async def _generate_insights(
        self,
        threads: List[ConversationThread],
        language: Language
    ) -> List[str]:
        """Generate insights from conversation threads."""
        insights = []
        
        if not threads:
            return insights
        
        # Time-based insights
        total_duration = sum(thread.total_duration for thread in threads)
        avg_duration = total_duration / len(threads)
        
        if language == Language.ARABIC:
            insights.append(f"إجمالي وقت المحادثات: {total_duration / 3600:.1f} ساعة")
            insights.append(f"متوسط مدة المحادثة: {avg_duration / 60:.0f} دقيقة")
        else:
            insights.append(f"Total conversation time: {total_duration / 3600:.1f} hours")
            insights.append(f"Average conversation duration: {avg_duration / 60:.0f} minutes")
        
        # Topic insights
        all_topics = set().union(*(thread.topics for thread in threads))
        if language == Language.ARABIC:
            insights.append(f"المواضيع المختلفة: {len(all_topics)}")
            if "project" in all_topics:
                insights.append("تركيز كبير على مناقشات المشروع")
        else:
            insights.append(f"Different topics covered: {len(all_topics)}")
            if "project" in all_topics:
                insights.append("Strong focus on project discussions")
        
        # Frequency insights
        if len(threads) > 5:
            if language == Language.ARABIC:
                insights.append("محادثات متكررة - تفاعل نشط")
            else:
                insights.append("Frequent conversations - active engagement")
        
        return insights