"""
Query Parser for Context Time Machine
Author: Drmusab
Last Modified: 2025-01-08

Parses natural language queries in Arabic and English to extract:
- Time ranges and references
- Intent and query type
- Topics and entities
- Behavioral queries

Supports bilingual query understanding with proper Arabic language processing.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager, Language


class QueryType(Enum):
    """Types of time machine queries."""
    CONVERSATION_RECALL = "conversation_recall"  # "ماذا تحدثنا عنه الأسبوع الماضي؟"
    PROJECT_MENTION = "project_mention"  # "متى كانت آخر مرة ذكرت فيها المشروع؟"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"  # "هل تحسن أسلوبي في الاجتماعات؟"
    TOPIC_SEARCH = "topic_search"  # "ابحث عن محادثات حول الذكاء الاصطناعي"
    TEMPORAL_SUMMARY = "temporal_summary"  # "لخص المحادثات من الشهر الماضي"
    MOOD_TRACKING = "mood_tracking"  # "كيف كان مزاجي في الأسبوع الماضي؟"
    PROGRESS_TRACKING = "progress_tracking"  # "ما التقدم في مشروعي؟"
    GENERAL_SEARCH = "general_search"  # Fallback for unclear queries


class TimeReferenceType(Enum):
    """Types of time references in queries."""
    ABSOLUTE = "absolute"  # "في 15 يناير"
    RELATIVE = "relative"  # "الأسبوع الماضي"
    RANGE = "range"  # "من الاثنين إلى الجمعة"
    FUZZY = "fuzzy"  # "مؤخراً", "منذ فترة"


@dataclass
class TimeRange:
    """Represents a time range extracted from a query."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    reference_type: TimeReferenceType = TimeReferenceType.FUZZY
    original_text: str = ""
    confidence: float = 0.0


@dataclass
class ParsedQuery:
    """Represents a parsed natural language query."""
    original_text: str
    language: Language
    query_type: QueryType
    time_range: Optional[TimeRange] = None
    topics: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    behavioral_aspects: Set[str] = field(default_factory=set)
    intent_confidence: float = 0.0
    extracted_keywords: Set[str] = field(default_factory=set)
    context_hints: Dict[str, Any] = field(default_factory=dict)


class QueryParser:
    """
    Parses natural language queries for the Context Time Machine.
    
    Handles both Arabic and English queries, extracting time references,
    topics, entities, and behavioral analysis requests.
    """
    
    def __init__(self, container: Container):
        """Initialize the query parser."""
        self.container = container
        self.logger = get_logger(__name__)
        self.bilingual_manager = container.get(BilingualManager)
        
        # Initialize pattern dictionaries
        self._setup_patterns()
        
        self.logger.info("QueryParser initialized with bilingual support")
    
    def _setup_patterns(self) -> None:
        """Setup regex patterns for different query types and time references."""
        
        # Arabic time patterns
        self.arabic_time_patterns = {
            # Relative time references
            'last_week': [
                r'الأسبوع\s+الماضي', r'الأسبوع\s+اللي\s+فات', r'آخر\s+أسبوع'
            ],
            'last_month': [
                r'الشهر\s+الماضي', r'الشهر\s+اللي\s+فات', r'آخر\s+شهر'
            ],
            'yesterday': [
                r'أمس', r'البارحة', r'يوم\s+أمس'
            ],
            'today': [
                r'اليوم', r'النهارده'
            ],
            'last_year': [
                r'السنة\s+الماضية', r'العام\s+الماضي', r'آخر\s+سنة'
            ],
            'recently': [
                r'مؤخراً', r'في\s+الآونة\s+الأخيرة', r'منذ\s+فترة\s+قريبة'
            ],
            'since': [
                r'منذ', r'من\s+وقت', r'من\s+بعد'
            ]
        }
        
        # English time patterns
        self.english_time_patterns = {
            'last_week': [
                r'last\s+week', r'previous\s+week', r'a\s+week\s+ago'
            ],
            'last_month': [
                r'last\s+month', r'previous\s+month', r'a\s+month\s+ago'
            ],
            'yesterday': [
                r'yesterday', r'last\s+day'
            ],
            'today': [
                r'today', r'this\s+day'
            ],
            'last_year': [
                r'last\s+year', r'previous\s+year', r'a\s+year\s+ago'
            ],
            'recently': [
                r'recently', r'lately', r'not\s+long\s+ago'
            ],
            'since': [
                r'since', r'from\s+the\s+time', r'after'
            ]
        }
        
        # Arabic query type patterns
        self.arabic_query_patterns = {
            QueryType.CONVERSATION_RECALL: [
                r'ماذا\s+تحدثنا', r'عن\s+ماذا\s+تكلمنا', r'ما\s+الذي\s+ناقشناه',
                r'عن\s+ماذا\s+تحادثنا', r'ما\s+هي\s+المواضيع\s+التي\s+تطرقنا'
            ],
            QueryType.PROJECT_MENTION: [
                r'متى\s+.*\s+ذكرت', r'آخر\s+مرة\s+.*\s+المشروع', r'متى\s+تحدثت\s+عن',
                r'متى\s+كان\s+.*\s+المشروع'
            ],
            QueryType.BEHAVIORAL_ANALYSIS: [
                r'هل\s+تحسن\s+.*\s+أسلوبي', r'كيف\s+.*\s+سلوكي', r'هل\s+.*\s+تقدم',
                r'تحليل\s+سلوكي', r'تقييم\s+الأداء'
            ],
            QueryType.MOOD_TRACKING: [
                r'كيف\s+كان\s+مزاجي', r'ما\s+هو\s+مزاجي', r'حالتي\s+النفسية',
                r'مشاعري', r'عواطفي'
            ],
            QueryType.TOPIC_SEARCH: [
                r'ابحث\s+عن', r'اعثر\s+على', r'أريد\s+.*\s+محادثات',
                r'محادثات\s+حول', r'موضوع'
            ],
            QueryType.TEMPORAL_SUMMARY: [
                r'لخص\s+.*\s+المحادثات', r'ملخص\s+.*\s+الفترة', r'خلاصة\s+ما\s+حدث'
            ]
        }
        
        # English query type patterns
        self.english_query_patterns = {
            QueryType.CONVERSATION_RECALL: [
                r'what\s+did\s+we\s+talk\s+about', r'what\s+did\s+we\s+discuss',
                r'what\s+were\s+we\s+talking\s+about', r'conversation\s+topics'
            ],
            QueryType.PROJECT_MENTION: [
                r'when\s+.*\s+mentioned', r'last\s+time\s+.*\s+project',
                r'when\s+did\s+.*\s+talk\s+about', r'mention\s+of'
            ],
            QueryType.BEHAVIORAL_ANALYSIS: [
                r'have\s+I\s+improved', r'how\s+.*\s+behavior', r'progress\s+in',
                r'behavioral\s+analysis', r'performance\s+evaluation'
            ],
            QueryType.MOOD_TRACKING: [
                r'how\s+was\s+my\s+mood', r'what\s+was\s+my\s+mood', r'emotional\s+state',
                r'feelings', r'emotions'
            ],
            QueryType.TOPIC_SEARCH: [
                r'search\s+for', r'find\s+conversations', r'look\s+for',
                r'conversations\s+about', r'topic', r'discuss.*about'
            ],
            QueryType.TEMPORAL_SUMMARY: [
                r'summarize\s+.*\s+conversations', r'summary\s+of.*period',
                r'what\s+happened\s+during'
            ]
        }
        
        # Behavioral aspects patterns
        self.behavioral_patterns = {
            'tone': {
                'ar': [r'نبرة', r'أسلوب', r'طريقة\s+الكلام'],
                'en': [r'tone', r'style', r'way\s+of\s+speaking']
            },
            'mood': {
                'ar': [r'مزاج', r'حالة\s+نفسية', r'مشاعر'],
                'en': [r'mood', r'emotional\s+state', r'feelings']
            },
            'energy': {
                'ar': [r'طاقة', r'حيوية', r'نشاط'],
                'en': [r'energy', r'vitality', r'activity']
            },
            'confidence': {
                'ar': [r'ثقة', r'اعتماد\s+على\s+النفس'],
                'en': [r'confidence', r'self-assurance']
            },
            'engagement': {
                'ar': [r'مشاركة', r'تفاعل', r'انخراط'],
                'en': [r'engagement', r'participation', r'involvement']
            }
        }
    
    async def parse_query(self, query_text: str) -> ParsedQuery:
        """
        Parse a natural language query into structured components.
        
        Args:
            query_text: The input query text
            
        Returns:
            ParsedQuery object with extracted information
        """
        # Detect language
        language = self.bilingual_manager.detect_language(query_text)
        
        # Initialize parsed query
        parsed_query = ParsedQuery(
            original_text=query_text.strip(),
            language=language,
            query_type=QueryType.GENERAL_SEARCH
        )
        
        # Extract time range
        parsed_query.time_range = await self._extract_time_range(query_text, language)
        
        # Determine query type
        parsed_query.query_type, parsed_query.intent_confidence = await self._classify_query_type(
            query_text, language
        )
        
        # Extract topics and entities
        parsed_query.topics = await self._extract_topics(query_text, language)
        parsed_query.entities = await self._extract_entities(query_text, language)
        
        # Extract behavioral aspects
        parsed_query.behavioral_aspects = await self._extract_behavioral_aspects(
            query_text, language
        )
        
        # Extract keywords
        parsed_query.extracted_keywords = await self._extract_keywords(query_text, language)
        
        # Add context hints
        parsed_query.context_hints = await self._extract_context_hints(query_text, language)
        
        self.logger.debug(f"Parsed query: {parsed_query.query_type.value} in {language.value}")
        
        return parsed_query
    
    async def _extract_time_range(self, text: str, language: Language) -> Optional[TimeRange]:
        """Extract time range from query text."""
        time_patterns = (
            self.arabic_time_patterns if language == Language.ARABIC 
            else self.english_time_patterns
        )
        
        now = datetime.now(timezone.utc)
        
        for time_type, patterns in time_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    if time_type == 'last_week':
                        start = now - timedelta(weeks=1)
                        end = now
                        return TimeRange(
                            start=start,
                            end=end,
                            reference_type=TimeReferenceType.RELATIVE,
                            original_text=pattern,
                            confidence=0.9
                        )
                    elif time_type == 'last_month':
                        start = now - timedelta(days=30)
                        end = now
                        return TimeRange(
                            start=start,
                            end=end,
                            reference_type=TimeReferenceType.RELATIVE,
                            original_text=pattern,
                            confidence=0.9
                        )
                    elif time_type == 'yesterday':
                        start = now - timedelta(days=1)
                        end = now
                        return TimeRange(
                            start=start,
                            end=end,
                            reference_type=TimeReferenceType.RELATIVE,
                            original_text=pattern,
                            confidence=0.95
                        )
                    elif time_type == 'today':
                        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                        end = now
                        return TimeRange(
                            start=start,
                            end=end,
                            reference_type=TimeReferenceType.RELATIVE,
                            original_text=pattern,
                            confidence=0.95
                        )
                    elif time_type == 'last_year':
                        start = now - timedelta(days=365)
                        end = now
                        return TimeRange(
                            start=start,
                            end=end,
                            reference_type=TimeReferenceType.RELATIVE,
                            original_text=pattern,
                            confidence=0.8
                        )
                    elif time_type == 'recently':
                        start = now - timedelta(days=7)  # Default to last week
                        end = now
                        return TimeRange(
                            start=start,
                            end=end,
                            reference_type=TimeReferenceType.FUZZY,
                            original_text=pattern,
                            confidence=0.6
                        )
        
        return None
    
    async def _classify_query_type(self, text: str, language: Language) -> Tuple[QueryType, float]:
        """Classify the query type and return confidence score."""
        query_patterns = (
            self.arabic_query_patterns if language == Language.ARABIC 
            else self.english_query_patterns
        )
        
        best_match = QueryType.GENERAL_SEARCH
        best_confidence = 0.0
        
        for query_type, patterns in query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    confidence = 0.8 + (len(re.findall(pattern, text, re.IGNORECASE)) * 0.1)
                    confidence = min(confidence, 1.0)
                    
                    if confidence > best_confidence:
                        best_match = query_type
                        best_confidence = confidence
        
        return best_match, best_confidence
    
    async def _extract_topics(self, text: str, language: Language) -> Set[str]:
        """Extract topics from the query text."""
        topics = set()
        
        # Common technical topics in both languages
        tech_topics = {
            'ar': {
                'الذكاء الاصطناعي': 'artificial_intelligence',
                'البرمجة': 'programming', 
                'المشروع': 'project',
                'التطوير': 'development',
                'التقنية': 'technology',
                'الكود': 'code',
                'النظام': 'system',
                'التطبيق': 'application'
            },
            'en': {
                'artificial intelligence': 'artificial_intelligence',
                'programming': 'programming',
                'project': 'project', 
                'development': 'development',
                'technology': 'technology',
                'code': 'code',
                'system': 'system',
                'application': 'application'
            }
        }
        
        topic_dict = tech_topics.get(language.value, {})
        
        for topic_phrase, normalized_topic in topic_dict.items():
            if topic_phrase in text.lower():
                topics.add(normalized_topic)
        
        return topics
    
    async def _extract_entities(self, text: str, language: Language) -> Set[str]:
        """Extract named entities from the query text."""
        entities = set()
        
        # Simple entity extraction - in production would use NER
        words = text.split()
        for word in words:
            # Look for capitalized words as potential entities
            if word and len(word) > 2 and word[0].isupper():
                entities.add(word)
        
        return entities
    
    async def _extract_behavioral_aspects(self, text: str, language: Language) -> Set[str]:
        """Extract behavioral aspects mentioned in the query."""
        aspects = set()
        
        lang_code = language.value
        
        for aspect, patterns in self.behavioral_patterns.items():
            if lang_code in patterns:
                for pattern in patterns[lang_code]:
                    if re.search(pattern, text, re.IGNORECASE):
                        aspects.add(aspect)
        
        return aspects
    
    async def _extract_keywords(self, text: str, language: Language) -> Set[str]:
        """Extract important keywords from the query."""
        # Simple keyword extraction - remove common words
        common_words = {
            'ar': {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'ذلك', 'التي', 'الذي'},
            'en': {'in', 'on', 'at', 'to', 'from', 'with', 'this', 'that', 'the', 'a', 'an'}
        }
        
        stop_words = common_words.get(language.value, set())
        
        words = text.lower().split()
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
        
        return keywords
    
    async def _extract_context_hints(self, text: str, language: Language) -> Dict[str, Any]:
        """Extract additional context hints from the query."""
        hints = {}
        
        # Check for urgency indicators
        urgency_patterns = {
            'ar': [r'عاجل', r'سريع', r'فوري'],
            'en': [r'urgent', r'quick', r'immediate']
        }
        
        lang_patterns = urgency_patterns.get(language.value, [])
        for pattern in lang_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hints['urgency'] = 'high'
                break
        
        # Check for scope indicators
        if re.search(r'(جميع|كل|all|everything)', text, re.IGNORECASE):
            hints['scope'] = 'comprehensive'
        elif re.search(r'(بعض|قليل|some|few)', text, re.IGNORECASE):
            hints['scope'] = 'limited'
        
        # Check for detail level
        if re.search(r'(مفصل|تفصيل|detailed|comprehensive)', text, re.IGNORECASE):
            hints['detail_level'] = 'high'
        elif re.search(r'(موجز|ملخص|brief|summary)', text, re.IGNORECASE):
            hints['detail_level'] = 'low'
        
        return hints