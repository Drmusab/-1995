"""
Reflection Analyzer for Context Time Machine
Author: Drmusab
Last Modified: 2025-01-08

Analyzes behavioral patterns and changes over time:
- Tone and mood analysis
- Energy level tracking
- Confidence progression
- Communication pattern analysis
- Behavioral trend identification
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.dependency_injection import Container
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager, Language

from .memory_replayer import ConversationThread, ConversationSegment


class BehavioralMetric(Enum):
    """Types of behavioral metrics to analyze."""
    TONE = "tone"                    # Communication tone
    MOOD = "mood"                    # Emotional state
    ENERGY = "energy"                # Energy levels
    CONFIDENCE = "confidence"        # Confidence indicators
    ENGAGEMENT = "engagement"        # Level of engagement
    FORMALITY = "formality"         # Formal vs informal communication
    COMPLEXITY = "complexity"       # Language complexity
    RESPONSIVENESS = "responsiveness" # Response time and quality


class TrendDirection(Enum):
    """Direction of behavioral trends."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    FLUCTUATING = "fluctuating"


@dataclass
class BehavioralDataPoint:
    """A single behavioral measurement point."""
    timestamp: datetime
    metric: BehavioralMetric
    value: float  # Normalized 0-1 score
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    source_segments: List[str] = field(default_factory=list)


@dataclass
class BehavioralTrend:
    """Represents a trend in behavioral metrics."""
    metric: BehavioralMetric
    direction: TrendDirection
    change_magnitude: float  # How much change (0-1)
    confidence: float
    data_points: List[BehavioralDataPoint]
    time_span: Tuple[datetime, datetime]
    insights: List[str] = field(default_factory=list)


@dataclass
class BehavioralAnalysis:
    """Complete behavioral analysis result."""
    user_id: Optional[str]
    analysis_period: Tuple[datetime, datetime]
    trends: List[BehavioralTrend]
    overall_insights: List[str]
    recommendations: List[str]
    confidence_score: float
    language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReflectionAnalyzer:
    """
    Analyzes behavioral patterns and changes in conversation data.
    
    Provides insights into how communication style, mood, energy,
    and other behavioral aspects have changed over time.
    """
    
    def __init__(self, container: Container):
        """Initialize the reflection analyzer."""
        self.container = container
        self.logger = get_logger(__name__)
        self.bilingual_manager = container.get(BilingualManager)
        
        # Initialize analysis patterns
        self._setup_analysis_patterns()
        
        # Configuration
        self.min_data_points = 3
        self.trend_sensitivity = 0.1
        self.confidence_threshold = 0.6
        
        self.logger.info("ReflectionAnalyzer initialized")
    
    def _setup_analysis_patterns(self) -> None:
        """Setup patterns for behavioral analysis."""
        
        # Tone indicators
        self.tone_patterns = {
            'positive': {
                'ar': [
                    'ممتاز', 'رائع', 'جيد', 'شكراً', 'أحب', 'سعيد', 'متحمس',
                    'أقدر', 'ممكن', 'نعم', 'أوافق', 'إيجابي'
                ],
                'en': [
                    'excellent', 'great', 'good', 'thank', 'love', 'happy', 'excited',
                    'appreciate', 'yes', 'agree', 'positive', 'wonderful'
                ]
            },
            'negative': {
                'ar': [
                    'سيء', 'لا أحب', 'مشكلة', 'صعب', 'مستحيل', 'لا', 'أرفض',
                    'غضبان', 'محبط', 'قلق', 'خائف'
                ],
                'en': [
                    'bad', 'hate', 'problem', 'difficult', 'impossible', 'no', 'refuse',
                    'angry', 'frustrated', 'worried', 'afraid'
                ]
            },
            'neutral': {
                'ar': [
                    'ربما', 'لا أدري', 'عادي', 'لا بأس', 'محايد', 'نوعاً ما'
                ],
                'en': [
                    'maybe', 'perhaps', 'okay', 'neutral', 'somewhat', 'fine'
                ]
            }
        }
        
        # Confidence indicators
        self.confidence_patterns = {
            'high': {
                'ar': [
                    'أعتقد', 'متأكد', 'بالتأكيد', 'لا شك', 'واثق', 'أعرف',
                    'سأفعل', 'يمكنني', 'قادر'
                ],
                'en': [
                    'i think', 'sure', 'definitely', 'certainly', 'confident', 'know',
                    'will do', 'can do', 'able'
                ]
            },
            'low': {
                'ar': [
                    'لست متأكد', 'ربما', 'لا أعرف', 'صعب', 'لا أستطيع',
                    'قد', 'محتمل', 'أحاول'
                ],
                'en': [
                    'not sure', 'maybe', 'don\'t know', 'difficult', 'can\'t',
                    'might', 'probably', 'try'
                ]
            }
        }
        
        # Energy indicators
        self.energy_patterns = {
            'high': {
                'ar': [
                    'متحمس', 'نشيط', 'سريع', 'فوراً', 'بسرعة', 'حماس',
                    'طاقة', 'نشاط'
                ],
                'en': [
                    'excited', 'energetic', 'quick', 'immediately', 'fast', 'enthusiasm',
                    'energy', 'active'
                ]
            },
            'low': {
                'ar': [
                    'متعب', 'مرهق', 'بطيء', 'لاحقاً', 'تأجيل', 'كسول'
                ],
                'en': [
                    'tired', 'exhausted', 'slow', 'later', 'postpone', 'lazy'
                ]
            }
        }
        
        # Engagement indicators
        self.engagement_patterns = {
            'high': {
                'ar': [
                    'أسئلة', 'كيف', 'ماذا', 'لماذا', 'متى', 'أين', 'مثير للاهتمام',
                    'أريد أن أعرف', 'أخبرني', 'شرح'
                ],
                'en': [
                    'questions', 'how', 'what', 'why', 'when', 'where', 'interesting',
                    'want to know', 'tell me', 'explain'
                ]
            },
            'low': {
                'ar': [
                    'نعم', 'لا', 'حسناً', 'أوك', 'مفهوم'
                ],
                'en': [
                    'yes', 'no', 'okay', 'ok', 'understood'
                ]
            }
        }
    
    async def analyze_behavioral_patterns(
        self,
        conversation_threads: List[ConversationThread],
        language: Language = Language.ARABIC,
        user_id: Optional[str] = None
    ) -> BehavioralAnalysis:
        """
        Analyze behavioral patterns from conversation threads.
        
        Args:
            conversation_threads: List of conversation threads to analyze
            language: Language for insights and recommendations
            user_id: User ID for analysis
            
        Returns:
            BehavioralAnalysis with trends and insights
        """
        try:
            if not conversation_threads:
                return self._create_empty_analysis(language, user_id)
            
            # Calculate time span
            start_time = min(thread.start_time for thread in conversation_threads)
            end_time = max(thread.end_time for thread in conversation_threads)
            
            # Extract behavioral data points
            data_points = await self._extract_behavioral_data_points(conversation_threads)
            
            # Analyze trends for each metric
            trends = []
            for metric in BehavioralMetric:
                metric_points = [dp for dp in data_points if dp.metric == metric]
                if len(metric_points) >= self.min_data_points:
                    trend = await self._analyze_metric_trend(metric, metric_points)
                    if trend.confidence >= self.confidence_threshold:
                        trends.append(trend)
            
            # Generate overall insights
            overall_insights = await self._generate_overall_insights(trends, language)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(trends, language)
            
            # Calculate overall confidence
            confidence_score = (
                statistics.mean([trend.confidence for trend in trends])
                if trends else 0.0
            )
            
            analysis = BehavioralAnalysis(
                user_id=user_id,
                analysis_period=(start_time, end_time),
                trends=trends,
                overall_insights=overall_insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
                language=language,
                metadata={
                    "total_threads": len(conversation_threads),
                    "total_segments": sum(len(thread.segments) for thread in conversation_threads),
                    "data_points_analyzed": len(data_points),
                    "metrics_tracked": len(trends)
                }
            )
            
            self.logger.debug(f"Behavioral analysis completed: {len(trends)} trends identified")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Behavioral analysis failed: {str(e)}")
            return self._create_empty_analysis(language, user_id, error=str(e))
    
    async def _extract_behavioral_data_points(
        self,
        conversation_threads: List[ConversationThread]
    ) -> List[BehavioralDataPoint]:
        """Extract behavioral data points from conversation threads."""
        data_points = []
        
        for thread in conversation_threads:
            for segment in thread.segments:
                # Analyze each segment for behavioral indicators
                segment_points = await self._analyze_segment_behavior(segment)
                data_points.extend(segment_points)
        
        return data_points
    
    async def _analyze_segment_behavior(
        self,
        segment: ConversationSegment
    ) -> List[BehavioralDataPoint]:
        """Analyze a conversation segment for behavioral indicators."""
        data_points = []
        content = segment.content.lower()
        
        # Detect language
        language = self.bilingual_manager.detect_language(segment.content)
        lang_code = language.value
        
        # Analyze tone
        tone_score = await self._analyze_tone(content, lang_code)
        if tone_score is not None:
            data_points.append(BehavioralDataPoint(
                timestamp=segment.timestamp,
                metric=BehavioralMetric.TONE,
                value=tone_score,
                confidence=0.7,
                context={"language": lang_code},
                source_segments=[segment.id]
            ))
        
        # Analyze confidence
        confidence_score = await self._analyze_confidence(content, lang_code)
        if confidence_score is not None:
            data_points.append(BehavioralDataPoint(
                timestamp=segment.timestamp,
                metric=BehavioralMetric.CONFIDENCE,
                value=confidence_score,
                confidence=0.6,
                context={"language": lang_code},
                source_segments=[segment.id]
            ))
        
        # Analyze energy
        energy_score = await self._analyze_energy(content, lang_code)
        if energy_score is not None:
            data_points.append(BehavioralDataPoint(
                timestamp=segment.timestamp,
                metric=BehavioralMetric.ENERGY,
                value=energy_score,
                confidence=0.6,
                context={"language": lang_code},
                source_segments=[segment.id]
            ))
        
        # Analyze engagement
        engagement_score = await self._analyze_engagement(content, lang_code)
        if engagement_score is not None:
            data_points.append(BehavioralDataPoint(
                timestamp=segment.timestamp,
                metric=BehavioralMetric.ENGAGEMENT,
                value=engagement_score,
                confidence=0.8,
                context={"language": lang_code},
                source_segments=[segment.id]
            ))
        
        # Analyze complexity
        complexity_score = await self._analyze_complexity(content, lang_code)
        if complexity_score is not None:
            data_points.append(BehavioralDataPoint(
                timestamp=segment.timestamp,
                metric=BehavioralMetric.COMPLEXITY,
                value=complexity_score,
                confidence=0.9,
                context={"language": lang_code, "word_count": len(content.split())},
                source_segments=[segment.id]
            ))
        
        return data_points
    
    async def _analyze_tone(self, content: str, language: str) -> Optional[float]:
        """Analyze tone from content."""
        if language not in self.tone_patterns['positive']:
            return None
        
        positive_count = sum(
            1 for pattern in self.tone_patterns['positive'][language]
            if pattern in content
        )
        negative_count = sum(
            1 for pattern in self.tone_patterns['negative'][language]
            if pattern in content
        )
        neutral_count = sum(
            1 for pattern in self.tone_patterns['neutral'][language]
            if pattern in content
        )
        
        total_indicators = positive_count + negative_count + neutral_count
        if total_indicators == 0:
            return 0.5  # Neutral if no indicators
        
        # Calculate weighted score
        positive_weight = positive_count / total_indicators
        negative_weight = negative_count / total_indicators
        
        # Tone score: 0 = negative, 0.5 = neutral, 1 = positive
        tone_score = positive_weight + 0.5 * (neutral_count / total_indicators)
        
        return tone_score
    
    async def _analyze_confidence(self, content: str, language: str) -> Optional[float]:
        """Analyze confidence from content."""
        if language not in self.confidence_patterns['high']:
            return None
        
        high_confidence_count = sum(
            1 for pattern in self.confidence_patterns['high'][language]
            if pattern in content
        )
        low_confidence_count = sum(
            1 for pattern in self.confidence_patterns['low'][language]
            if pattern in content
        )
        
        total_indicators = high_confidence_count + low_confidence_count
        if total_indicators == 0:
            return 0.5  # Neutral confidence
        
        confidence_score = high_confidence_count / total_indicators
        return confidence_score
    
    async def _analyze_energy(self, content: str, language: str) -> Optional[float]:
        """Analyze energy level from content."""
        if language not in self.energy_patterns['high']:
            return None
        
        high_energy_count = sum(
            1 for pattern in self.energy_patterns['high'][language]
            if pattern in content
        )
        low_energy_count = sum(
            1 for pattern in self.energy_patterns['low'][language]
            if pattern in content
        )
        
        total_indicators = high_energy_count + low_energy_count
        if total_indicators == 0:
            return 0.5  # Neutral energy
        
        energy_score = high_energy_count / total_indicators
        return energy_score
    
    async def _analyze_engagement(self, content: str, language: str) -> Optional[float]:
        """Analyze engagement level from content."""
        if language not in self.engagement_patterns['high']:
            return None
        
        high_engagement_count = sum(
            1 for pattern in self.engagement_patterns['high'][language]
            if pattern in content
        )
        low_engagement_count = sum(
            1 for pattern in self.engagement_patterns['low'][language]
            if pattern in content
        )
        
        # Count questions (engagement indicator)
        question_count = content.count('?') + content.count('؟')
        
        total_engagement = high_engagement_count + question_count
        total_indicators = total_engagement + low_engagement_count
        
        if total_indicators == 0:
            return 0.5  # Neutral engagement
        
        engagement_score = total_engagement / total_indicators
        return engagement_score
    
    async def _analyze_complexity(self, content: str, language: str) -> Optional[float]:
        """Analyze language complexity from content."""
        words = content.split()
        if len(words) < 3:
            return 0.2  # Very simple
        
        # Simple complexity measures
        avg_word_length = statistics.mean(len(word) for word in words)
        sentence_count = content.count('.') + content.count('!') + content.count('?') + content.count('؟')
        
        if sentence_count == 0:
            sentence_count = 1
        
        avg_sentence_length = len(words) / sentence_count
        
        # Normalize complexity score (0-1)
        word_complexity = min(1.0, avg_word_length / 10)  # Max complexity at 10 chars/word
        sentence_complexity = min(1.0, avg_sentence_length / 20)  # Max complexity at 20 words/sentence
        
        complexity_score = (word_complexity + sentence_complexity) / 2
        return complexity_score
    
    async def _analyze_metric_trend(
        self,
        metric: BehavioralMetric,
        data_points: List[BehavioralDataPoint]
    ) -> BehavioralTrend:
        """Analyze trend for a specific metric."""
        if len(data_points) < 2:
            return BehavioralTrend(
                metric=metric,
                direction=TrendDirection.STABLE,
                change_magnitude=0.0,
                confidence=0.0,
                data_points=data_points,
                time_span=(data_points[0].timestamp, data_points[0].timestamp),
                insights=[]
            )
        
        # Sort by timestamp
        sorted_points = sorted(data_points, key=lambda x: x.timestamp)
        
        # Calculate trend using linear regression (simplified)
        x_values = [(dp.timestamp - sorted_points[0].timestamp).total_seconds() 
                   for dp in sorted_points]
        y_values = [dp.value for dp in sorted_points]
        
        # Simple linear trend calculation
        n = len(sorted_points)
        if n < 2:
            slope = 0
        else:
            mean_x = statistics.mean(x_values)
            mean_y = statistics.mean(y_values)
            
            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
            denominator = sum((x - mean_x) ** 2 for x in x_values)
            
            slope = numerator / denominator if denominator != 0 else 0
        
        # Determine trend direction
        if abs(slope) < self.trend_sensitivity:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING
        
        # Calculate change magnitude
        if len(y_values) >= 2:
            change_magnitude = abs(y_values[-1] - y_values[0])
        else:
            change_magnitude = 0.0
        
        # Calculate confidence based on data consistency
        if len(y_values) >= 3:
            variance = statistics.variance(y_values)
            confidence = max(0.0, 1.0 - variance)
        else:
            confidence = 0.5
        
        # Generate insights
        insights = await self._generate_metric_insights(metric, direction, change_magnitude)
        
        return BehavioralTrend(
            metric=metric,
            direction=direction,
            change_magnitude=change_magnitude,
            confidence=confidence,
            data_points=sorted_points,
            time_span=(sorted_points[0].timestamp, sorted_points[-1].timestamp),
            insights=insights
        )
    
    async def _generate_metric_insights(
        self,
        metric: BehavioralMetric,
        direction: TrendDirection,
        magnitude: float
    ) -> List[str]:
        """Generate insights for a specific metric trend."""
        insights = []
        
        if metric == BehavioralMetric.TONE:
            if direction == TrendDirection.IMPROVING:
                insights.append("النبرة تتحسن بمرور الوقت")
            elif direction == TrendDirection.DECLINING:
                insights.append("النبرة تصبح أكثر سلبية")
            else:
                insights.append("النبرة مستقرة")
        
        elif metric == BehavioralMetric.CONFIDENCE:
            if direction == TrendDirection.IMPROVING:
                insights.append("الثقة في تزايد")
            elif direction == TrendDirection.DECLINING:
                insights.append("الثقة في تراجع")
            else:
                insights.append("مستوى الثقة مستقر")
        
        elif metric == BehavioralMetric.ENERGY:
            if direction == TrendDirection.IMPROVING:
                insights.append("مستوى الطاقة يرتفع")
            elif direction == TrendDirection.DECLINING:
                insights.append("مستوى الطاقة ينخفض")
            else:
                insights.append("مستوى الطاقة مستقر")
        
        elif metric == BehavioralMetric.ENGAGEMENT:
            if direction == TrendDirection.IMPROVING:
                insights.append("التفاعل يتحسن")
            elif direction == TrendDirection.DECLINING:
                insights.append("التفاعل يقل")
            else:
                insights.append("مستوى التفاعل مستقر")
        
        elif metric == BehavioralMetric.COMPLEXITY:
            if direction == TrendDirection.IMPROVING:
                insights.append("اللغة تصبح أكثر تعقيداً")
            elif direction == TrendDirection.DECLINING:
                insights.append("اللغة تصبح أبسط")
            else:
                insights.append("تعقيد اللغة مستقر")
        
        # Add magnitude-based insights
        if magnitude > 0.3:
            insights.append("تغيير ملحوظ")
        elif magnitude > 0.1:
            insights.append("تغيير تدريجي")
        
        return insights
    
    async def _generate_overall_insights(
        self,
        trends: List[BehavioralTrend],
        language: Language
    ) -> List[str]:
        """Generate overall insights from all trends."""
        insights = []
        
        if not trends:
            if language == Language.ARABIC:
                insights.append("لا توجد بيانات كافية للتحليل السلوكي")
            else:
                insights.append("Insufficient data for behavioral analysis")
            return insights
        
        # Count improving vs declining trends
        improving_count = sum(1 for trend in trends if trend.direction == TrendDirection.IMPROVING)
        declining_count = sum(1 for trend in trends if trend.direction == TrendDirection.DECLINING)
        stable_count = sum(1 for trend in trends if trend.direction == TrendDirection.STABLE)
        
        if language == Language.ARABIC:
            if improving_count > declining_count:
                insights.append("التحسن العام في السلوك الاتصالي")
            elif declining_count > improving_count:
                insights.append("بعض التراجع في السلوك الاتصالي")
            else:
                insights.append("السلوك الاتصالي مستقر بشكل عام")
            
            insights.append(f"تم تحليل {len(trends)} مؤشر سلوكي")
            
            if stable_count > len(trends) / 2:
                insights.append("السلوك الاتصالي متسق")
        else:
            if improving_count > declining_count:
                insights.append("Overall improvement in communication behavior")
            elif declining_count > improving_count:
                insights.append("Some decline in communication behavior")
            else:
                insights.append("Communication behavior is generally stable")
            
            insights.append(f"Analyzed {len(trends)} behavioral indicators")
            
            if stable_count > len(trends) / 2:
                insights.append("Communication behavior is consistent")
        
        return insights
    
    async def _generate_recommendations(
        self,
        trends: List[BehavioralTrend],
        language: Language
    ) -> List[str]:
        """Generate recommendations based on trends."""
        recommendations = []
        
        if not trends:
            return recommendations
        
        for trend in trends:
            if trend.direction == TrendDirection.DECLINING and trend.confidence > 0.7:
                if trend.metric == BehavioralMetric.CONFIDENCE:
                    if language == Language.ARABIC:
                        recommendations.append("حاول استخدام عبارات أكثر حزماً")
                    else:
                        recommendations.append("Try using more assertive phrases")
                
                elif trend.metric == BehavioralMetric.ENERGY:
                    if language == Language.ARABIC:
                        recommendations.append("خذ استراحات منتظمة لتجديد الطاقة")
                    else:
                        recommendations.append("Take regular breaks to refresh energy")
                
                elif trend.metric == BehavioralMetric.ENGAGEMENT:
                    if language == Language.ARABIC:
                        recommendations.append("اطرح المزيد من الأسئلة للتفاعل")
                    else:
                        recommendations.append("Ask more questions to increase engagement")
        
        # General recommendations
        if len([t for t in trends if t.direction == TrendDirection.IMPROVING]) > 2:
            if language == Language.ARABIC:
                recommendations.append("استمر في النهج الحالي - النتائج إيجابية")
            else:
                recommendations.append("Continue current approach - results are positive")
        
        return recommendations
    
    def _create_empty_analysis(
        self,
        language: Language,
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> BehavioralAnalysis:
        """Create an empty analysis result."""
        now = datetime.now(timezone.utc)
        
        insights = []
        if language == Language.ARABIC:
            insights.append("لا توجد بيانات كافية للتحليل السلوكي")
        else:
            insights.append("Insufficient data for behavioral analysis")
        
        metadata = {}
        if error:
            metadata["error"] = error
        
        return BehavioralAnalysis(
            user_id=user_id,
            analysis_period=(now, now),
            trends=[],
            overall_insights=insights,
            recommendations=[],
            confidence_score=0.0,
            language=language,
            metadata=metadata
        )