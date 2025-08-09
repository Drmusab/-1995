"""
Pattern Detector Module
Author: Drmusab
Last Modified: 2025-01-20

Detects recurring patterns and themes in user thoughts, conversations,
and behaviors over time periods.
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

from src.core.config.loader import ConfigLoader
from src.observability.logging.config import get_logger
from .types import ThoughtPattern, ThoughtTheme


class PatternDetector:
    """Detects and analyzes recurring patterns in user thoughts and behaviors."""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize the pattern detector."""
        self.logger = get_logger(__name__)
        self.config = config
        
        # Theme keywords and indicators
        self.theme_keywords = {
            ThoughtTheme.TIME_MANAGEMENT: {
                "primary": ["time", "schedule", "deadline", "organize", "plan", "priority"],
                "secondary": ["busy", "overwhelmed", "efficient", "productivity", "focus", "manage"],
                "context": ["calendar", "appointment", "meeting", "task", "urgent", "late"]
            },
            ThoughtTheme.CREATIVITY: {
                "primary": ["creative", "art", "design", "innovative", "original", "imagination"],
                "secondary": ["brainstorm", "idea", "concept", "vision", "artistic", "inventive"],
                "context": ["express", "create", "make", "build", "craft", "compose"]
            },
            ThoughtTheme.PROBLEM_SOLVING: {
                "primary": ["problem", "solution", "solve", "fix", "resolve", "challenge"],
                "secondary": ["approach", "method", "strategy", "technique", "process", "way"],
                "context": ["analyze", "troubleshoot", "debug", "investigate", "figure", "issue"]
            },
            ThoughtTheme.RELATIONSHIPS: {
                "primary": ["relationship", "friend", "family", "partner", "colleague", "social"],
                "secondary": ["communicate", "talk", "discuss", "share", "listen", "understand"],
                "context": ["conflict", "harmony", "support", "trust", "love", "care", "bond"]
            },
            ThoughtTheme.PRODUCTIVITY: {
                "primary": ["productive", "efficient", "output", "achieve", "accomplish", "complete"],
                "secondary": ["goal", "target", "milestone", "progress", "result", "success"],
                "context": ["improve", "optimize", "streamline", "automate", "system", "workflow"]
            },
            ThoughtTheme.PERSONAL_GROWTH: {
                "primary": ["growth", "develop", "improve", "learn", "skill", "knowledge"],
                "secondary": ["self", "personal", "better", "progress", "advance", "evolve"],
                "context": ["mindset", "habit", "behavior", "change", "transform", "journey"]
            },
            ThoughtTheme.EMOTIONAL_AWARENESS: {
                "primary": ["feel", "emotion", "mood", "sentiment", "anxiety", "stress"],
                "secondary": ["mindful", "aware", "conscious", "reflect", "meditation", "balance"],
                "context": ["therapy", "counseling", "mental", "wellbeing", "self-care", "calm"]
            },
            ThoughtTheme.DECISION_MAKING: {
                "primary": ["decide", "choice", "option", "alternative", "consider", "weigh"],
                "secondary": ["uncertain", "confused", "clear", "confident", "hesitate", "doubt"],
                "context": ["pros", "cons", "trade-off", "risk", "benefit", "consequence"]
            },
            ThoughtTheme.LEARNING: {
                "primary": ["learn", "study", "understand", "knowledge", "skill", "practice"],
                "secondary": ["course", "class", "book", "research", "explore", "discover"],
                "context": ["teacher", "mentor", "expert", "guide", "instruction", "education"]
            },
            ThoughtTheme.STRESS_MANAGEMENT: {
                "primary": ["stress", "pressure", "overwhelm", "anxiety", "tension", "worry"],
                "secondary": ["relax", "calm", "peace", "rest", "break", "vacation"],
                "context": ["cope", "manage", "handle", "deal", "overcome", "resilience"]
            }
        }
        
        # Pattern confidence thresholds
        self.min_frequency = 2
        self.min_confidence = 0.3
        self.context_weight = 0.3
        self.recency_weight = 0.2
    
    async def detect_patterns(
        self,
        user_id: str,
        time_window_days: int = 7
    ) -> List[ThoughtPattern]:
        """
        Detect thought patterns for a user within a time window.
        
        Args:
            user_id: User identifier
            time_window_days: Number of days to analyze
            
        Returns:
            List of detected thought patterns
        """
        # Get user interactions from the specified time window
        interactions = await self._get_user_interactions(user_id, time_window_days)
        
        if not interactions:
            return []
        
        # Analyze text for theme frequencies
        theme_analysis = self._analyze_themes_in_interactions(interactions)
        
        # Convert analysis to pattern objects
        patterns = self._create_patterns_from_analysis(theme_analysis, interactions)
        
        # Filter patterns by confidence threshold
        filtered_patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        
        # Sort by relevance (frequency * confidence)
        filtered_patterns.sort(key=lambda p: p.frequency * p.confidence, reverse=True)
        
        self.logger.debug(f"Detected {len(filtered_patterns)} patterns for user {user_id}")
        return filtered_patterns
    
    async def _get_user_interactions(self, user_id: str, days: int) -> List[Dict[str, Any]]:
        """Get user interactions from memory (mock implementation)."""
        # In production, this would query the memory manager for:
        # - Conversation messages
        # - Notes taken
        # - Tasks created/completed
        # - Questions asked
        # - Problems discussed
        
        # Mock data for demonstration
        mock_interactions = [
            {
                "text": "I need to organize my daily schedule better and manage my time more effectively.",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=1),
                "type": "conversation"
            },
            {
                "text": "Let me brainstorm some creative approaches to this design challenge.",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=2),
                "type": "note"
            },
            {
                "text": "I'm feeling overwhelmed with all these deadlines. Need to prioritize better.",
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=6),
                "type": "conversation"
            },
            {
                "text": "How can I approach this problem systematically? Let me break it down step by step.",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=3),
                "type": "task"
            },
            {
                "text": "I want to improve my communication skills and build better relationships.",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=4),
                "type": "goal"
            },
            {
                "text": "Time management has been a real challenge lately. I keep missing important deadlines.",
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=12),
                "type": "reflection"
            },
            {
                "text": "Need to find ways to manage stress better and create more balance in my life.",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=5),
                "type": "conversation"
            },
            {
                "text": "I've been thinking about learning new skills to advance my career and personal growth.",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=6),
                "type": "planning"
            }
        ]
        
        # Filter by time window
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        recent_interactions = [
            interaction for interaction in mock_interactions
            if interaction["timestamp"] >= cutoff_date
        ]
        
        return recent_interactions
    
    def _analyze_themes_in_interactions(self, interactions: List[Dict[str, Any]]) -> Dict[ThoughtTheme, Dict[str, Any]]:
        """Analyze themes present in user interactions."""
        theme_analysis = {}
        total_interactions = len(interactions)
        
        for theme in ThoughtTheme:
            analysis = {
                "frequency": 0,
                "primary_matches": 0,
                "secondary_matches": 0,
                "context_matches": 0,
                "examples": [],
                "recent_mentions": 0,
                "total_score": 0
            }
            
            keywords = self.theme_keywords.get(theme, {"primary": [], "secondary": [], "context": []})
            
            for interaction in interactions:
                text = interaction["text"].lower()
                timestamp = interaction["timestamp"]
                
                # Count keyword matches
                primary_count = sum(1 for keyword in keywords["primary"] if keyword in text)
                secondary_count = sum(1 for keyword in keywords["secondary"] if keyword in text)
                context_count = sum(1 for keyword in keywords["context"] if keyword in text)
                
                if primary_count > 0 or secondary_count > 0 or context_count > 0:
                    analysis["frequency"] += 1
                    analysis["primary_matches"] += primary_count
                    analysis["secondary_matches"] += secondary_count
                    analysis["context_matches"] += context_count
                    analysis["examples"].append(interaction["text"][:100] + "..." if len(interaction["text"]) > 100 else interaction["text"])
                    
                    # Weight recent mentions more heavily
                    hours_ago = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
                    if hours_ago < 24:  # Last 24 hours
                        analysis["recent_mentions"] += 1
            
            # Calculate total score
            analysis["total_score"] = (
                analysis["primary_matches"] * 3 +
                analysis["secondary_matches"] * 2 +
                analysis["context_matches"] * self.context_weight +
                analysis["recent_mentions"] * self.recency_weight
            )
            
            theme_analysis[theme] = analysis
        
        return theme_analysis
    
    def _create_patterns_from_analysis(
        self,
        theme_analysis: Dict[ThoughtTheme, Dict[str, Any]],
        interactions: List[Dict[str, Any]]
    ) -> List[ThoughtPattern]:
        """Create pattern objects from theme analysis."""
        patterns = []
        max_score = max(analysis["total_score"] for analysis in theme_analysis.values()) or 1
        
        for theme, analysis in theme_analysis.items():
            if analysis["frequency"] >= self.min_frequency:
                # Calculate confidence based on various factors
                confidence = self._calculate_pattern_confidence(analysis, max_score, len(interactions))
                
                # Generate insights for this pattern
                insights = self._generate_pattern_insights(theme, analysis)
                
                pattern = ThoughtPattern(
                    theme=theme,
                    frequency=analysis["frequency"],
                    confidence=confidence,
                    examples=analysis["examples"][:3],  # Keep top 3 examples
                    insights=insights,
                    first_detected=datetime.now(timezone.utc) - timedelta(days=7),  # Mock
                    last_detected=datetime.now(timezone.utc)
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_confidence(
        self,
        analysis: Dict[str, Any],
        max_score: float,
        total_interactions: int
    ) -> float:
        """Calculate confidence score for a pattern."""
        # Base confidence from normalized score
        score_confidence = analysis["total_score"] / max_score if max_score > 0 else 0
        
        # Frequency confidence (more frequent = more confident)
        frequency_confidence = min(analysis["frequency"] / total_interactions, 1.0)
        
        # Primary keyword confidence (strong indicators)
        primary_confidence = min(analysis["primary_matches"] / 10, 1.0)
        
        # Recent mention boost
        recency_boost = min(analysis["recent_mentions"] / analysis["frequency"], 0.3) if analysis["frequency"] > 0 else 0
        
        # Combine confidences
        combined_confidence = (
            score_confidence * 0.4 +
            frequency_confidence * 0.3 +
            primary_confidence * 0.2 +
            recency_boost * 0.1
        )
        
        return min(combined_confidence, 1.0)
    
    def _generate_pattern_insights(self, theme: ThoughtTheme, analysis: Dict[str, Any]) -> List[str]:
        """Generate insights for a detected pattern."""
        insights = []
        frequency = analysis["frequency"]
        primary_matches = analysis["primary_matches"]
        recent_mentions = analysis["recent_mentions"]
        
        # Theme-specific insights
        if theme == ThoughtTheme.TIME_MANAGEMENT:
            if primary_matches >= 3:
                insights.append("Time management is a significant focus area for you")
            if recent_mentions > 0:
                insights.append("You've been actively thinking about time-related challenges recently")
            if frequency >= 3:
                insights.append("Time organization appears to be an ongoing consideration")
        
        elif theme == ThoughtTheme.CREATIVITY:
            if primary_matches >= 2:
                insights.append("Creative expression and innovation are important to you")
            if recent_mentions > 0:
                insights.append("You're actively engaging with creative ideas and possibilities")
            insights.append("Your mind naturally gravitates toward creative solutions")
        
        elif theme == ThoughtTheme.PROBLEM_SOLVING:
            if primary_matches >= 3:
                insights.append("You approach challenges with a structured, solution-oriented mindset")
            if frequency >= 4:
                insights.append("Problem-solving is a core part of how you engage with the world")
            insights.append("You demonstrate resilience in facing difficulties")
        
        elif theme == ThoughtTheme.RELATIONSHIPS:
            if primary_matches >= 2:
                insights.append("Interpersonal connections are a priority in your life")
            if recent_mentions > 0:
                insights.append("You're currently focused on improving your relationships")
            insights.append("You value communication and understanding with others")
        
        elif theme == ThoughtTheme.PERSONAL_GROWTH:
            if primary_matches >= 3:
                insights.append("Self-improvement and development are central to your identity")
            if frequency >= 3:
                insights.append("You consistently seek opportunities for growth and learning")
            insights.append("You have a growth mindset and embrace challenges as learning opportunities")
        
        elif theme == ThoughtTheme.STRESS_MANAGEMENT:
            if primary_matches >= 2:
                insights.append("Managing stress and pressure is a current focus area")
            if recent_mentions > 0:
                insights.append("You're actively seeking ways to reduce stress and find balance")
            insights.append("You recognize the importance of mental health and self-care")
        
        # Generic insights if no specific ones apply
        if not insights:
            insights.append(f"{theme.value.replace('_', ' ').title()} is a recurring theme in your thoughts")
            if frequency >= 3:
                insights.append("This appears to be an area of ongoing interest or concern")
        
        return insights
    
    def detect_temporal_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect temporal patterns in when certain themes appear."""
        temporal_analysis = {
            "peak_hours": defaultdict(int),
            "daily_distribution": defaultdict(int),
            "theme_timing": defaultdict(lambda: defaultdict(int))
        }
        
        for interaction in interactions:
            timestamp = interaction["timestamp"]
            hour = timestamp.hour
            day_name = timestamp.strftime("%A")
            
            temporal_analysis["peak_hours"][hour] += 1
            temporal_analysis["daily_distribution"][day_name] += 1
            
            # Analyze themes by time
            text = interaction["text"].lower()
            for theme in ThoughtTheme:
                keywords = self.theme_keywords.get(theme, {"primary": []})
                if any(keyword in text for keyword in keywords["primary"]):
                    temporal_analysis["theme_timing"][theme.value][hour] += 1
        
        return temporal_analysis
    
    def detect_emotional_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect emotional patterns in user interactions."""
        emotional_indicators = {
            "positive": ["happy", "excited", "grateful", "confident", "optimistic", "motivated"],
            "negative": ["frustrated", "worried", "stressed", "anxious", "overwhelmed", "confused"],
            "neutral": ["thinking", "considering", "planning", "analyzing", "reviewing", "wondering"]
        }
        
        emotion_analysis = {
            "overall_sentiment": defaultdict(int),
            "theme_emotions": defaultdict(lambda: defaultdict(int)),
            "emotional_frequency": defaultdict(int)
        }
        
        for interaction in interactions:
            text = interaction["text"].lower()
            
            # Count emotional indicators
            for emotion_type, indicators in emotional_indicators.items():
                for indicator in indicators:
                    if indicator in text:
                        emotion_analysis["overall_sentiment"][emotion_type] += 1
                        emotion_analysis["emotional_frequency"][indicator] += 1
                        
                        # Associate emotions with themes
                        for theme in ThoughtTheme:
                            keywords = self.theme_keywords.get(theme, {"primary": []})
                            if any(keyword in text for keyword in keywords["primary"]):
                                emotion_analysis["theme_emotions"][theme.value][emotion_type] += 1
        
        return emotion_analysis