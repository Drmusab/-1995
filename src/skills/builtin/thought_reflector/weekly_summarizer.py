"""
Weekly Summarizer Module
Author: Drmusab
Last Modified: 2025-01-20

Creates comprehensive weekly summaries of user's thought patterns,
interactions, and behavioral trends.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter

from src.core.config.loader import ConfigLoader
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger
from .types import ThoughtTheme


class WeeklySummarizer:
    """Creates weekly summaries of user thought patterns and activities."""
    
    def __init__(self, config: Optional[ConfigLoader] = None, memory_manager: Optional[MemoryManager] = None):
        """Initialize the weekly summarizer."""
        self.logger = get_logger(__name__)
        self.config = config
        self.memory_manager = memory_manager
        
        # Summary categories
        self.activity_categories = [
            "conversations",
            "notes",
            "tasks",
            "questions",
            "reflections",
            "decisions",
            "learning",
            "planning"
        ]
    
    async def generate_summary(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive weekly summary for a user.
        
        Args:
            user_id: User identifier
            start_date: Start of the summary period
            end_date: End of the summary period
            
        Returns:
            Comprehensive summary dictionary
        """
        # Get all user interactions in the time period
        interactions = await self._get_interactions_in_period(user_id, start_date, end_date)
        
        if not interactions:
            return self._create_empty_summary(start_date, end_date)
        
        # Analyze different aspects
        activity_summary = self._analyze_activity_patterns(interactions)
        theme_summary = self._analyze_theme_distribution(interactions)
        temporal_summary = self._analyze_temporal_patterns(interactions)
        progress_summary = self._analyze_progress_indicators(interactions)
        mood_summary = self._analyze_mood_trends(interactions)
        
        # Create comprehensive summary
        summary = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "overview": {
                "interaction_count": len(interactions),
                "daily_average": len(interactions) / max((end_date - start_date).days, 1),
                "most_active_day": self._find_most_active_day(interactions),
                "primary_focus_areas": self._identify_primary_focus_areas(theme_summary)
            },
            "activities": activity_summary,
            "themes": theme_summary,
            "temporal_patterns": temporal_summary,
            "progress_indicators": progress_summary,
            "mood_trends": mood_summary,
            "insights": self._generate_weekly_insights(interactions, theme_summary, activity_summary),
            "recommendations": self._generate_recommendations(theme_summary, activity_summary, mood_summary)
        }
        
        return summary
    
    async def _get_interactions_in_period(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get user interactions within the specified time period."""
        
        # In production, this would query the memory manager for:
        # - All conversation messages
        # - Notes created or updated
        # - Tasks created, updated, or completed
        # - Questions asked
        # - Decisions made
        # - Learning activities
        # - Planning sessions
        
        # Mock data for demonstration
        mock_interactions = []
        current_date = start_date
        
        while current_date < end_date:
            # Generate mock interactions for each day
            day_interactions = self._generate_mock_day_interactions(current_date, user_id)
            mock_interactions.extend(day_interactions)
            current_date += timedelta(days=1)
        
        return mock_interactions
    
    def _generate_mock_day_interactions(self, date: datetime, user_id: str) -> List[Dict[str, Any]]:
        """Generate mock interactions for a specific day."""
        import random
        
        # Vary interaction count by day
        base_interactions = random.randint(2, 8)
        
        # Weekend vs weekday patterns
        if date.weekday() >= 5:  # Weekend
            base_interactions = max(1, base_interactions - 2)
        
        interactions = []
        
        mock_texts = [
            "I need to better organize my schedule and manage my time more effectively.",
            "Let me brainstorm some creative solutions to this design challenge I'm facing.",
            "Feeling a bit overwhelmed with deadlines. Need to prioritize tasks better.",
            "How should I approach this complex problem? Let me think systematically.",
            "Want to improve my communication skills and build stronger relationships.",
            "Been thinking about learning new skills for personal and professional growth.",
            "Need to find better ways to manage stress and maintain work-life balance.",
            "Reflecting on my goals and whether I'm making meaningful progress.",
            "Had an interesting conversation about innovation and creative thinking today.",
            "Struggling with decision-making on this important choice. Need more clarity.",
            "Planning my week ahead and setting priorities for the most important tasks.",
            "Grateful for the progress I've made recently, even if it feels small.",
            "Wondering how I can be more productive without burning myself out.",
            "Had a breakthrough moment in understanding this concept I've been studying.",
            "Feeling motivated to tackle some challenging projects I've been avoiding."
        ]
        
        interaction_types = ["conversation", "note", "task", "question", "reflection", "planning"]
        
        for i in range(base_interactions):
            # Add some time variation throughout the day
            hour_offset = random.randint(6, 22)  # Between 6 AM and 10 PM
            interaction_time = date.replace(hour=hour_offset, minute=random.randint(0, 59))
            
            interactions.append({
                "text": random.choice(mock_texts),
                "timestamp": interaction_time,
                "type": random.choice(interaction_types),
                "user_id": user_id
            })
        
        return interactions
    
    def _create_empty_summary(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Create an empty summary when no interactions are found."""
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "duration_days": (end_date - start_date).days
            },
            "overview": {
                "interaction_count": 0,
                "daily_average": 0,
                "message": "No significant interactions detected in this period"
            },
            "insights": ["Continue engaging in self-reflection to develop meaningful patterns"],
            "recommendations": ["Try to maintain regular interaction with the assistant for better insights"]
        }
    
    def _analyze_activity_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in user activities."""
        activity_counts = Counter(interaction["type"] for interaction in interactions)
        
        # Calculate activity distribution
        total_activities = sum(activity_counts.values())
        activity_percentages = {
            activity: (count / total_activities) * 100
            for activity, count in activity_counts.items()
        }
        
        # Find peak activity times
        hourly_activity = defaultdict(int)
        for interaction in interactions:
            hour = interaction["timestamp"].hour
            hourly_activity[hour] += 1
        
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else 12
        
        return {
            "total_activities": total_activities,
            "activity_breakdown": dict(activity_counts),
            "activity_percentages": activity_percentages,
            "most_common_activity": activity_counts.most_common(1)[0] if activity_counts else ("none", 0),
            "peak_activity_hour": peak_hour,
            "daily_distribution": self._calculate_daily_distribution(interactions)
        }
    
    def _analyze_theme_distribution(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of themes across interactions."""
        theme_keywords = {
            "time_management": ["time", "schedule", "deadline", "organize", "plan", "priority", "busy", "overwhelmed"],
            "creativity": ["creative", "brainstorm", "innovative", "design", "art", "idea", "imagination"],
            "problem_solving": ["problem", "solution", "solve", "challenge", "approach", "method", "systematic"],
            "relationships": ["communication", "relationships", "team", "social", "connect", "understand"],
            "personal_growth": ["learn", "growth", "develop", "improve", "skill", "progress", "goals"],
            "stress_management": ["stress", "overwhelmed", "balance", "calm", "manage", "pressure", "relax"],
            "productivity": ["productive", "efficient", "accomplish", "complete", "optimize", "workflow"],
            "decision_making": ["decide", "choice", "consider", "evaluate", "uncertain", "clarity"]
        }
        
        theme_counts = defaultdict(int)
        theme_examples = defaultdict(list)
        
        for interaction in interactions:
            text = interaction["text"].lower()
            for theme, keywords in theme_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > 0:
                    theme_counts[theme] += matches
                    if len(theme_examples[theme]) < 2:  # Keep up to 2 examples
                        theme_examples[theme].append(interaction["text"][:80] + "..." if len(interaction["text"]) > 80 else interaction["text"])
        
        # Calculate percentages
        total_theme_mentions = sum(theme_counts.values())
        theme_percentages = {
            theme: (count / total_theme_mentions) * 100
            for theme, count in theme_counts.items()
        } if total_theme_mentions > 0 else {}
        
        return {
            "theme_counts": dict(theme_counts),
            "theme_percentages": theme_percentages,
            "dominant_themes": sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "theme_examples": dict(theme_examples),
            "total_theme_mentions": total_theme_mentions
        }
    
    def _analyze_temporal_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in user interactions."""
        hourly_distribution = defaultdict(int)
        daily_distribution = defaultdict(int)
        weekly_trend = defaultdict(int)
        
        for interaction in interactions:
            timestamp = interaction["timestamp"]
            hourly_distribution[timestamp.hour] += 1
            daily_distribution[timestamp.strftime("%A")] += 1
            
            # Calculate week number for trend analysis
            week_start = timestamp - timedelta(days=timestamp.weekday())
            week_key = week_start.strftime("%Y-W%U")
            weekly_trend[week_key] += 1
        
        # Find patterns
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else 12
        most_active_day = max(daily_distribution.items(), key=lambda x: x[1])[0] if daily_distribution else "Monday"
        
        # Determine if user is more active in morning, afternoon, or evening
        morning_activity = sum(hourly_distribution[h] for h in range(6, 12))
        afternoon_activity = sum(hourly_distribution[h] for h in range(12, 18))
        evening_activity = sum(hourly_distribution[h] for h in range(18, 24))
        
        peak_period = "morning"
        if afternoon_activity > morning_activity and afternoon_activity > evening_activity:
            peak_period = "afternoon"
        elif evening_activity > morning_activity and evening_activity > afternoon_activity:
            peak_period = "evening"
        
        return {
            "hourly_distribution": dict(hourly_distribution),
            "daily_distribution": dict(daily_distribution),
            "peak_hour": peak_hour,
            "most_active_day": most_active_day,
            "peak_period": peak_period,
            "activity_periods": {
                "morning": morning_activity,
                "afternoon": afternoon_activity,
                "evening": evening_activity
            },
            "weekly_trend": dict(weekly_trend)
        }
    
    def _analyze_progress_indicators(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze indicators of progress and growth."""
        progress_keywords = {
            "achievements": ["completed", "finished", "accomplished", "achieved", "succeeded", "breakthrough"],
            "challenges": ["struggling", "difficult", "stuck", "confused", "frustrated", "blocked"],
            "learning": ["learned", "discovered", "understood", "realized", "insight", "knowledge"],
            "planning": ["plan", "goal", "target", "strategy", "prepare", "organize"],
            "reflection": ["reflect", "think", "consider", "analyze", "evaluate", "review"]
        }
        
        indicator_counts = defaultdict(int)
        examples = defaultdict(list)
        
        for interaction in interactions:
            text = interaction["text"].lower()
            for category, keywords in progress_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > 0:
                    indicator_counts[category] += matches
                    if len(examples[category]) < 1:  # Keep one example
                        examples[category].append(interaction["text"])
        
        # Calculate progress score
        positive_indicators = indicator_counts["achievements"] + indicator_counts["learning"] + indicator_counts["planning"]
        negative_indicators = indicator_counts["challenges"]
        total_indicators = sum(indicator_counts.values())
        
        progress_score = 0.5  # Neutral baseline
        if total_indicators > 0:
            progress_score = (positive_indicators + 0.5 * indicator_counts["reflection"]) / total_indicators
        
        return {
            "indicator_counts": dict(indicator_counts),
            "progress_score": progress_score,
            "progress_trend": "positive" if progress_score > 0.6 else "neutral" if progress_score > 0.4 else "challenging",
            "examples": dict(examples),
            "summary": self._generate_progress_summary(progress_score, indicator_counts)
        }
    
    def _analyze_mood_trends(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze mood and emotional trends."""
        mood_indicators = {
            "positive": ["happy", "excited", "grateful", "confident", "optimistic", "motivated", "satisfied", "proud"],
            "negative": ["frustrated", "worried", "stressed", "anxious", "overwhelmed", "sad", "disappointed", "tired"],
            "neutral": ["thinking", "considering", "planning", "analyzing", "wondering", "reviewing", "working"]
        }
        
        mood_counts = defaultdict(int)
        daily_mood = defaultdict(lambda: defaultdict(int))
        
        for interaction in interactions:
            text = interaction["text"].lower()
            day = interaction["timestamp"].strftime("%A")
            
            for mood, indicators in mood_indicators.items():
                matches = sum(1 for indicator in indicators if indicator in text)
                if matches > 0:
                    mood_counts[mood] += matches
                    daily_mood[day][mood] += matches
        
        # Calculate overall mood score
        total_mood_indicators = sum(mood_counts.values())
        mood_score = 0.5  # Neutral baseline
        if total_mood_indicators > 0:
            mood_score = (mood_counts["positive"] + 0.5 * mood_counts["neutral"]) / total_mood_indicators
        
        return {
            "mood_counts": dict(mood_counts),
            "mood_score": mood_score,
            "mood_trend": "positive" if mood_score > 0.6 else "neutral" if mood_score > 0.4 else "concerning",
            "daily_mood_patterns": dict(daily_mood),
            "dominant_mood": max(mood_counts.items(), key=lambda x: x[1])[0] if mood_counts else "neutral"
        }
    
    def _find_most_active_day(self, interactions: List[Dict[str, Any]]) -> str:
        """Find the day with the most interactions."""
        daily_counts = defaultdict(int)
        for interaction in interactions:
            day = interaction["timestamp"].strftime("%A")
            daily_counts[day] += 1
        
        return max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else "Monday"
    
    def _identify_primary_focus_areas(self, theme_summary: Dict[str, Any]) -> List[str]:
        """Identify the primary focus areas based on theme analysis."""
        dominant_themes = theme_summary.get("dominant_themes", [])
        return [theme[0].replace("_", " ").title() for theme in dominant_themes[:3]]
    
    def _calculate_daily_distribution(self, interactions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of interactions across days of the week."""
        daily_counts = defaultdict(int)
        for interaction in interactions:
            day = interaction["timestamp"].strftime("%A")
            daily_counts[day] += 1
        return dict(daily_counts)
    
    def _generate_progress_summary(self, progress_score: float, indicator_counts: Dict[str, int]) -> str:
        """Generate a summary of progress indicators."""
        if progress_score > 0.7:
            return "Strong progress indicators with consistent achievements and learning"
        elif progress_score > 0.5:
            return "Positive progress with balanced mix of achievements and planning"
        elif progress_score > 0.3:
            return "Mixed progress with some challenges but continued engagement"
        else:
            return "Facing significant challenges but showing persistence and reflection"
    
    def _generate_weekly_insights(
        self,
        interactions: List[Dict[str, Any]],
        theme_summary: Dict[str, Any],
        activity_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate insights based on the weekly analysis."""
        insights = []
        
        # Interaction volume insights
        interaction_count = len(interactions)
        daily_average = activity_summary.get("daily_distribution", {})
        
        if interaction_count > 20:
            insights.append("You've been highly engaged in self-reflection and planning this week")
        elif interaction_count > 10:
            insights.append("You've maintained consistent engagement with personal development activities")
        else:
            insights.append("You've had some meaningful moments of reflection this week")
        
        # Theme insights
        dominant_themes = theme_summary.get("dominant_themes", [])
        if dominant_themes:
            top_theme = dominant_themes[0][0].replace("_", " ").title()
            insights.append(f"{top_theme} has been your primary area of focus")
        
        # Activity pattern insights
        most_common_activity = activity_summary.get("most_common_activity", ("none", 0))
        if most_common_activity[0] == "reflection":
            insights.append("You've been particularly reflective and introspective")
        elif most_common_activity[0] == "planning":
            insights.append("You've been focused on organization and future planning")
        elif most_common_activity[0] == "conversation":
            insights.append("You've been actively engaging in dialogue and exploration")
        
        return insights
    
    def _generate_recommendations(
        self,
        theme_summary: Dict[str, Any],
        activity_summary: Dict[str, Any],
        mood_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on the weekly analysis."""
        recommendations = []
        
        # Theme-based recommendations
        dominant_themes = theme_summary.get("dominant_themes", [])
        for theme, count in dominant_themes[:2]:
            if theme == "stress_management":
                recommendations.append("Consider incorporating stress-reduction techniques into your daily routine")
            elif theme == "time_management":
                recommendations.append("Explore time-blocking or prioritization frameworks to enhance your time management")
            elif theme == "creativity":
                recommendations.append("Set aside dedicated time for creative exploration and experimentation")
            elif theme == "personal_growth":
                recommendations.append("Consider setting specific, measurable goals for your development areas")
        
        # Mood-based recommendations
        mood_trend = mood_summary.get("mood_trend", "neutral")
        if mood_trend == "concerning":
            recommendations.append("Consider reaching out for support or practicing self-care activities")
        elif mood_trend == "positive":
            recommendations.append("Continue the practices that are supporting your positive mindset")
        
        # Activity-based recommendations
        interaction_count = activity_summary.get("total_activities", 0)
        if interaction_count < 10:
            recommendations.append("Try to maintain more regular reflection and planning sessions")
        
        # Default recommendation if none specific
        if not recommendations:
            recommendations.append("Continue your thoughtful approach to self-reflection and personal development")
        
        return recommendations