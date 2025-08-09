"""
Thought Analyzer Module
Author: Drmusab
Last Modified: 2025-01-20

Analyzes user thoughts and conversations to identify cognitive patterns,
problem-solving approaches, and mental frameworks.
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict

from src.core.config.loader import ConfigLoader
from src.observability.logging.config import get_logger
from .types import ThoughtTheme, ProblemSolvingStyle


class ThoughtAnalyzer:
    """Analyzes thought patterns and cognitive styles from user interactions."""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize the thought analyzer."""
        self.logger = get_logger(__name__)
        self.config = config
        
        # Problem-solving style indicators
        self.style_indicators = {
            "analytical": [
                r"\b(analyze|breakdown|step by step|systematic|logical|data|evidence|research)\b",
                r"\b(pros and cons|compare|evaluate|assess|measure|calculate)\b",
                r"\b(first|second|third|finally|therefore|because|thus|hence)\b"
            ],
            "creative": [
                r"\b(brainstorm|creative|innovative|imagine|what if|alternative|outside the box)\b",
                r"\b(inspire|vision|dream|artistic|unique|original|inventive)\b",
                r"\b(could|might|perhaps|maybe|potentially|possibilities)\b"
            ],
            "intuitive": [
                r"\b(feel|sense|instinct|gut|intuition|hunch|seems like)\b",
                r"\b(naturally|spontaneous|immediate|obvious|clear|apparent)\b",
                r"\b(trust|believe|confident|certain|sure|know)\b"
            ],
            "collaborative": [
                r"\b(team|together|discuss|share|feedback|input|advice|help)\b",
                r"\b(we|us|our|collaborate|cooperate|group|meeting|consensus)\b",
                r"\b(others|people|someone|anyone|everyone|colleague|friend)\b"
            ],
            "pragmatic": [
                r"\b(practical|realistic|doable|feasible|simple|efficient|quick)\b",
                r"\b(works|effective|useful|functional|straightforward|direct)\b",
                r"\b(time|cost|resource|budget|deadline|priority|urgent)\b"
            ]
        }
        
        # Theme detection patterns
        self.theme_patterns = {
            ThoughtTheme.TIME_MANAGEMENT: [
                r"\b(time|schedule|deadline|urgent|priority|organize|plan|calendar)\b",
                r"\b(busy|overwhelmed|rushing|late|procrastinate|delay|manage)\b",
                r"\b(efficient|productivity|focus|distraction|multitask)\b"
            ],
            ThoughtTheme.CREATIVITY: [
                r"\b(creative|art|design|innovative|original|imagination|inspire)\b",
                r"\b(brainstorm|idea|concept|vision|artistic|inventive|novel)\b",
                r"\b(express|create|make|build|craft|compose|draw|write)\b"
            ],
            ThoughtTheme.PROBLEM_SOLVING: [
                r"\b(problem|solution|solve|fix|resolve|challenge|issue|difficulty)\b",
                r"\b(approach|method|strategy|technique|way|how to|process)\b",
                r"\b(analyze|troubleshoot|debug|investigate|figure out)\b"
            ],
            ThoughtTheme.RELATIONSHIPS: [
                r"\b(relationship|friend|family|partner|colleague|social|connect)\b",
                r"\b(communicate|talk|discuss|share|listen|understand|empathy)\b",
                r"\b(conflict|harmony|support|trust|love|care|bond)\b"
            ],
            ThoughtTheme.PRODUCTIVITY: [
                r"\b(productive|efficient|output|achieve|accomplish|complete|finish)\b",
                r"\b(goal|target|milestone|progress|result|success|performance)\b",
                r"\b(improve|optimize|streamline|automate|system|workflow)\b"
            ],
            ThoughtTheme.PERSONAL_GROWTH: [
                r"\b(growth|develop|improve|learn|skill|knowledge|education)\b",
                r"\b(self|personal|better|progress|advance|evolve|mature)\b",
                r"\b(mindset|habit|behavior|change|transform|journey)\b"
            ],
            ThoughtTheme.EMOTIONAL_AWARENESS: [
                r"\b(feel|emotion|mood|sentiment|anxiety|stress|calm|happy|sad)\b",
                r"\b(mindful|aware|conscious|reflect|meditation|balance)\b",
                r"\b(therapy|counseling|mental health|wellbeing|self-care)\b"
            ],
            ThoughtTheme.DECISION_MAKING: [
                r"\b(decide|choice|option|alternative|consider|weigh|evaluate)\b",
                r"\b(uncertain|confused|clear|confident|hesitate|doubt)\b",
                r"\b(pros and cons|trade-off|risk|benefit|consequence|outcome)\b"
            ],
            ThoughtTheme.LEARNING: [
                r"\b(learn|study|understand|knowledge|skill|practice|training)\b",
                r"\b(course|class|book|research|explore|discover|curious)\b",
                r"\b(teacher|mentor|expert|guide|instruction|education)\b"
            ],
            ThoughtTheme.STRESS_MANAGEMENT: [
                r"\b(stress|pressure|overwhelm|anxiety|tension|worry|concern)\b",
                r"\b(relax|calm|peace|rest|break|vacation|recharge)\b",
                r"\b(cope|manage|handle|deal with|overcome|resilience)\b"
            ]
        }
    
    async def analyze_problem_solving_style(
        self,
        user_id: str,
        time_window_days: int = 7
    ) -> ProblemSolvingStyle:
        """
        Analyze user's problem-solving style from recent interactions.
        
        Args:
            user_id: User identifier
            time_window_days: Number of days to look back
            
        Returns:
            Problem-solving style analysis
        """
        # In a real implementation, this would query the memory manager
        # For now, we'll create a mock analysis
        
        # This would typically analyze conversation history, problem descriptions,
        # solution attempts, and decision-making patterns
        
        # Mock analysis - in production, replace with actual conversation analysis
        interactions = await self._get_user_interactions(user_id, time_window_days)
        
        if not interactions:
            return self._create_default_style()
        
        # Analyze style indicators
        style_scores = self._calculate_style_scores(interactions)
        dominant_style = max(style_scores, key=style_scores.get)
        confidence = style_scores[dominant_style] / sum(style_scores.values())
        
        return self._create_style_profile(dominant_style, confidence, interactions)
    
    async def _get_user_interactions(self, user_id: str, days: int) -> List[str]:
        """Get user interactions from memory (mock implementation)."""
        # In production, this would query the memory manager for conversations,
        # notes, tasks, and other interactions
        
        # Mock data for demonstration
        mock_interactions = [
            "I need to organize my daily schedule better. Let me break this down step by step.",
            "First, I'll analyze my current time usage, then identify inefficiencies.",
            "What creative approaches could I use to manage my energy levels?",
            "I feel like I should trust my instincts more when making decisions.",
            "Let me research some productivity systems and compare their effectiveness.",
            "Maybe I could brainstorm with my team about this challenge.",
            "I need a practical solution that I can implement immediately.",
            "How can I balance creativity with the need for structured planning?"
        ]
        
        return mock_interactions
    
    def _calculate_style_scores(self, interactions: List[str]) -> Dict[str, float]:
        """Calculate scores for different problem-solving styles."""
        style_scores = defaultdict(float)
        total_text = " ".join(interactions).lower()
        
        for style, patterns in self.style_indicators.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, total_text, re.IGNORECASE))
                style_scores[style] += matches
        
        # Normalize scores
        total_score = sum(style_scores.values())
        if total_score > 0:
            for style in style_scores:
                style_scores[style] = style_scores[style] / total_score
        
        return dict(style_scores)
    
    def _create_style_profile(
        self,
        dominant_style: str,
        confidence: float,
        interactions: List[str]
    ) -> ProblemSolvingStyle:
        """Create a comprehensive style profile."""
        
        style_profiles = {
            "analytical": ProblemSolvingStyle(
                style_name="Analytical Thinker",
                characteristics=[
                    "You approach problems systematically and logically",
                    "You prefer breaking down complex issues into smaller parts",
                    "You value data and evidence in decision-making",
                    "You like to evaluate pros and cons carefully"
                ],
                strengths=[
                    "Thorough analysis leads to well-informed decisions",
                    "Systematic approach reduces risk of overlooking details",
                    "Strong logical reasoning abilities",
                    "Good at identifying patterns and relationships"
                ],
                suggestions=[
                    "Balance analysis with action to avoid analysis paralysis",
                    "Set time limits for research and evaluation phases",
                    "Trust your intuition occasionally for faster decisions",
                    "Consider emotional factors alongside logical ones"
                ],
                confidence=confidence,
                examples=self._extract_relevant_examples(interactions, "analytical")
            ),
            "creative": ProblemSolvingStyle(
                style_name="Creative Innovator",
                characteristics=[
                    "You generate multiple novel solutions to problems",
                    "You think outside conventional boundaries",
                    "You enjoy exploring possibilities and alternatives",
                    "You use imagination and inspiration in problem-solving"
                ],
                strengths=[
                    "Ability to find unique and innovative solutions",
                    "Flexible thinking that adapts to new situations",
                    "Strong imagination and ideation skills",
                    "Good at seeing connections others might miss"
                ],
                suggestions=[
                    "Develop systems to capture and organize your ideas",
                    "Practice evaluating ideas for feasibility",
                    "Collaborate with analytical thinkers for implementation",
                    "Set aside dedicated time for creative thinking"
                ],
                confidence=confidence,
                examples=self._extract_relevant_examples(interactions, "creative")
            ),
            "intuitive": ProblemSolvingStyle(
                style_name="Intuitive Decision Maker",
                characteristics=[
                    "You rely on gut feelings and first impressions",
                    "You make quick decisions based on patterns you sense",
                    "You trust your instincts and inner wisdom",
                    "You prefer holistic rather than detailed analysis"
                ],
                strengths=[
                    "Quick decision-making in uncertain situations",
                    "Good at reading between the lines",
                    "Strong pattern recognition abilities",
                    "Comfortable with ambiguity and uncertainty"
                ],
                suggestions=[
                    "Validate important decisions with some analysis",
                    "Document your reasoning for future reference",
                    "Combine intuition with data for complex decisions",
                    "Practice explaining your intuitive insights to others"
                ],
                confidence=confidence,
                examples=self._extract_relevant_examples(interactions, "intuitive")
            ),
            "collaborative": ProblemSolvingStyle(
                style_name="Collaborative Problem Solver",
                characteristics=[
                    "You prefer working with others to solve problems",
                    "You value diverse perspectives and input",
                    "You build consensus and seek team buy-in",
                    "You communicate openly about challenges"
                ],
                strengths=[
                    "Leverages collective wisdom and expertise",
                    "Builds strong team relationships and trust",
                    "Good at facilitating group problem-solving",
                    "Creates solutions with broad support"
                ],
                suggestions=[
                    "Develop skills for independent problem-solving",
                    "Practice making decisions when consensus isn't possible",
                    "Learn to manage group dynamics effectively",
                    "Balance collaboration with efficiency needs"
                ],
                confidence=confidence,
                examples=self._extract_relevant_examples(interactions, "collaborative")
            ),
            "pragmatic": ProblemSolvingStyle(
                style_name="Pragmatic Problem Solver",
                characteristics=[
                    "You focus on practical and implementable solutions",
                    "You consider resources and constraints realistically",
                    "You prefer simple and direct approaches",
                    "You prioritize what works over what's perfect"
                ],
                strengths=[
                    "Solutions are realistic and actionable",
                    "Efficient use of time and resources",
                    "Good at prioritizing and focusing efforts",
                    "Strong implementation and execution skills"
                ],
                suggestions=[
                    "Occasionally explore more innovative approaches",
                    "Consider long-term implications alongside immediate needs",
                    "Balance practicality with quality and excellence",
                    "Don't dismiss ideas too quickly for being 'impractical'"
                ],
                confidence=confidence,
                examples=self._extract_relevant_examples(interactions, "pragmatic")
            )
        }
        
        return style_profiles.get(dominant_style, self._create_default_style())
    
    def _create_default_style(self) -> ProblemSolvingStyle:
        """Create a default style when insufficient data is available."""
        return ProblemSolvingStyle(
            style_name="Emerging Problem Solver",
            characteristics=[
                "Your problem-solving style is still developing",
                "You show flexibility in approaching different challenges",
                "You're open to learning new approaches and methods"
            ],
            strengths=[
                "Openness to learning and growth",
                "Flexibility in trying different approaches",
                "Willingness to adapt and experiment"
            ],
            suggestions=[
                "Continue engaging with diverse types of problems",
                "Observe and learn from different problem-solving approaches",
                "Practice reflecting on what works best for you",
                "Experiment with structured and creative methods"
            ],
            confidence=0.3,  # Low confidence due to insufficient data
            examples=[]
        )
    
    def _extract_relevant_examples(self, interactions: List[str], style: str) -> List[str]:
        """Extract examples from interactions that demonstrate the given style."""
        examples = []
        patterns = self.style_indicators.get(style, [])
        
        for interaction in interactions:
            for pattern in patterns:
                if re.search(pattern, interaction, re.IGNORECASE):
                    examples.append(interaction)
                    break  # Only add each interaction once
        
        return examples[:3]  # Return up to 3 examples
    
    def analyze_themes_in_text(self, text: str) -> Dict[ThoughtTheme, float]:
        """Analyze themes present in a text."""
        theme_scores = {}
        text_lower = text.lower()
        
        for theme, patterns in self.theme_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            if score > 0:
                # Normalize by text length
                theme_scores[theme] = score / len(text.split())
        
        return theme_scores