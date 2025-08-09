"""
Reflection Generator Module
Author: Drmusab
Last Modified: 2025-01-20

Generates personalized reflective content including journaling prompts,
affirmations, reframing exercises, and deeper inquiry questions.
"""

import random
from typing import List, Dict, Any, Optional

from src.core.config.loader import ConfigLoader
from src.observability.logging.config import get_logger
from .types import ThoughtPattern, ThoughtTheme


class ReflectionGenerator:
    """Generates personalized reflective content based on thought patterns."""
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """Initialize the reflection generator."""
        self.logger = get_logger(__name__)
        self.config = config
        
        # Journaling prompt templates by theme
        self.journaling_prompts = {
            ThoughtTheme.TIME_MANAGEMENT: [
                "Reflect on a recent day when you felt in control of your time. What made it different?",
                "How would your ideal day be structured? What would you prioritize?",
                "Write about a time when you felt overwhelmed. What could you have done differently?",
                "Describe your relationship with time. Is it a friend or foe? Why?",
                "What activities consistently make you lose track of time in a positive way?"
            ],
            ThoughtTheme.CREATIVITY: [
                "Describe a moment when you felt most creatively alive. What sparked that feeling?",
                "Write about an idea that excites you but you haven't pursued yet. What's holding you back?",
                "Reflect on how you express creativity in your daily life, both big and small ways.",
                "Imagine you had unlimited resources to create something. What would it be?",
                "Write about a creative person who inspires you. What qualities do you admire?"
            ],
            ThoughtTheme.PROBLEM_SOLVING: [
                "Think of a recent problem you solved. Walk through your thought process step by step.",
                "Describe a challenge you're currently facing. What different approaches could you try?",
                "Reflect on a time when your usual problem-solving approach didn't work. What did you learn?",
                "Write about someone whose problem-solving skills you admire. What can you learn from them?",
                "How do you typically react when first encountering a difficult problem?"
            ],
            ThoughtTheme.RELATIONSHIPS: [
                "Write a letter of appreciation to someone important in your life (you don't have to send it).",
                "Reflect on a relationship that has changed you for the better. How did it shape you?",
                "Describe your ideal friendship or partnership. What qualities matter most to you?",
                "Think about a difficult conversation you need to have. How could you approach it with compassion?",
                "Write about a time when someone truly understood you. How did that feel?"
            ],
            ThoughtTheme.PERSONAL_GROWTH: [
                "Reflect on how you've grown in the past year. What are you most proud of?",
                "Write about a habit you want to develop. Why is it important to you?",
                "Describe your younger self from five years ago. What would you tell them?",
                "What aspects of yourself are you still discovering? What excites you about that journey?",
                "Reflect on a mistake that became a valuable learning experience."
            ],
            ThoughtTheme.EMOTIONAL_AWARENESS: [
                "Describe an emotion you felt strongly today. What triggered it and how did you respond?",
                "Write about a time when you successfully managed a difficult emotion. What strategies worked?",
                "Reflect on which emotions you find easiest and hardest to express. Why might that be?",
                "How do you typically comfort yourself when you're struggling? What works best?",
                "Write about an emotion you'd like to understand better about yourself."
            ],
            ThoughtTheme.DECISION_MAKING: [
                "Reflect on the best decision you've made in the past year. What made it right for you?",
                "Write about a decision you're avoiding. What fears or concerns are holding you back?",
                "Describe your decision-making process. Do you rely more on logic, intuition, or advice?",
                "Think about a decision you regret. What would you do differently with what you know now?",
                "How do you handle uncertainty when making important choices?"
            ],
            ThoughtTheme.LEARNING: [
                "Reflect on something new you learned recently. How did it change your perspective?",
                "Write about a skill you've always wanted to develop. What's your plan to start?",
                "Describe your favorite way to learn new things. What environment helps you absorb information?",
                "Think about a teacher or mentor who influenced you. What did they teach you beyond the subject matter?",
                "How has your approach to learning changed as you've gotten older?"
            ],
            ThoughtTheme.STRESS_MANAGEMENT: [
                "Describe a technique that helps you manage stress. When and how do you use it?",
                "Write about the sources of stress in your life. Which ones can you control or influence?",
                "Reflect on how your body signals stress to you. How can you listen to these signals better?",
                "Think about a time when you handled stress well. What did you do right?",
                "What would change in your life if you worried less? How would you spend that mental energy?"
            ],
            ThoughtTheme.PRODUCTIVITY: [
                "Reflect on when you feel most productive and energized. What conditions support this state?",
                "Write about something you accomplished recently that you're proud of. What made it successful?",
                "Describe your relationship with productivity. Is it healthy or causing you stress?",
                "Think about your most important goals. Are your daily actions aligned with achieving them?",
                "How do you define productivity for yourself? Has this definition changed over time?"
            ]
        }
        
        # Affirmation templates by theme
        self.affirmations = {
            ThoughtTheme.TIME_MANAGEMENT: [
                "I am in control of my time and use it wisely to support my goals and well-being.",
                "I trust my ability to prioritize what truly matters and let go of what doesn't serve me.",
                "My relationship with time is healthy and balanced, allowing for both productivity and rest.",
                "I honor my commitments to myself and others while maintaining healthy boundaries.",
                "I am learning to be present in each moment rather than rushing toward the next."
            ],
            ThoughtTheme.CREATIVITY: [
                "My creativity flows naturally and I trust in my ability to express myself authentically.",
                "I am open to inspiration from unexpected sources and welcome creative surprises.",
                "My unique perspective and experiences make my creative expression valuable and meaningful.",
                "I give myself permission to experiment, play, and create imperfectly.",
                "I am a creative being, and my creativity manifests in countless ways throughout my life."
            ],
            ThoughtTheme.PROBLEM_SOLVING: [
                "I approach challenges with confidence, knowing I have the skills and wisdom to find solutions.",
                "I am resourceful and resilient, capable of thinking creatively when faced with obstacles.",
                "Every problem contains within it the seeds of its own solution, and I trust my ability to find them.",
                "I learn and grow stronger from each challenge I overcome, building my problem-solving abilities.",
                "I remain calm and focused when facing difficulties, allowing clear thinking to guide my actions."
            ],
            ThoughtTheme.RELATIONSHIPS: [
                "I cultivate meaningful connections built on mutual respect, understanding, and authentic communication.",
                "I am worthy of love and belonging, and I attract relationships that support my growth and happiness.",
                "I communicate my needs and boundaries clearly while remaining open to others' perspectives.",
                "I choose to see the best in people while maintaining healthy discernment in my relationships.",
                "I am a good friend to myself and others, offering compassion, support, and genuine care."
            ],
            ThoughtTheme.PERSONAL_GROWTH: [
                "I am constantly evolving and growing, embracing both my strengths and areas for improvement.",
                "I trust the timing of my life and honor my unique path of development and discovery.",
                "I am committed to becoming the best version of myself while accepting where I am right now.",
                "I learn from every experience, whether it brings joy or challenge, growth or rest.",
                "I have the courage to change what I can and the wisdom to accept what I cannot."
            ],
            ThoughtTheme.EMOTIONAL_AWARENESS: [
                "I honor all my emotions as valid sources of information about my needs and experiences.",
                "I am developing greater emotional intelligence and the ability to respond rather than react.",
                "I create safe spaces for myself to feel and process emotions without judgment.",
                "I am learning to balance emotional expression with thoughtful consideration of others.",
                "My emotional life is rich and meaningful, contributing to my overall wisdom and compassion."
            ],
            ThoughtTheme.DECISION_MAKING: [
                "I trust my ability to make good decisions using both my analytical mind and intuitive wisdom.",
                "I gather the information I need while avoiding paralysis from over-analysis.",
                "I make decisions aligned with my values and goals, and I learn from every choice I make.",
                "I am comfortable with uncertainty and make the best decisions I can with available information.",
                "I take responsibility for my choices and view mistakes as opportunities for learning and growth."
            ],
            ThoughtTheme.LEARNING: [
                "I am a lifelong learner, curious and open to new ideas, perspectives, and experiences.",
                "I trust my ability to understand and integrate new information in ways that serve me.",
                "I learn at my own pace and in my own way, honoring my unique learning style and needs.",
                "I am excited about what I don't yet know and approach learning with enthusiasm and patience.",
                "Every day offers opportunities to learn something new about myself, others, and the world."
            ],
            ThoughtTheme.STRESS_MANAGEMENT: [
                "I have effective tools and strategies for managing stress and maintaining my inner peace.",
                "I recognize stress signals early and respond with self-care and appropriate action.",
                "I release what I cannot control and focus my energy on what I can influence.",
                "I maintain perspective during challenging times, knowing that this too shall pass.",
                "I deserve rest, relaxation, and peace, and I create space for these in my life."
            ],
            ThoughtTheme.PRODUCTIVITY: [
                "I am naturally productive when I align my actions with my values and energy levels.",
                "I balance accomplishment with rest, knowing both are essential for sustainable success.",
                "I celebrate progress over perfection and acknowledge my efforts and achievements.",
                "I work with my natural rhythms and honor my need for both focused work and restorative breaks.",
                "My worth is not determined by my productivity; I am valuable simply for being who I am."
            ]
        }
        
        # Reframing exercise templates
        self.reframing_exercises = {
            "general": [
                {
                    "title": "The Multiple Perspectives Exercise",
                    "steps": [
                        "1. Write down the challenging thought or situation you're facing.",
                        "2. Describe it from your current perspective, including your emotions.",
                        "3. Now imagine how a wise, compassionate friend would view this situation.",
                        "4. Consider how you might view this same situation 5 years from now.",
                        "5. Write a balanced perspective that incorporates insights from steps 3 and 4."
                    ]
                },
                {
                    "title": "The Growth Opportunity Frame",
                    "steps": [
                        "1. Identify what you're struggling with or finding difficult.",
                        "2. Ask yourself: 'What could this situation teach me?'",
                        "3. Consider: 'How might this challenge help me grow stronger or wiser?'",
                        "4. Reflect: 'What skills or qualities could I develop through this experience?'",
                        "5. Write a statement that reframes this challenge as a growth opportunity."
                    ]
                },
                {
                    "title": "The Evidence Examination",
                    "steps": [
                        "1. Write down a negative thought or belief you're having.",
                        "2. List evidence that supports this thought.",
                        "3. List evidence that contradicts or challenges this thought.",
                        "4. Consider alternative explanations for the situation.",
                        "5. Create a more balanced and realistic perspective based on all evidence."
                    ]
                }
            ]
        }
        
        # Deeper inquiry questions by theme
        self.deeper_inquiry = {
            "general": [
                "What patterns do you notice in how you respond to uncertainty?",
                "What would you do differently if you knew you couldn't fail?",
                "How do your current choices reflect your deepest values?",
                "What stories do you tell yourself about your limitations?",
                "When do you feel most authentic and true to yourself?",
                "What would change if you trusted yourself completely?",
                "How do you define success, and has this definition evolved?",
                "What does your intuition tell you about your current path?",
                "Where in your life are you playing it too safe?",
                "What aspects of yourself are you still afraid to fully embrace?"
            ],
            ThoughtTheme.TIME_MANAGEMENT: [
                "What does 'enough time' mean to you, and how would you know when you have it?",
                "How does your relationship with time reflect your relationship with yourself?",
                "What would you prioritize if you only had half the time you currently have?",
                "In what ways do you use busyness to avoid something deeper?"
            ],
            ThoughtTheme.CREATIVITY: [
                "What creative expression did you love as a child that you've abandoned?",
                "How does fear of judgment impact your willingness to create?",
                "What would you create if only you would ever see it?",
                "How does creativity serve purposes beyond the final product in your life?"
            ],
            ThoughtTheme.RELATIONSHIPS: [
                "How do you know when you're truly seen and understood by someone?",
                "What patterns from your family of origin do you see in your current relationships?",
                "How comfortable are you with being fully known by others?",
                "What do your relationships teach you about yourself?"
            ]
        }
    
    async def generate_journaling_prompt(self, patterns: List[ThoughtPattern]) -> str:
        """Generate a personalized journaling prompt based on thought patterns."""
        if not patterns:
            # Generic prompt when no patterns are detected
            return self._format_journaling_prompt(
                "Reflect on what's been on your mind lately. What thoughts or feelings have been most present for you?",
                ["general"]
            )
        
        # Choose prompt based on most prominent theme
        dominant_pattern = max(patterns, key=lambda p: p.frequency * p.confidence)
        theme = dominant_pattern.theme
        
        if theme in self.journaling_prompts:
            prompt = random.choice(self.journaling_prompts[theme])
            themes = [theme.value]
        else:
            # Fallback to general prompt
            prompt = "Reflect on the patterns you've noticed in your thinking lately. What themes keep emerging?"
            themes = [p.theme.value for p in patterns[:2]]
        
        return self._format_journaling_prompt(prompt, themes)
    
    async def generate_affirmation(self, patterns: List[ThoughtPattern]) -> str:
        """Generate a personalized affirmation based on thought patterns."""
        if not patterns:
            return "I am growing in self-awareness and trust in my own wisdom and capabilities."
        
        # Choose affirmation based on most prominent theme
        dominant_pattern = max(patterns, key=lambda p: p.frequency * p.confidence)
        theme = dominant_pattern.theme
        
        if theme in self.affirmations:
            affirmation = random.choice(self.affirmations[theme])
        else:
            affirmation = "I trust my journey and embrace both the challenges and joys that come with growth."
        
        return self._format_affirmation(affirmation, theme)
    
    async def generate_reframing_exercise(self, patterns: List[ThoughtPattern]) -> str:
        """Generate a reframing exercise based on thought patterns."""
        exercise = random.choice(self.reframing_exercises["general"])
        
        # Customize based on patterns if available
        if patterns:
            theme_names = [p.theme.value.replace('_', ' ').title() for p in patterns[:2]]
            context = f"Consider this exercise particularly in relation to your thoughts about {' and '.join(theme_names)}."
        else:
            context = "Apply this exercise to any challenging thought or situation you're currently facing."
        
        return self._format_reframing_exercise(exercise, context)
    
    async def generate_deeper_inquiry(self, patterns: List[ThoughtPattern]) -> str:
        """Generate deeper inquiry questions based on thought patterns."""
        questions = []
        
        # Add theme-specific questions if patterns exist
        if patterns:
            for pattern in patterns[:2]:  # Top 2 patterns
                if pattern.theme in self.deeper_inquiry:
                    theme_questions = self.deeper_inquiry[pattern.theme]
                    questions.append(random.choice(theme_questions))
        
        # Add general questions
        general_questions = random.sample(self.deeper_inquiry["general"], 2)
        questions.extend(general_questions)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in questions:
            if q not in seen:
                unique_questions.append(q)
                seen.add(q)
        
        return self._format_deeper_inquiry(unique_questions[:4])  # Limit to 4 questions
    
    def _format_journaling_prompt(self, prompt: str, themes: List[str]) -> str:
        """Format a journaling prompt with context."""
        formatted = "## Journaling Prompt\n\n"
        formatted += f"**{prompt}**\n\n"
        
        formatted += "### Reflection Guidelines:\n"
        formatted += "- Set aside 10-15 minutes without interruptions\n"
        formatted += "- Write freely without worrying about grammar or structure\n"
        formatted += "- Allow your thoughts to flow naturally\n"
        formatted += "- Be honest and compassionate with yourself\n"
        formatted += "- Don't judge your thoughts—simply observe and explore\n\n"
        
        if themes and themes != ["general"]:
            theme_names = [theme.replace('_', ' ').title() for theme in themes]
            formatted += f"*This prompt is inspired by your recent focus on {' and '.join(theme_names)}.*"
        
        return formatted
    
    def _format_affirmation(self, affirmation: str, theme: ThoughtTheme) -> str:
        """Format an affirmation with usage guidance."""
        formatted = "## Personal Affirmation\n\n"
        formatted += f"**{affirmation}**\n\n"
        
        formatted += "### How to Use This Affirmation:\n"
        formatted += "- Repeat it to yourself throughout the day\n"
        formatted += "- Say it aloud to hear the words in your own voice\n"
        formatted += "- Write it down and place it somewhere you'll see it often\n"
        formatted += "- Take a moment to truly feel the meaning behind the words\n"
        formatted += "- Adapt the language to make it feel more personal to you\n\n"
        
        theme_name = theme.value.replace('_', ' ').title()
        formatted += f"*This affirmation is crafted to support your growth in {theme_name}.*"
        
        return formatted
    
    def _format_reframing_exercise(self, exercise: Dict[str, Any], context: str) -> str:
        """Format a reframing exercise with instructions."""
        formatted = f"## Reframing Exercise: {exercise['title']}\n\n"
        formatted += f"{context}\n\n"
        
        formatted += "### Steps:\n"
        for step in exercise['steps']:
            formatted += f"{step}\n"
        formatted += "\n"
        
        formatted += "### Remember:\n"
        formatted += "- Take your time with each step\n"
        formatted += "- Be patient and compassionate with yourself\n"
        formatted += "- It's okay if new perspectives don't feel immediately convincing\n"
        formatted += "- The goal is to expand your thinking, not to eliminate all negative thoughts\n"
        formatted += "- Practice makes this process more natural and effective\n"
        
        return formatted
    
    def _format_deeper_inquiry(self, questions: List[str]) -> str:
        """Format deeper inquiry questions."""
        formatted = "## Deeper Inquiry Questions\n\n"
        formatted += "Take time to sit with these questions. There are no right or wrong answers—"
        formatted += "the value lies in the exploration itself.\n\n"
        
        for i, question in enumerate(questions, 1):
            formatted += f"**{i}. {question}**\n\n"
            formatted += "*Pause here to reflect before moving to the next question.*\n\n"
        
        formatted += "### Approach:\n"
        formatted += "- Don't rush to answer immediately\n"
        formatted += "- Allow unexpected insights to emerge\n"
        formatted += "- Notice what feelings arise with each question\n"
        formatted += "- Consider journaling about your responses\n"
        formatted += "- Discuss with a trusted friend or counselor if helpful\n"
        
        return formatted