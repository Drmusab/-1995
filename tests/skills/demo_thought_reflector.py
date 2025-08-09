"""
Demonstration of Thought Reflector Skill Capabilities
Author: Drmusab
Last Modified: 2025-01-20

Shows examples of what the Thought Reflector skill can generate.
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def demonstrate_features():
    """Demonstrate the key features of the Thought Reflector skill."""
    
    print("🧠 Meta-Cognition / Thought Reflection Skill Demonstration")
    print("=" * 60)
    
    # Load types
    exec(open('src/skills/builtin/thought_reflector/types.py').read(), globals())
    
    print("\n📊 1. THOUGHT THEMES")
    print("The skill can identify and analyze these thought patterns:")
    for theme in ThoughtTheme:
        print(f"  • {theme.value.replace('_', ' ').title()}")
    
    print("\n🎯 2. REFLECTION TYPES")
    print("The skill can generate these types of reflections:")
    for reflection_type in ReflectionType:
        description = {
            ReflectionType.WEEKLY_SUMMARY: "Summarizes your thoughts across the week",
            ReflectionType.PROBLEM_SOLVING_STYLE: "Reflects on your problem-solving style", 
            ReflectionType.JOURNALING_PROMPT: "Provides personalized journaling prompts",
            ReflectionType.AFFIRMATION: "Creates affirmations based on your patterns",
            ReflectionType.REFRAMING_EXERCISE: "Helps reframe challenging thoughts",
            ReflectionType.DEEPER_INQUIRY: "Encourages deeper self-exploration"
        }
        print(f"  • {reflection_type.value.replace('_', ' ').title()}: {description[reflection_type]}")
    
    print("\n💭 3. SAMPLE WEEKLY SUMMARY")
    print("-" * 40)
    sample_summary = """## Weekly Thought Reflection Summary

Over the past week, you've engaged in 12 meaningful interactions. Your thoughts have primarily centered around **Time Management and Creativity**.

### Key Insights:
- You've been thinking a lot about time management and efficiency
- Creative expression and innovation are important to you
- You approach challenges with structured thinking and analytical methods

### Patterns Discovered:
- **Time Management**: Appeared 8 times with 85% confidence
- **Creativity**: Appeared 5 times with 72% confidence
- **Problem Solving**: Appeared 4 times with 68% confidence

### Questions for Reflection:
- How do these patterns align with your current goals and values?
- What insights surprise you about your thinking patterns?
- Which patterns would you like to strengthen or modify?

Remember, self-awareness is the first step toward intentional growth and positive change."""
    print(sample_summary)
    
    print("\n🎨 4. SAMPLE JOURNALING PROMPT")
    print("-" * 40)
    sample_prompt = """## Journaling Prompt

**Reflect on a recent day when you felt in control of your time. What made it different?**

### Reflection Guidelines:
- Set aside 10-15 minutes without interruptions
- Write freely without worrying about grammar or structure
- Allow your thoughts to flow naturally
- Be honest and compassionate with yourself
- Don't judge your thoughts—simply observe and explore

*This prompt is inspired by your recent focus on Time Management.*"""
    print(sample_prompt)
    
    print("\n💪 5. SAMPLE AFFIRMATION")
    print("-" * 40)
    sample_affirmation = """## Personal Affirmation

**I am in control of my time and use it wisely to support my goals and well-being.**

### How to Use This Affirmation:
- Repeat it to yourself throughout the day
- Say it aloud to hear the words in your own voice
- Write it down and place it somewhere you'll see it often
- Take a moment to truly feel the meaning behind the words
- Adapt the language to make it feel more personal to you

*This affirmation is crafted to support your growth in Time Management.*"""
    print(sample_affirmation)
    
    print("\n🔄 6. SAMPLE REFRAMING EXERCISE")
    print("-" * 40)
    sample_reframing = """## Reframing Exercise: The Multiple Perspectives Exercise

Consider this exercise particularly in relation to your thoughts about Time Management and Creativity.

### Steps:
1. Write down the challenging thought or situation you're facing.
2. Describe it from your current perspective, including your emotions.
3. Now imagine how a wise, compassionate friend would view this situation.
4. Consider how you might view this same situation 5 years from now.
5. Write a balanced perspective that incorporates insights from steps 3 and 4.

### Remember:
- Take your time with each step
- Be patient and compassionate with yourself
- It's okay if new perspectives don't feel immediately convincing
- The goal is to expand your thinking, not to eliminate all negative thoughts
- Practice makes this process more natural and effective"""
    print(sample_reframing)
    
    print("\n🤔 7. SAMPLE DEEPER INQUIRY")
    print("-" * 40)
    sample_inquiry = """## Deeper Inquiry Questions

Take time to sit with these questions. There are no right or wrong answers—the value lies in the exploration itself.

**1. What patterns do you notice in how you respond to uncertainty?**

*Pause here to reflect before moving to the next question.*

**2. What does 'enough time' mean to you, and how would you know when you have it?**

*Pause here to reflect before moving to the next question.*

**3. How does your relationship with time reflect your relationship with yourself?**

*Pause here to reflect before moving to the next question.*

**4. What would you prioritize if you only had half the time you currently have?**

*Pause here to reflect before moving to the next question.*

### Approach:
- Don't rush to answer immediately
- Allow unexpected insights to emerge
- Notice what feelings arise with each question
- Consider journaling about your responses
- Discuss with a trusted friend or counselor if helpful"""
    print(sample_inquiry)
    
    print("\n🎯 8. HOW TO USE THE SKILL")
    print("-" * 40)
    usage_examples = [
        "Give me a weekly summary of my thought patterns",
        "What's my problem-solving style?", 
        "Generate a journaling prompt for me",
        "I need an affirmation based on my recent thoughts",
        "Help me reframe this challenging situation",
        "Give me some deeper questions to reflect on"
    ]
    
    print("You can interact with the skill using natural language like:")
    for example in usage_examples:
        print(f"  • \"{example}\"")
    
    print("\n✨ KEY BENEFITS")
    print("-" * 40)
    benefits = [
        "Increased self-awareness through pattern recognition",
        "Personalized insights based on your unique thinking style",
        "Structured reflection exercises for deeper understanding", 
        "Encouraging affirmations tailored to your needs",
        "Tools for reframing negative thoughts constructively",
        "Weekly summaries to track your mental and emotional patterns",
        "Guided questions for meaningful self-exploration"
    ]
    
    for benefit in benefits:
        print(f"  ✓ {benefit}")
    
    print(f"\n🌟 The Thought Reflector skill helps you develop meta-cognitive awareness—")
    print("   thinking about thinking—which is essential for personal growth,")
    print("   emotional intelligence, and intentional living.")
    
    print("\n" + "=" * 60)
    print("💡 Ready to begin your journey of self-reflection and awareness!")

if __name__ == "__main__":
    demonstrate_features()