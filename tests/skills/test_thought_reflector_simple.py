"""
Simple test for Thought Reflector Skill types and basic functionality
Author: Drmusab
Last Modified: 2025-01-20

Tests the basic types and structure without full dependency loading.
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_types():
    """Test the types module."""
    try:
        from src.skills.builtin.thought_reflector.types import (
            ReflectionType, ThoughtTheme, ThoughtPattern, ProblemSolvingStyle, ReflectionResult
        )
        
        print("✓ Successfully imported all types")
        
        # Test enums
        print(f"✓ ReflectionType has {len(ReflectionType)} values")
        print(f"✓ ThoughtTheme has {len(ThoughtTheme)} values")
        
        # Test creating objects
        pattern = ThoughtPattern(
            theme=ThoughtTheme.TIME_MANAGEMENT,
            frequency=5,
            confidence=0.8,
            examples=["I need to manage my time better"],
            insights=["Time management is a key focus area"]
        )
        print(f"✓ Created ThoughtPattern: {pattern.theme.value}")
        
        style = ProblemSolvingStyle(
            style_name="Analytical Thinker",
            characteristics=["Systematic approach", "Data-driven"],
            strengths=["Thorough analysis", "Logical reasoning"],
            suggestions=["Trust intuition more", "Set time limits"],
            confidence=0.75
        )
        print(f"✓ Created ProblemSolvingStyle: {style.style_name}")
        
        result = ReflectionResult(
            reflection_type=ReflectionType.WEEKLY_SUMMARY,
            content="This is a test weekly summary",
            themes=[ThoughtTheme.TIME_MANAGEMENT],
            patterns=[pattern],
            suggestions=["Continue reflecting"]
        )
        print(f"✓ Created ReflectionResult: {result.reflection_type.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Types test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_helper_modules():
    """Test the helper modules can be imported."""
    try:
        from src.skills.builtin.thought_reflector.types import ThoughtTheme, ThoughtPattern
        
        # Test thought analyzer
        from src.skills.builtin.thought_reflector.thought_analyzer import ThoughtAnalyzer
        analyzer = ThoughtAnalyzer()
        print("✓ Successfully created ThoughtAnalyzer")
        
        # Test reflection generator  
        from src.skills.builtin.thought_reflector.reflection_generator import ReflectionGenerator
        generator = ReflectionGenerator()
        print("✓ Successfully created ReflectionGenerator")
        
        # Test pattern detector
        from src.skills.builtin.thought_reflector.pattern_detector import PatternDetector
        detector = PatternDetector()
        print("✓ Successfully created PatternDetector")
        
        # Test weekly summarizer
        from src.skills.builtin.thought_reflector.weekly_summarizer import WeeklySummarizer
        summarizer = WeeklySummarizer()
        print("✓ Successfully created WeeklySummarizer")
        
        # Test some basic functionality
        themes = analyzer.analyze_themes_in_text("I need to manage my time better and be more creative")
        print(f"✓ Theme analysis returned {len(themes)} themes")
        
        return True
        
    except Exception as e:
        print(f"❌ Helper modules test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_skill_structure():
    """Test that the skill directory structure is correct."""
    skill_dir = "src/skills/builtin/thought_reflector"
    
    expected_files = [
        "__init__.py",
        "types.py", 
        "thought_reflector_skill.py",
        "thought_analyzer.py",
        "reflection_generator.py",
        "pattern_detector.py",
        "weekly_summarizer.py"
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(os.path.join(skill_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✓ All expected files are present")
        return True

def test_configuration():
    """Test that the configuration has been added."""
    config_file = "configs/skills/skill_configs.yaml"
    
    if not os.path.exists(config_file):
        print("❌ Configuration file not found")
        return False
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    if "thought_reflector" in content and "meta_cognitive" in content:
        print("✓ Configuration has been added")
        return True
    else:
        print("❌ Configuration not found in file")
        return False

def main():
    """Run all tests."""
    print("🧠 Testing Thought Reflector Skill Implementation\n")
    
    tests = [
        ("File structure", test_skill_structure),
        ("Configuration", test_configuration), 
        ("Types module", test_types),
        ("Helper modules", test_helper_modules)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Thought Reflector skill is properly implemented.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)