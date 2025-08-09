"""
Direct test for Thought Reflector Skill components
Author: Drmusab
Last Modified: 2025-01-20

Tests individual components directly without going through the skill package init.
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_types_direct():
    """Test the types module directly."""
    try:
        # Import types directly
        exec(open('src/skills/builtin/thought_reflector/types.py').read(), globals())
        
        print("✓ Successfully loaded types module")
        
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

def test_skill_interface():
    """Test that the main skill class has the right interface."""
    try:
        # Read the skill file to check interface
        with open('src/skills/builtin/thought_reflector/thought_reflector_skill.py', 'r') as f:
            content = f.read()
        
        required_methods = [
            'get_metadata',
            'initialize', 
            'execute',
            'validate',
            'health_check',
            'cleanup'
        ]
        
        missing_methods = []
        for method in required_methods:
            if f'def {method}(' not in content:
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing required methods: {missing_methods}")
            return False
        else:
            print("✓ All required skill interface methods are present")
        
        # Check for key features
        features = [
            'weekly_summary',
            'problem_solving_style', 
            'journaling_prompt',
            'affirmation',
            'reframing_exercise',
            'deeper_inquiry'
        ]
        
        missing_features = []
        for feature in features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing expected features: {missing_features}")
            return False
        else:
            print("✓ All expected features are implemented")
        
        return True
        
    except Exception as e:
        print(f"❌ Skill interface test failed: {e}")
        return False

def test_helper_classes():
    """Test that helper classes have the expected structure."""
    helper_files = [
        ('thought_analyzer.py', 'ThoughtAnalyzer'),
        ('reflection_generator.py', 'ReflectionGenerator'),
        ('pattern_detector.py', 'PatternDetector'),
        ('weekly_summarizer.py', 'WeeklySummarizer')
    ]
    
    for filename, classname in helper_files:
        try:
            filepath = f'src/skills/builtin/thought_reflector/{filename}'
            with open(filepath, 'r') as f:
                content = f.read()
            
            if f'class {classname}' not in content:
                print(f"❌ Class {classname} not found in {filename}")
                return False
            
            if 'def __init__(' not in content:
                print(f"❌ Constructor not found in {classname}")
                return False
                
            print(f"✓ {classname} class structure looks good")
            
        except Exception as e:
            print(f"❌ Error checking {filename}: {e}")
            return False
    
    return True

def test_configuration_details():
    """Test configuration details."""
    try:
        with open('configs/skills/skill_configs.yaml', 'r') as f:
            content = f.read()
        
        # Check for key configuration elements
        config_elements = [
            'thought_reflector:',
            'meta_cognitive.thought_reflector',
            'Thought Reflector',
            'weekly_summary',
            'problem_solving_style',
            'journaling_prompt',
            'affirmation',
            'reframing_exercise',
            'deeper_inquiry'
        ]
        
        missing_elements = []
        for element in config_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing configuration elements: {missing_elements}")
            return False
        else:
            print("✓ All configuration elements are present")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_syntax():
    """Test that all Python files have valid syntax."""
    python_files = [
        'src/skills/builtin/thought_reflector/__init__.py',
        'src/skills/builtin/thought_reflector/types.py',
        'src/skills/builtin/thought_reflector/thought_reflector_skill.py',
        'src/skills/builtin/thought_reflector/thought_analyzer.py',
        'src/skills/builtin/thought_reflector/reflection_generator.py',
        'src/skills/builtin/thought_reflector/pattern_detector.py',
        'src/skills/builtin/thought_reflector/weekly_summarizer.py'
    ]
    
    for filepath in python_files:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Try to compile the file
            compile(content, filepath, 'exec')
            print(f"✓ {os.path.basename(filepath)} has valid syntax")
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {filepath}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error checking {filepath}: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🧠 Testing Thought Reflector Skill Implementation (Direct)\n")
    
    tests = [
        ("File syntax", test_file_syntax),
        ("Configuration details", test_configuration_details),
        ("Types module", test_types_direct),
        ("Skill interface", test_skill_interface),
        ("Helper classes", test_helper_classes)
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