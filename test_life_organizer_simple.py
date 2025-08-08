"""
Simple Test for Life Organizer Skill - اختبار بسيط لمهارة منظم الحياة
Author: Drmusab
Last Modified: 2025-01-20

A simplified test to validate the core Life Organizer functionality.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, AsyncMock


# Mock basic components for testing
class MockLogger:
    def info(self, msg): print(f"ℹ️  {msg}")
    def error(self, msg): print(f"❌ {msg}")
    def warning(self, msg): print(f"⚠️  {msg}")


class MockContainer:
    def resolve(self, service_type):
        mock = MagicMock()
        # Add async methods where needed
        if hasattr(mock, 'analyze_sentiment'):
            mock.analyze_sentiment = AsyncMock(return_value={
                "polarity": 0.7, "arousal": 0.8, "confidence": 0.9
            })
        if hasattr(mock, 'analyze_expression'):
            mock.analyze_expression = AsyncMock(return_value={
                "emotions": {"happy": 0.8, "sad": 0.1},
                "energy_indicators": {"alertness": 0.9, "engagement": 0.8},
                "confidence": 0.85
            })
        if hasattr(mock, 'emit'):
            mock.emit = AsyncMock()
        return mock


# Import and test the mood energy tracker directly
def test_mood_energy_tracker():
    """Test the mood energy tracker component."""
    print("🧪 Testing Mood Energy Tracker...")
    
    try:
        # Test the enums
        from src.skills.builtin.life_organizer.mood_energy_tracker import (
            MoodLevel, EnergyLevel, MoodEnergyState
        )
        
        # Test enum values (Arabic first)
        assert MoodLevel.VERY_HIGH.value == "عالي جداً"
        assert EnergyLevel.ENERGETIC.value == "نشيط"
        
        print("✅ Mood and Energy enums working correctly")
        
        # Test data structure
        state = MoodEnergyState(
            mood_level=MoodLevel.HIGH,
            energy_level=EnergyLevel.HIGH,
            confidence=0.9,
            timestamp=datetime.now(timezone.utc),
            sources=["voice", "vision"]
        )
        
        assert state.mood_level == MoodLevel.HIGH
        assert state.confidence == 0.9
        
        print("✅ MoodEnergyState data structure working")
        
        return True
        
    except Exception as e:
        print(f"❌ Mood Energy Tracker test failed: {e}")
        return False


def test_voice_kanban_interface():
    """Test the voice Kanban interface component."""
    print("🧪 Testing Voice Kanban Interface...")
    
    try:
        from src.skills.builtin.life_organizer.voice_kanban_interface import (
            KanbanColumn, TaskPriority, KanbanTask, KanbanBoard
        )
        
        # Test enums (Arabic values)
        assert KanbanColumn.TODO.value == "للقيام"
        assert TaskPriority.URGENT.value == "عاجل"
        
        print("✅ Kanban enums working correctly")
        
        # Test task structure
        task = KanbanTask(
            id=str(uuid.uuid4()),
            title="كتابة التقرير",
            description="تقرير شهري",
            column=KanbanColumn.TODO,
            priority=TaskPriority.HIGH,
            tags=["عمل", "تقرير"]
        )
        
        assert task.title == "كتابة التقرير"
        assert task.column == KanbanColumn.TODO
        
        print("✅ KanbanTask structure working")
        
        # Test board structure
        board = KanbanBoard(
            id="test_board",
            name="لوحة الاختبار"
        )
        
        board.tasks[task.id] = task
        assert len(board.tasks) == 1
        
        print("✅ KanbanBoard structure working")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice Kanban Interface test failed: {e}")
        return False


def test_adaptive_recommendation_engine():
    """Test the adaptive recommendation engine component."""
    print("🧪 Testing Adaptive Recommendation Engine...")
    
    try:
        from src.skills.builtin.life_organizer.adaptive_recommendation_engine import (
            RecommendationType, AdaptiveRecommendation
        )
        
        # Test recommendation types (Arabic values)
        assert RecommendationType.TAKE_BREAK.value == "أخذ استراحة"
        assert RecommendationType.RESCHEDULE.value == "إعادة جدولة"
        
        print("✅ Recommendation types working correctly")
        
        # Test recommendation structure
        recommendation = AdaptiveRecommendation(
            id="test_rec",
            type=RecommendationType.TAKE_BREAK,
            title_arabic="خذ استراحة",
            title_english="Take a break",
            description_arabic="حان وقت الراحة",
            description_english="Time for a rest",
            confidence=0.8,
            suggested_action="schedule_break",
            reasoning_arabic="تبدو متعباً",
            reasoning_english="You seem tired",
            urgency=3,
            estimated_benefit=0.7
        )
        
        assert recommendation.title_arabic == "خذ استراحة"
        assert recommendation.urgency == 3
        
        print("✅ AdaptiveRecommendation structure working")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive Recommendation Engine test failed: {e}")
        return False


async def test_mood_tracker_functionality():
    """Test mood tracker functionality with mocks."""
    print("🧪 Testing Mood Tracker Functionality...")
    
    try:
        from src.skills.builtin.life_organizer.mood_energy_tracker import (
            MoodEnergyTracker, MoodLevel, EnergyLevel
        )
        
        # Create tracker with mock container
        container = MockContainer()
        tracker = MoodEnergyTracker(container)
        
        # Test score mapping functions
        mood_score = tracker._mood_to_score(MoodLevel.HIGH)
        assert mood_score == 0.5
        
        energy_score = tracker._energy_to_score(EnergyLevel.HIGH)
        assert energy_score == 0.75
        
        print("✅ Score mapping functions working")
        
        # Test recommendations initialization
        assert len(tracker.recommendations) > 0
        
        # Test key for high mood/energy
        key = (MoodLevel.HIGH, EnergyLevel.HIGH)
        assert key in tracker.recommendations
        
        print("✅ Recommendations initialization working")
        
        return True
        
    except Exception as e:
        print(f"❌ Mood Tracker Functionality test failed: {e}")
        return False


async def test_kanban_voice_patterns():
    """Test Kanban voice pattern matching."""
    print("🧪 Testing Kanban Voice Patterns...")
    
    try:
        from src.skills.builtin.life_organizer.voice_kanban_interface import (
            VoiceKanbanInterface, KanbanColumn, TaskPriority
        )
        
        # Create interface with mock container
        container = MockContainer()
        interface = VoiceKanbanInterface(container)
        
        # Test Arabic task title extraction
        title = interface._extract_task_title("أضف مهمة كتابة التقرير", "ar")
        assert title == "كتابة التقرير"
        
        print("✅ Arabic task title extraction working")
        
        # Test English task title extraction
        title = interface._extract_task_title("add task write report", "en")
        assert title == "write report"
        
        print("✅ English task title extraction working")
        
        # Test priority extraction
        priority = interface._extract_priority("هذه مهمة عاجل", "ar")
        assert priority == TaskPriority.URGENT
        
        priority = interface._extract_priority("this is urgent task", "en")
        assert priority == TaskPriority.URGENT
        
        print("✅ Priority extraction working")
        
        # Test column extraction
        column = interface._extract_column("ضع في قائمة المهام", "ar")
        assert column == KanbanColumn.BACKLOG
        
        column = interface._extract_column("put in todo", "en")
        assert column == KanbanColumn.TODO
        
        print("✅ Column extraction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Kanban Voice Patterns test failed: {e}")
        return False


def test_life_organizer_metadata():
    """Test Life Organizer skill metadata."""
    print("🧪 Testing Life Organizer Metadata...")
    
    try:
        from src.skills.builtin.life_organizer.life_organizer import (
            LifeOrganizerSkill, LifeOrganizerMode, LifeGoal
        )
        
        # Test operating modes
        assert LifeOrganizerMode.GOAL_PLANNING.value == "تخطيط الأهداف"
        assert LifeOrganizerMode.MOOD_TRACKING.value == "تتبع المزاج"
        
        print("✅ Operating modes working correctly")
        
        # Test life goal structure
        goal = LifeGoal(
            id="test_goal",
            title="تعلم البرمجة",
            description="تطوير مهارات البرمجة",
            category="تعليم",
            priority="عالي"
        )
        
        assert goal.title == "تعلم البرمجة"
        assert goal.category == "تعليم"
        
        print("✅ LifeGoal structure working")
        
        return True
        
    except Exception as e:
        print(f"❌ Life Organizer Metadata test failed: {e}")
        return False


async def test_arabic_language_support():
    """Test Arabic language support throughout the system."""
    print("🧪 Testing Arabic Language Support...")
    
    try:
        # Test all Arabic enums and text
        from src.skills.builtin.life_organizer.mood_energy_tracker import MoodLevel, EnergyLevel
        from src.skills.builtin.life_organizer.voice_kanban_interface import KanbanColumn, TaskPriority
        from src.skills.builtin.life_organizer.adaptive_recommendation_engine import RecommendationType
        
        # Verify Arabic text in all enums
        arabic_texts = [
            MoodLevel.VERY_HIGH.value,  # "عالي جداً"
            EnergyLevel.ENERGETIC.value,  # "نشيط"
            KanbanColumn.TODO.value,  # "للقيام"
            TaskPriority.URGENT.value,  # "عاجل"
            RecommendationType.TAKE_BREAK.value  # "أخذ استراحة"
        ]
        
        # Check that all contain Arabic characters
        arabic_chars = set("اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويى")
        
        for text in arabic_texts:
            has_arabic = any(char in arabic_chars for char in text)
            assert has_arabic, f"Text '{text}' should contain Arabic characters"
        
        print("✅ All enums contain proper Arabic text")
        
        return True
        
    except Exception as e:
        print(f"❌ Arabic Language Support test failed: {e}")
        return False


async def run_simple_tests():
    """Run all simple tests."""
    print("🧪 Testing Life Organizer Skill Components...")
    print("=" * 60)
    
    tests = [
        ("Mood Energy Tracker", test_mood_energy_tracker),
        ("Voice Kanban Interface", test_voice_kanban_interface),
        ("Adaptive Recommendation Engine", test_adaptive_recommendation_engine),
        ("Life Organizer Metadata", test_life_organizer_metadata),
        ("Mood Tracker Functionality", test_mood_tracker_functionality),
        ("Kanban Voice Patterns", test_kanban_voice_patterns),
        ("Arabic Language Support", test_arabic_language_support)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
        
        print()
    
    print("=" * 60)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All component tests passed!")
        print()
        print("✨ Life Organizer Features Validated:")
        print("   • Arabic-first language support ✅")
        print("   • Mood and energy level enums ✅")
        print("   • Voice Kanban data structures ✅")
        print("   • Adaptive recommendation types ✅")
        print("   • Goal planning structures ✅")
        print("   • Voice command pattern matching ✅")
        print("   • Multilingual support (Arabic/English) ✅")
        print()
        print("🚀 Ready for integration with the main AI assistant!")
    else:
        failed = total - passed
        print(f"⚠️  {failed} test(s) failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_simple_tests())
    exit(0 if success else 1)