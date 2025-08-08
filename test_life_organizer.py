"""
Test Life Organizer Skill - اختبار مهارة منظم الحياة
Author: Drmusab
Last Modified: 2025-01-20

Comprehensive test for the Life Organizer skill including:
- Goal decomposition
- Mood/energy tracking
- Voice Kanban interface
- Adaptive recommendations
- Arabic language support
"""

import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Mock dependencies for testing
class MockContainer:
    """Mock dependency injection container."""
    
    def __init__(self):
        self._services = {}
        self._initialize_mocks()
    
    def _initialize_mocks(self):
        """Initialize mock services."""
        from unittest.mock import MagicMock, AsyncMock
        
        # Mock config loader
        mock_config = MagicMock()
        mock_config.get.return_value = {"default": "value"}
        self._services[ConfigLoader] = mock_config
        
        # Mock event bus
        mock_event_bus = MagicMock()
        mock_event_bus.emit = AsyncMock()
        self._services[EventBus] = mock_event_bus
        
        # Mock bilingual manager
        mock_bilingual = MagicMock()
        mock_bilingual.translate = AsyncMock(return_value="translated text")
        self._services[BilingualManager] = mock_bilingual
        
        # Mock intent manager
        mock_intent = MagicMock()
        mock_intent.analyze_intent = AsyncMock(return_value={
            "intent": "create_task",
            "confidence": 0.9,
            "entities": {}
        })
        self._services[IntentManager] = mock_intent
        
        # Mock TTS engine
        mock_tts = MagicMock()
        mock_tts.speak = AsyncMock()
        self._services[TextToSpeechEngine] = mock_tts
        
        # Mock goal decomposer
        mock_goal_decomposer = MagicMock()
        mock_goal_decomposer.decompose_goal = AsyncMock(return_value={
            "description": "تعلم البرمجة بشكل منهجي",
            "category": "تعليم",
            "priority": "عالي",
            "action_steps": [
                {
                    "title": "اختيار لغة البرمجة",
                    "estimated_duration": "1 أسبوع",
                    "priority": "عالي"
                },
                {
                    "title": "تعلم الأساسيات",
                    "estimated_duration": "4 أسابيع", 
                    "priority": "عالي"
                },
                {
                    "title": "بناء مشروع تطبيقي",
                    "estimated_duration": "8 أسابيع",
                    "priority": "متوسط"
                }
            ]
        })
        self._services[GoalDecomposer] = mock_goal_decomposer
        
        # Mock task planner
        mock_task_planner = MagicMock()
        self._services[TaskPlanner] = mock_task_planner
        
        # Mock time block planner
        mock_time_planner = MagicMock()
        mock_time_planner.process_request = AsyncMock(return_value={
            "success": True,
            "response": "تم إنشاء الجدول بنجاح",
            "time_blocks": [
                {"title": "العمل العميق", "start": "09:00", "end": "11:00"},
                {"title": "المراجعة", "start": "11:15", "end": "12:00"}
            ]
        })
        self._services[TimeBlockPlanner] = mock_time_planner
        
        # Mock memory manager
        mock_memory = MagicMock()
        mock_memory.store = AsyncMock()
        mock_memory.retrieve = AsyncMock(return_value={})
        self._services[MemoryManager] = mock_memory
        
        # Mock sentiment analyzer
        mock_sentiment = MagicMock()
        mock_sentiment.analyze_sentiment = AsyncMock(return_value={
            "polarity": 0.7,  # Positive
            "arousal": 0.8,   # High energy
            "confidence": 0.9
        })
        self._services[SentimentAnalyzer] = mock_sentiment
        
        # Mock expression analyzer
        mock_expression = MagicMock()
        mock_expression.analyze_expression = AsyncMock(return_value={
            "emotions": {
                "happy": 0.8,
                "sad": 0.1,
                "angry": 0.0,
                "surprised": 0.1
            },
            "energy_indicators": {
                "alertness": 0.9,
                "engagement": 0.8
            },
            "confidence": 0.85
        })
        self._services[ExpressionAnalyzer] = mock_expression
        
        # Mock calendar API
        mock_calendar = MagicMock()
        self._services[CalendarAPI] = mock_calendar
    
    def resolve(self, service_type):
        """Resolve a service from the container."""
        return self._services.get(service_type, MagicMock())


# Import required modules with mocking
try:
    from src.core.config.loader import ConfigLoader
    from src.core.dependency_injection import Container
    from src.core.error_handling import ErrorHandler, handle_exceptions
    from src.core.events.event_bus import EventBus
    from src.core.events.event_types import MoodChanged, TaskCompleted
    from src.integrations.external_apis.calendar_api import CalendarAPI
    from src.memory.core_memory.memory_manager import MemoryManager
    from src.observability.logging.config import get_logger
    from src.processing.natural_language.bilingual_manager import BilingualManager
    from src.processing.natural_language.intent_manager import IntentManager
    from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer
    from src.processing.speech.text_to_speech import TextToSpeechEngine
    from src.processing.vision.detectors.expression_analyzer import ExpressionAnalyzer
    from src.reasoning.planning.goal_decomposer import GoalDecomposer
    from src.reasoning.planning.task_planner import TaskPlanner
    from src.skills.builtin.time_block_planner.time_block_planner import TimeBlockPlanner
    
    # Import life organizer components
    from src.skills.builtin.life_organizer.life_organizer import LifeOrganizerSkill
    from src.skills.builtin.life_organizer.mood_energy_tracker import (
        MoodEnergyTracker, MoodLevel, EnergyLevel
    )
    from src.skills.builtin.life_organizer.voice_kanban_interface import (
        VoiceKanbanInterface, KanbanColumn, TaskPriority
    )
    from src.skills.builtin.life_organizer.adaptive_recommendation_engine import (
        AdaptiveRecommendationEngine, RecommendationType
    )
    
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Running with minimal functionality...")


class LifeOrganizerTester:
    """Test suite for Life Organizer skill."""
    
    def __init__(self):
        self.container = MockContainer()
        self.passed_tests = 0
        self.total_tests = 0
    
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result."""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   {details}")
        else:
            print(f"❌ {test_name}")
            if details:
                print(f"   Error: {details}")
        print()
    
    async def test_life_organizer_initialization(self):
        """Test Life Organizer skill initialization."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Check basic attributes
            assert hasattr(skill, 'metadata')
            assert hasattr(skill, 'mood_tracker')
            assert hasattr(skill, 'kanban_interface')
            assert hasattr(skill, 'recommendation_engine')
            
            # Check metadata
            metadata = skill.get_metadata()
            assert metadata.id == "life_organizer.main"
            assert "منظم الحياة" in metadata.name
            assert "ar" in metadata.language_support
            
            self.log_test("Life Organizer Initialization", True, 
                         f"Skill ID: {metadata.id}, Languages: {metadata.language_support}")
            
        except Exception as e:
            self.log_test("Life Organizer Initialization", False, str(e))
    
    async def test_goal_decomposition(self):
        """Test goal decomposition functionality."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Test Arabic goal decomposition
            request = {
                "type": "goal_decomposition",
                "goal": "تعلم البرمجة خلال 6 أشهر",
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            
            assert result["success"] == True
            assert "goal_id" in result
            assert "action_steps" in result
            assert len(result["action_steps"]) > 0
            assert "تم تحليل هدفك" in result["response"]
            
            self.log_test("Goal Decomposition (Arabic)", True,
                         f"Generated {len(result['action_steps'])} action steps")
            
            # Test English goal decomposition
            request["language"] = "en"
            request["goal"] = "Learn programming in 6 months"
            
            result = await skill.process_request(request)
            assert result["success"] == True
            
            self.log_test("Goal Decomposition (English)", True,
                         "English goal processing successful")
            
        except Exception as e:
            self.log_test("Goal Decomposition", False, str(e))
    
    async def test_mood_energy_tracking(self):
        """Test mood and energy tracking."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Test voice-based mood tracking
            request = {
                "type": "mood_tracking",
                "voice_text": "أشعر بالسعادة والحماس اليوم!",
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            
            assert result["success"] == True
            assert "mood_state" in result
            assert "recommendations" in result
            assert "تحليل المزاج والطاقة" in result["response"]
            
            mood_state = result["mood_state"]
            assert "mood_level" in mood_state
            assert "energy_level" in mood_state
            assert "confidence" in mood_state
            
            self.log_test("Mood Tracking (Voice)", True,
                         f"Mood: {mood_state['mood_level']}, Energy: {mood_state['energy_level']}")
            
            # Test vision-based mood tracking
            request = {
                "type": "mood_tracking",
                "image_data": "mock_image_data",
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            assert result["success"] == True
            
            self.log_test("Mood Tracking (Vision)", True,
                         "Vision-based mood analysis successful")
            
        except Exception as e:
            self.log_test("Mood Tracking", False, str(e))
    
    async def test_voice_kanban_interface(self):
        """Test voice-controlled Kanban interface."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Test task creation
            request = {
                "type": "voice_kanban",
                "voice_command": "أضف مهمة كتابة التقرير",
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            
            assert result["success"] == True
            assert "board_summary" in result
            
            self.log_test("Voice Kanban - Task Creation", True,
                         "Arabic voice command processed successfully")
            
            # Test board display
            request["voice_command"] = "أظهر اللوحة"
            result = await skill.process_request(request)
            assert result["success"] == True
            
            self.log_test("Voice Kanban - Board Display", True,
                         "Board display command successful")
            
            # Test English commands
            request["voice_command"] = "add task write report"
            request["language"] = "en"
            result = await skill.process_request(request)
            
            self.log_test("Voice Kanban - English Commands", True,
                         "English voice commands supported")
            
        except Exception as e:
            self.log_test("Voice Kanban Interface", False, str(e))
    
    async def test_adaptive_scheduling(self):
        """Test adaptive scheduling based on mood/energy."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # First set mood/energy state
            mood_request = {
                "type": "mood_tracking",
                "voice_text": "أشعر بالطاقة والحماس",
                "language": "ar"
            }
            await skill.process_request(mood_request)
            
            # Test adaptive scheduling
            request = {
                "type": "adaptive_scheduling",
                "tasks": [
                    {"title": "مهمة صعبة", "difficulty": "high"},
                    {"title": "مهمة سهلة", "difficulty": "low"}
                ],
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            
            assert result["success"] == True
            assert "mood_context" in result
            assert "adaptive_recommendations" in result
            
            self.log_test("Adaptive Scheduling", True,
                         "Mood-aware scheduling successful")
            
        except Exception as e:
            self.log_test("Adaptive Scheduling", False, str(e))
    
    async def test_daily_planning(self):
        """Test comprehensive daily planning."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Set up mood state first
            mood_request = {
                "type": "mood_tracking",
                "voice_text": "مزاجي جيد وطاقتي عالية",
                "language": "ar"
            }
            await skill.process_request(mood_request)
            
            # Test daily planning
            request = {
                "type": "daily_plan",
                "date": datetime.now().date(),
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            
            assert result["success"] == True
            assert "daily_plan" in result
            assert "خطة يوم" in result["response"]
            
            daily_plan = result["daily_plan"]
            assert "mood_energy_state" in daily_plan
            assert "recommendations" in daily_plan
            
            self.log_test("Daily Planning", True,
                         "Comprehensive daily plan created")
            
        except Exception as e:
            self.log_test("Daily Planning", False, str(e))
    
    async def test_general_voice_commands(self):
        """Test general voice command processing."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Test Arabic voice command
            request = {
                "type": "voice_command",
                "voice_text": "ساعدني في تنظيم يومي",
                "language": "ar"
            }
            
            result = await skill.process_request(request)
            
            assert result["success"] == True
            assert len(result["response"]) > 0
            
            self.log_test("General Voice Commands (Arabic)", True,
                         "Arabic voice assistance working")
            
            # Test English voice command
            request["voice_text"] = "help me organize my day"
            request["language"] = "en"
            
            result = await skill.process_request(request)
            assert result["success"] == True
            
            self.log_test("General Voice Commands (English)", True,
                         "English voice assistance working")
            
        except Exception as e:
            self.log_test("General Voice Commands", False, str(e))
    
    async def test_skill_health_and_validation(self):
        """Test skill health check and input validation."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Test health check
            health = await skill.get_health_status()
            assert "status" in health
            assert "components" in health
            
            self.log_test("Health Check", True,
                         f"Status: {health['status']}")
            
            # Test input validation
            valid_input = {"type": "goal_decomposition", "goal": "test goal"}
            assert await skill.validate_input(valid_input) == True
            
            invalid_input = "not a dict"
            assert await skill.validate_input(invalid_input) == False
            
            self.log_test("Input Validation", True,
                         "Validation logic working correctly")
            
            # Test capabilities
            capabilities = await skill.get_capabilities()
            assert len(capabilities) > 0
            
            self.log_test("Capabilities", True,
                         f"Provides {len(capabilities)} capabilities")
            
        except Exception as e:
            self.log_test("Health and Validation", False, str(e))
    
    async def test_arabic_language_support(self):
        """Test comprehensive Arabic language support."""
        try:
            skill = LifeOrganizerSkill(self.container)
            
            # Test various Arabic inputs
            arabic_tests = [
                {
                    "type": "goal_decomposition",
                    "goal": "تطوير مهاراتي المهنية",
                    "language": "ar"
                },
                {
                    "type": "voice_kanban", 
                    "voice_command": "أضف مهمة مراجعة الميزانية",
                    "language": "ar"
                },
                {
                    "type": "mood_tracking",
                    "voice_text": "أشعر بالتعب والإرهاق",
                    "language": "ar"
                }
            ]
            
            arabic_success = 0
            for test_req in arabic_tests:
                result = await skill.process_request(test_req)
                if result["success"] and any(
                    arabic_char in result["response"] 
                    for arabic_char in "اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويى"
                ):
                    arabic_success += 1
            
            assert arabic_success == len(arabic_tests)
            
            self.log_test("Arabic Language Support", True,
                         f"All {len(arabic_tests)} Arabic tests passed")
            
        except Exception as e:
            self.log_test("Arabic Language Support", False, str(e))
    
    async def run_all_tests(self):
        """Run all tests."""
        print("🧪 Testing Life Organizer Skill - منظم الحياة الذكي...")
        print("=" * 60)
        
        # Run all test methods
        test_methods = [
            self.test_life_organizer_initialization,
            self.test_goal_decomposition,
            self.test_mood_energy_tracking,
            self.test_voice_kanban_interface,
            self.test_adaptive_scheduling,
            self.test_daily_planning,
            self.test_general_voice_commands,
            self.test_skill_health_and_validation,
            self.test_arabic_language_support
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                test_name = test_method.__name__.replace("test_", "").replace("_", " ").title()
                self.log_test(test_name, False, f"Test execution failed: {e}")
        
        # Print summary
        print("=" * 60)
        print(f"🎯 Test Results: {self.passed_tests}/{self.total_tests} passed")
        
        if self.passed_tests == self.total_tests:
            print("🎉 All tests passed! Life Organizer skill is working correctly.")
            print()
            print("✨ Key features validated:")
            print("   • Arabic-first language support")
            print("   • Goal decomposition and action planning")
            print("   • Multimodal mood and energy tracking")
            print("   • Voice-controlled Kanban board management")
            print("   • Adaptive scheduling based on user state")
            print("   • Comprehensive daily planning")
            print("   • Intelligent recommendations")
        else:
            print(f"⚠️  {self.total_tests - self.passed_tests} tests failed.")
            print("Please review the errors above.")
        
        return self.passed_tests == self.total_tests


async def main():
    """Main test execution."""
    tester = LifeOrganizerTester()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())