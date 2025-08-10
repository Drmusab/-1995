"""
Life Organizer Skill - مهارة منظم الحياة
Author: Drmusab
Last Modified: 2025-01-20

A comprehensive life organizer skill that helps users break down goals into actionable steps,
provides mood-aware reminders and priorities, tracks energy levels through voice and vision,
and implements a voice-controlled Kanban-style planner.

Features:
- Goal decomposition into actionable steps
- Mood and energy tracking via voice and vision
- Voice-controlled Kanban board
- Adaptive recommendations based on current state
- Arabic-first language support
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
from src.processing.speech.text_to_speech import TextToSpeechEngine
from src.reasoning.planning.goal_decomposer import GoalDecomposer
from src.reasoning.planning.task_planner import TaskPlanner
from src.skills.skill_registry import (
    SkillCapability,
    SkillInterface,
    SkillMetadata,
    SkillType,
)
from src.skills.builtin.time_block_planner.time_block_planner import TimeBlockPlanner

from .mood_energy_tracker import MoodEnergyTracker, MoodEnergyState, MoodLevel, EnergyLevel
from .voice_kanban_interface import VoiceKanbanInterface, KanbanTask, KanbanColumn
from .adaptive_recommendation_engine import AdaptiveRecommendationEngine, AdaptiveRecommendation


class LifeOrganizerMode(Enum):
    """Operating modes for the life organizer."""
    GOAL_PLANNING = "تخطيط الأهداف"  # Goal Planning
    MOOD_TRACKING = "تتبع المزاج"  # Mood Tracking
    KANBAN_VOICE = "كانبان صوتي"  # Voice Kanban
    ADAPTIVE_SCHEDULING = "جدولة تكيفية"  # Adaptive Scheduling


@dataclass
class LifeGoal:
    """A life goal with decomposed action steps."""
    id: str
    title: str
    description: str
    category: str
    priority: str
    deadline: Optional[datetime] = None
    action_steps: List[Dict[str, Any]] = field(default_factory=list)
    progress: float = 0.0  # 0-1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DailyPlan:
    """Daily plan with mood-aware scheduling."""
    date: datetime
    mood_energy_state: Optional[MoodEnergyState]
    scheduled_tasks: List[Dict[str, Any]]
    recommendations: List[AdaptiveRecommendation]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LifeOrganizerSkill(SkillInterface):
    """Life Organizer Skill - منظم الحياة الذكي"""

    def __init__(self, container: Container):
        self.container = container
        self.config = container.resolve(ConfigLoader)
        self.event_bus = container.resolve(EventBus)
        self.logger = get_logger(__name__)
        
        # Language and communication
        self.bilingual_manager = container.resolve(BilingualManager)
        self.intent_manager = container.resolve(IntentManager)
        self.tts_engine = container.resolve(TextToSpeechEngine)
        
        # Planning and memory
        self.goal_decomposer = container.resolve(GoalDecomposer)
        self.task_planner = container.resolve(TaskPlanner)
        self.time_block_planner = container.resolve(TimeBlockPlanner)
        self.memory_manager = container.resolve(MemoryManager)
        
        # Life organizer components
        self.mood_tracker = MoodEnergyTracker(container)
        self.kanban_interface = VoiceKanbanInterface(container)
        self.recommendation_engine = AdaptiveRecommendationEngine(container)
        
        # State management
        self.current_mode = LifeOrganizerMode.GOAL_PLANNING
        self.life_goals: Dict[str, LifeGoal] = {}
        self.daily_plans: Dict[str, DailyPlan] = {}
        self.current_session_state: Dict[str, Any] = {}
        
        # Initialize skill metadata
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Initialize skill metadata."""
        self.metadata = SkillMetadata(
            id="life_organizer.main",
            name="منظم الحياة الذكي",  # Smart Life Organizer
            description="منظم شامل للحياة يساعد في تحليل الأهداف وتتبع المزاج والطاقة وإدارة المهام بالصوت",
            version="1.0.0",
            author="Drmusab",
            skill_type=SkillType.PRODUCTIVITY,
            capabilities=[
                SkillCapability.GOAL_DECOMPOSITION,
                SkillCapability.MOOD_TRACKING,
                SkillCapability.VOICE_CONTROL,
                SkillCapability.ADAPTIVE_SCHEDULING,
                SkillCapability.MULTILINGUAL
            ],
            language_support=["ar", "en"],
            dependencies=["time_block_planner", "sentiment_analyzer", "expression_analyzer"],
            tags=["productivity", "planning", "mood", "voice", "arabic"]
        )

    @handle_exceptions()
    async def process_request(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process life organizer request."""
        
        request_type = request.get("type", "general")
        language = request.get("language", "ar")
        user_input = request.get("input", "")
        
        self.logger.info(f"Processing life organizer request: {request_type}")
        
        try:
            # Route based on request type
            if request_type == "goal_decomposition":
                return await self._handle_goal_decomposition(request, language)
            elif request_type == "mood_tracking":
                return await self._handle_mood_tracking(request, language, context)
            elif request_type == "voice_kanban":
                return await self._handle_voice_kanban(request, language)
            elif request_type == "adaptive_scheduling":
                return await self._handle_adaptive_scheduling(request, language)
            elif request_type == "daily_plan":
                return await self._handle_daily_planning(request, language)
            elif request_type == "voice_command":
                return await self._handle_voice_command(request, language, context)
            else:
                # General request - determine intent
                return await self._handle_general_request(request, language)
                
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            return self._create_error_response(str(e), language)

    async def _handle_goal_decomposition(self, request: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Handle goal decomposition request."""
        goal_text = request.get("goal", "")
        
        if not goal_text:
            return self._create_error_response("لم يتم تقديم هدف للتحليل" if language == "ar" else "No goal provided for analysis", language)
        
        # Decompose goal using goal decomposer
        decomposition_result = await self.goal_decomposer.decompose_goal(
            goal_text=goal_text,
            language=language,
            context=request.get("context", {})
        )
        
        # Create life goal object
        goal = LifeGoal(
            id=str(uuid.uuid4()),
            title=goal_text,
            description=decomposition_result.get("description", ""),
            category=decomposition_result.get("category", "شخصي"),
            priority=decomposition_result.get("priority", "متوسط"),
            action_steps=decomposition_result.get("action_steps", [])
        )
        
        # Store goal
        self.life_goals[goal.id] = goal
        
        # Store in memory
        await self.memory_manager.store(
            key=f"life_goal_{goal.id}",
            data=asdict(goal),
            tags=["life_organizer", "goal", language]
        )
        
        # Create response
        if language == "ar":
            response_text = f"""تم تحليل هدفك '{goal.title}' بنجاح!

📋 خطوات العمل المقترحة:
"""
            for i, step in enumerate(goal.action_steps, 1):
                response_text += f"{i}. {step.get('title', '')}\n"
                response_text += f"   ⏱️ المدة المقدرة: {step.get('estimated_duration', 'غير محدد')}\n"
                response_text += f"   🎯 الأولوية: {step.get('priority', 'متوسط')}\n\n"
            
            response_text += "\n💡 هل تريد إضافة هذه المهام إلى لوحة كانبان الصوتية؟"
        else:
            response_text = f"""Successfully analyzed your goal '{goal.title}'!

📋 Suggested action steps:
"""
            for i, step in enumerate(goal.action_steps, 1):
                response_text += f"{i}. {step.get('title', '')}\n"
                response_text += f"   ⏱️ Estimated duration: {step.get('estimated_duration', 'Unknown')}\n"
                response_text += f"   🎯 Priority: {step.get('priority', 'Medium')}\n\n"
            
            response_text += "\n💡 Would you like to add these tasks to the voice Kanban board?"
        
        return {
            "success": True,
            "response": response_text,
            "goal_id": goal.id,
            "action_steps": goal.action_steps,
            "suggested_next_action": "add_to_kanban"
        }

    async def _handle_mood_tracking(self, request: Dict[str, Any], language: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle mood and energy tracking request."""
        
        voice_text = request.get("voice_text")
        image_data = request.get("image_data")
        manual_mood = request.get("manual_mood")
        manual_energy = request.get("manual_energy")
        
        # Extract user_id from context if available
        user_id = None
        if context:
            user_id = context.get("user_id") or context.get("session", {}).get("user_id")
        
        # Update mood/energy state
        state = await self.mood_tracker.update_state(
            voice_text=voice_text,
            image_data=image_data,
            manual_mood=MoodLevel(manual_mood) if manual_mood else None,
            manual_energy=EnergyLevel(manual_energy) if manual_energy else None,
            user_id=user_id
        )
        
        # Get recommendations
        recommendations = self.mood_tracker.get_recommendations()
        
        # Generate adaptive recommendations
        adaptive_recs = await self.recommendation_engine.generate_recommendations(
            current_state=state,
            current_tasks=request.get("current_tasks", []),
            context=request.get("context", {})
        )
        
        # Create response
        if language == "ar":
            response_text = f"""📊 تحليل المزاج والطاقة:

🎭 المزاج الحالي: {state.mood_level.value}
⚡ مستوى الطاقة: {state.energy_level.value}
🎯 دقة التحليل: {state.confidence:.0%}
📡 المصادر: {', '.join(state.sources)}

💡 التوصيات المناسبة لحالتك:
"""
            for i, rec in enumerate(recommendations[:3], 1):
                response_text += f"{i}. {rec.arabic_description}\n"
                response_text += f"   ⏱️ المدة المثلى: {rec.optimal_duration} دقيقة\n"
                response_text += f"   📝 السبب: {rec.reason_arabic}\n\n"
            
            if adaptive_recs:
                response_text += "\n🧠 نصائح ذكية إضافية:\n"
                for i, rec in enumerate(adaptive_recs[:2], 1):
                    response_text += f"{i}. {rec.title_arabic}: {rec.description_arabic}\n"
        else:
            response_text = f"""📊 Mood and Energy Analysis:

🎭 Current Mood: {state.mood_level.value}
⚡ Energy Level: {state.energy_level.value}
🎯 Analysis Accuracy: {state.confidence:.0%}
📡 Sources: {', '.join(state.sources)}

💡 Recommendations for your current state:
"""
            for i, rec in enumerate(recommendations[:3], 1):
                response_text += f"{i}. {rec.english_description}\n"
                response_text += f"   ⏱️ Optimal duration: {rec.optimal_duration} minutes\n"
                response_text += f"   📝 Reason: {rec.reason_english}\n\n"
            
            if adaptive_recs:
                response_text += "\n🧠 Additional smart tips:\n"
                for i, rec in enumerate(adaptive_recs[:2], 1):
                    response_text += f"{i}. {rec.title_english}: {rec.description_english}\n"
        
        return {
            "success": True,
            "response": response_text,
            "mood_state": asdict(state),
            "recommendations": [asdict(r) for r in recommendations],
            "adaptive_recommendations": [asdict(r) for r in adaptive_recs]
        }

    async def _handle_voice_kanban(self, request: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Handle voice Kanban command."""
        voice_command = request.get("voice_command", "")
        
        if not voice_command:
            return self._create_error_response("لم يتم تقديم أمر صوتي" if language == "ar" else "No voice command provided", language)
        
        # Process voice command through Kanban interface
        response_text = await self.kanban_interface.process_voice_command(voice_command, language)
        
        # Get board summary for additional context
        board_summary = self.kanban_interface.get_board_summary()
        
        return {
            "success": True,
            "response": response_text,
            "board_summary": board_summary,
            "suggested_next_action": "voice_command"
        }

    async def _handle_adaptive_scheduling(self, request: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Handle adaptive scheduling based on mood and energy."""
        
        # Get current mood/energy state
        current_state = self.mood_tracker.get_current_state()
        
        if not current_state:
            # Need to track mood first
            if language == "ar":
                return {
                    "success": False,
                    "response": "يجب تتبع مزاجك وطاقتك أولاً لإنشاء جدول تكيفي",
                    "suggested_next_action": "mood_tracking"
                }
            else:
                return {
                    "success": False,
                    "response": "Need to track your mood and energy first to create adaptive schedule",
                    "suggested_next_action": "mood_tracking"
                }
        
        # Get tasks from request or Kanban board
        tasks = request.get("tasks", [])
        if not tasks:
            # Get tasks from Kanban board
            board_summary = self.kanban_interface.get_board_summary()
            tasks = self._extract_tasks_from_board()
        
        # Generate adaptive recommendations
        recommendations = await self.recommendation_engine.generate_recommendations(
            current_state=current_state,
            current_tasks=tasks,
            context=request.get("context", {})
        )
        
        # Use time block planner with mood awareness
        schedule_request = {
            "tasks": tasks,
            "language": language,
            "user_preferences": {
                "mood_level": current_state.mood_level.value,
                "energy_level": current_state.energy_level.value,
                "recommendations": [asdict(r) for r in recommendations]
            }
        }
        
        schedule_result = await self.time_block_planner.process_request(schedule_request)
        
        # Enhance schedule with mood-aware recommendations
        enhanced_response = self._enhance_schedule_with_mood_awareness(
            schedule_result, current_state, recommendations, language
        )
        
        return enhanced_response

    async def _handle_daily_planning(self, request: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Handle comprehensive daily planning."""
        
        target_date = request.get("date", datetime.now().date())
        
        # Get current mood/energy state
        current_state = self.mood_tracker.get_current_state()
        
        # Get tasks and goals
        tasks = self._extract_tasks_from_board()
        goals = list(self.life_goals.values())
        
        # Generate daily plan
        daily_plan = DailyPlan(
            date=target_date,
            mood_energy_state=current_state,
            scheduled_tasks=[],
            recommendations=[]
        )
        
        if current_state:
            # Get adaptive recommendations
            recommendations = await self.recommendation_engine.generate_recommendations(
                current_state=current_state,
                current_tasks=tasks,
                context={"planning_date": target_date}
            )
            daily_plan.recommendations = recommendations
            
            # Create mood-aware schedule
            schedule_request = {
                "tasks": tasks,
                "language": language,
                "user_preferences": {
                    "mood_level": current_state.mood_level.value,
                    "energy_level": current_state.energy_level.value
                }
            }
            
            schedule_result = await self.time_block_planner.process_request(schedule_request)
            daily_plan.scheduled_tasks = schedule_result.get("time_blocks", [])
        
        # Store daily plan
        date_key = target_date.strftime("%Y-%m-%d")
        self.daily_plans[date_key] = daily_plan
        
        # Create response
        response_text = self._create_daily_plan_response(daily_plan, language)
        
        return {
            "success": True,
            "response": response_text,
            "daily_plan": asdict(daily_plan),
            "suggested_next_action": "start_day"
        }

    async def _handle_voice_command(self, request: Dict[str, Any], language: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle general voice command."""
        voice_text = request.get("voice_text", "")
        
        # Analyze intent
        intent_result = await self.intent_manager.analyze_intent(voice_text, language)
        intent = intent_result.get("intent", "unknown")
        
        # Route based on detected intent
        if intent in ["kanban", "task_management"]:
            return await self._handle_voice_kanban({"voice_command": voice_text}, language)
        elif intent in ["mood", "energy", "feeling"]:
            return await self._handle_mood_tracking({"voice_text": voice_text}, language, context)
        elif intent in ["goal", "planning"]:
            return await self._handle_goal_decomposition({"goal": voice_text}, language)
        elif intent in ["schedule", "time_management"]:
            return await self._handle_adaptive_scheduling({"context": {"voice_input": voice_text}}, language)
        else:
            # General life organizer assistance
            return await self._handle_general_assistance(voice_text, language)

    async def _handle_general_request(self, request: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Handle general life organizer request."""
        user_input = request.get("input", "")
        
        # Provide overview of capabilities
        if language == "ar":
            response_text = """🌟 مرحباً بك في منظم الحياة الذكي!

يمكنني مساعدتك في:

🎯 تحليل الأهداف وتقسيمها لخطوات عملية
📊 تتبع مزاجك وطاقتك عبر الصوت والرؤية
🗣️ إدارة مهامك بواسطة لوحة كانبان الصوتية
📅 إنشاء جداول ذكية تتكيف مع حالتك
💡 تقديم نصائح مخصصة لتحسين إنتاجيتك

💬 جرب قول أشياء مثل:
• "أريد تحليل هدفي: تعلم البرمجة"
• "أظهر مزاجي وطاقتي الحالية"
• "أضف مهمة: كتابة التقرير"
• "أظهر اللوحة"
• "اعمل لي جدول لليوم"

كيف يمكنني مساعدتك اليوم؟"""
        else:
            response_text = """🌟 Welcome to the Smart Life Organizer!

I can help you with:

🎯 Analyze goals and break them into actionable steps
📊 Track your mood and energy through voice and vision
🗣️ Manage tasks with voice-controlled Kanban board
📅 Create smart schedules that adapt to your state
💡 Provide personalized tips to improve productivity

💬 Try saying things like:
• "I want to analyze my goal: learn programming"
• "Show my current mood and energy"
• "Add task: write report"
• "Show board"
• "Create schedule for today"

How can I help you today?"""
        
        return {
            "success": True,
            "response": response_text,
            "suggested_actions": ["goal_decomposition", "mood_tracking", "voice_kanban", "daily_plan"]
        }

    async def _handle_general_assistance(self, voice_text: str, language: str) -> Dict[str, Any]:
        """Handle general assistance request."""
        # This would integrate with the main LLM to provide contextual help
        if language == "ar":
            response = f"فهمت أنك تقول: '{voice_text}'. يمكنني مساعدتك في تنظيم حياتك. هل تريد تحليل هدف أم إدارة مهام أم تتبع مزاجك؟"
        else:
            response = f"I understand you said: '{voice_text}'. I can help organize your life. Would you like to analyze a goal, manage tasks, or track your mood?"
        
        return {
            "success": True,
            "response": response,
            "suggested_actions": ["goal_decomposition", "mood_tracking", "voice_kanban"]
        }

    def _extract_tasks_from_board(self) -> List[Dict[str, Any]]:
        """Extract tasks from Kanban board."""
        board_summary = self.kanban_interface.get_board_summary()
        # This would extract actual task objects from the Kanban interface
        # For now, return empty list - would be implemented based on actual board structure
        return []

    def _enhance_schedule_with_mood_awareness(
        self,
        schedule_result: Dict[str, Any],
        mood_state: MoodEnergyState,
        recommendations: List[AdaptiveRecommendation],
        language: str
    ) -> Dict[str, Any]:
        """Enhance schedule result with mood-aware recommendations."""
        
        enhanced_response = schedule_result.copy()
        
        # Add mood context to response
        if language == "ar":
            mood_context = f"""

🎭 تم إنشاء الجدول بناءً على حالتك الحالية:
• المزاج: {mood_state.mood_level.value}
• الطاقة: {mood_state.energy_level.value}

💡 نصائح إضافية:"""
            for rec in recommendations[:2]:
                mood_context += f"\n• {rec.title_arabic}: {rec.description_arabic}"
        else:
            mood_context = f"""

🎭 Schedule created based on your current state:
• Mood: {mood_state.mood_level.value}
• Energy: {mood_state.energy_level.value}

💡 Additional tips:"""
            for rec in recommendations[:2]:
                mood_context += f"\n• {rec.title_english}: {rec.description_english}"
        
        enhanced_response["response"] = enhanced_response.get("response", "") + mood_context
        enhanced_response["mood_context"] = asdict(mood_state)
        enhanced_response["adaptive_recommendations"] = [asdict(r) for r in recommendations]
        
        return enhanced_response

    def _create_daily_plan_response(self, daily_plan: DailyPlan, language: str) -> str:
        """Create response text for daily plan."""
        if language == "ar":
            response = f"""📅 خطة يوم {daily_plan.date.strftime('%Y-%m-%d')}

"""
            if daily_plan.mood_energy_state:
                response += f"🎭 حالتك: {daily_plan.mood_energy_state.mood_level.value} | {daily_plan.mood_energy_state.energy_level.value}\n\n"
            
            response += "📋 المهام المجدولة:\n"
            for i, task in enumerate(daily_plan.scheduled_tasks, 1):
                response += f"{i}. {task.get('title', 'مهمة')}\n"
            
            if daily_plan.recommendations:
                response += "\n💡 توصيات اليوم:\n"
                for i, rec in enumerate(daily_plan.recommendations[:3], 1):
                    response += f"{i}. {rec.title_arabic}\n"
        else:
            response = f"""📅 Daily Plan for {daily_plan.date.strftime('%Y-%m-%d')}

"""
            if daily_plan.mood_energy_state:
                response += f"🎭 Your state: {daily_plan.mood_energy_state.mood_level.value} | {daily_plan.mood_energy_state.energy_level.value}\n\n"
            
            response += "📋 Scheduled tasks:\n"
            for i, task in enumerate(daily_plan.scheduled_tasks, 1):
                response += f"{i}. {task.get('title', 'Task')}\n"
            
            if daily_plan.recommendations:
                response += "\n💡 Today's recommendations:\n"
                for i, rec in enumerate(daily_plan.recommendations[:3], 1):
                    response += f"{i}. {rec.title_english}\n"
        
        return response

    def _create_error_response(self, error_message: str, language: str) -> Dict[str, Any]:
        """Create error response."""
        if language == "ar":
            response = f"عذراً، حدث خطأ: {error_message}"
        else:
            response = f"Sorry, an error occurred: {error_message}"
        
        return {
            "success": False,
            "response": response,
            "error": error_message
        }

    async def get_capabilities(self) -> List[SkillCapability]:
        """Get skill capabilities."""
        return self.metadata.capabilities

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        # Basic validation
        if not isinstance(input_data, dict):
            return False
        
        # Must have either type or input
        return "type" in input_data or "input" in input_data

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "status": "healthy",
            "components": {
                "mood_tracker": "active",
                "kanban_interface": "active", 
                "recommendation_engine": "active"
            },
            "goals_count": len(self.life_goals),
            "daily_plans_count": len(self.daily_plans)
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.mood_tracker.cleanup()
        await self.kanban_interface.cleanup()
        await self.recommendation_engine.cleanup()
        self.logger.info("LifeOrganizerSkill cleanup completed")

    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return self.metadata