"""
Time-Block Efficiency Planner Skill
Author: Drmusab
Last Modified: 2025-01-20

A skill that optimizes daily routines by implementing time-blocking techniques,
intelligently allocating tasks into focused time slots while minimizing interruptions
and multitasking. Arabic is the first and primary language.
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.integrations.external_apis.calendar_api import CalendarAPI
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.speech.text_to_speech import TextToSpeechEngine
from src.skills.skill_registry import (
    SkillCapability,
    SkillInterface,
    SkillMetadata,
    SkillType,
)

from .task_classifier import TaskClassifier
from .schedule_optimizer import ScheduleOptimizer
from .disruption_handler import DisruptionHandler


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "منخفض"  # Arabic first
    MEDIUM = "متوسط"
    HIGH = "عالي"
    URGENT = "عاجل"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "بسيط"  # Arabic first
    MODERATE = "متوسط"
    COMPLEX = "معقد"
    VERY_COMPLEX = "معقد جداً"


class FocusType(Enum):
    """Types of focus required for tasks."""
    DEEP_WORK = "عمل عميق"  # Arabic first
    ADMIN = "إداري"
    CREATIVE = "إبداعي"
    COMMUNICATION = "تواصل"
    LEARNING = "تعلم"


@dataclass
class Task:
    """Represents a task to be scheduled."""
    id: str
    title_ar: str  # Arabic title (primary)
    title_en: str  # English title (secondary)
    description_ar: str = ""
    description_en: str = ""
    estimated_duration: int = 30  # minutes
    priority: TaskPriority = TaskPriority.MEDIUM
    complexity: TaskComplexity = TaskComplexity.MODERATE
    focus_type: FocusType = FocusType.ADMIN
    deadline: Optional[datetime] = None
    is_flexible: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TimeBlock:
    """Represents a time block in the schedule."""
    id: str
    start_time: datetime
    end_time: datetime
    task: Optional[Task] = None
    block_type: str = "work"  # work, break, buffer
    is_locked: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserPreferences:
    """User preferences for time blocking."""
    preferred_block_length: int = 90  # minutes
    max_deep_work_blocks: int = 3
    preferred_break_length: int = 15  # minutes
    work_start_time: str = "09:00"
    work_end_time: str = "17:00"
    focus_peak_hours: List[str] = field(default_factory=lambda: ["09:00-11:00", "14:00-16:00"])
    buffer_time_percentage: float = 0.1  # 10% buffer time
    language_preference: str = "ar"  # Arabic first


class TimeBlockPlannerSkill(SkillInterface):
    """
    Advanced time-block efficiency planner skill with Arabic-first language support.
    
    Features:
    - Intelligent task allocation into focused time blocks
    - Arabic-first language processing and responses
    - Integration with calendar systems for conflict detection
    - Real-time adjustments based on disruptions
    - Learning from user feedback and completion patterns
    - Text-to-speech reminders and notifications
    """

    def __init__(self):
        """Initialize the time block planner skill."""
        self.logger = get_logger(__name__)
        self.container: Optional[Container] = None
        self.config: Optional[ConfigLoader] = None
        self.event_bus: Optional[EventBus] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.bilingual_manager: Optional[BilingualManager] = None
        self.intent_manager: Optional[IntentManager] = None
        self.calendar_api: Optional[CalendarAPI] = None
        self.tts_engine: Optional[TextToSpeechEngine] = None
        
        # Skill components
        self.task_classifier: Optional[TaskClassifier] = None
        self.schedule_optimizer: Optional[ScheduleOptimizer] = None
        self.disruption_handler: Optional[DisruptionHandler] = None
        
        # State
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.active_schedules: Dict[str, List[TimeBlock]] = {}
        self.task_completion_history: Dict[str, List[Dict]] = {}
        
        # Performance tracking
        self.completion_rates: Dict[str, float] = {}
        self.focus_patterns: Dict[str, Dict] = {}
        
        self.logger.info("TimeBlockPlannerSkill initialized")

    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return SkillMetadata(
            skill_id="productivity.time_block_planner",
            name="مخطط الكتل الزمنية للكفاءة",  # Arabic first
            version="1.0.0",
            description="مهارة تحسين الروتين اليومي عبر تقنيات تقسيم الوقت الذكية",  # Arabic description
            author="Drmusab",
            skill_type=SkillType.BUILTIN,
            capabilities=[
                SkillCapability(
                    name="plan_workday",
                    description="تخطيط يوم العمل بكتل زمنية محسنة",  # Arabic first
                    input_types=["string", "dict"],
                    output_types=["dict"],
                    metadata={"language_primary": "ar", "language_secondary": "en"}
                ),
                SkillCapability(
                    name="adjust_schedule",
                    description="تعديل الجدولة في الوقت الفعلي",
                    input_types=["dict"],
                    output_types=["dict"]
                ),
                SkillCapability(
                    name="track_focus",
                    description="تتبع أنماط التركيز والأداء",
                    input_types=["dict"],
                    output_types=["dict"]
                ),
                SkillCapability(
                    name="provide_reminders",
                    description="تقديم تذكيرات صوتية للكتل الزمنية",
                    input_types=["string"],
                    output_types=["audio", "text"]
                )
            ],
            dependencies=["bilingual_manager", "intent_manager", "calendar_api", "text_to_speech"],
            tags=["productivity", "time-management", "arabic", "scheduling", "focus"],
            configuration_schema={
                "block_length": {"type": "integer", "default": 90, "min": 15, "max": 240},
                "language": {"type": "string", "default": "ar", "enum": ["ar", "en"]},
                "auto_adjust": {"type": "boolean", "default": True},
                "reminder_interval": {"type": "integer", "default": 5}
            }
        )

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the skill with dependencies."""
        try:
            # Initialize components
            self.task_classifier = TaskClassifier()
            self.schedule_optimizer = ScheduleOptimizer()
            self.disruption_handler = DisruptionHandler()
            
            # Initialize component configurations
            if self.task_classifier:
                await self.task_classifier.initialize(config)
            if self.schedule_optimizer:
                await self.schedule_optimizer.initialize(config)
            if self.disruption_handler:
                await self.disruption_handler.initialize(config)
            
            self.logger.info("TimeBlockPlannerSkill initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TimeBlockPlannerSkill: {str(e)}")
            raise

    async def set_dependencies(self, dependencies: Dict[str, Any]) -> None:
        """Set dependencies injected by the skill factory."""
        self.container = dependencies.get("container")
        self.config = dependencies.get("config")
        self.event_bus = dependencies.get("event_bus")
        self.memory_manager = dependencies.get("memory_manager")
        
        # Get processing components
        if self.container:
            self.bilingual_manager = self.container.get(BilingualManager)
            self.intent_manager = self.container.get(IntentManager) 
            self.calendar_api = self.container.get(CalendarAPI)
            self.tts_engine = self.container.get(TextToSpeechEngine)

    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute the time block planner skill."""
        try:
            # Parse input to determine action
            action = await self._parse_user_input(input_data, context)
            
            # Route to appropriate handler
            if action["type"] == "plan_workday":
                return await self._plan_workday(action["data"], context)
            elif action["type"] == "adjust_schedule":
                return await self._adjust_schedule(action["data"], context)
            elif action["type"] == "track_focus":
                return await self._track_focus_performance(action["data"], context)
            elif action["type"] == "provide_reminder":
                return await self._provide_reminder(action["data"], context)
            else:
                return await self._handle_general_query(input_data, context)
                
        except Exception as e:
            self.logger.error(f"Error executing TimeBlockPlannerSkill: {str(e)}")
            return {
                "success": False,
                "error": f"خطأ في تنفيذ مخطط الكتل الزمنية: {str(e)}",  # Arabic error message
                "error_en": f"Error executing time block planner: {str(e)}"
            }

    async def _parse_user_input(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse user input to determine intent and extract data."""
        if isinstance(input_data, str):
            # Use bilingual intent detection
            if self.intent_manager:
                intent_result = await self.intent_manager.detect_intent(
                    input_data, 
                    language="ar"  # Arabic first
                )
                
                # Map intents to actions
                if "plan" in intent_result.get("intent", "").lower() or "خطط" in input_data:
                    return {
                        "type": "plan_workday",
                        "data": {"tasks": input_data, "preferences": {}}
                    }
                elif "adjust" in intent_result.get("intent", "").lower() or "عدل" in input_data:
                    return {
                        "type": "adjust_schedule", 
                        "data": {"adjustment": input_data}
                    }
                elif "remind" in intent_result.get("intent", "").lower() or "ذكر" in input_data:
                    return {
                        "type": "provide_reminder",
                        "data": {"message": input_data}
                    }
            
            # Default to workday planning
            return {
                "type": "plan_workday",
                "data": {"tasks": input_data, "preferences": {}}
            }
            
        elif isinstance(input_data, dict):
            action_type = input_data.get("action", "plan_workday")
            return {
                "type": action_type,
                "data": input_data.get("data", {})
            }
        
        return {
            "type": "plan_workday", 
            "data": {"tasks": str(input_data), "preferences": {}}
        }

    async def _plan_workday(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan a workday with time blocks."""
        user_id = context.get("user_id", "default")
        
        try:
            # Get or create user preferences
            preferences = self._get_user_preferences(user_id)
            
            # Parse tasks from input
            tasks = await self._parse_tasks(data.get("tasks", ""), context)
            
            # Check calendar conflicts
            calendar_events = await self._get_calendar_conflicts(user_id, datetime.now().date())
            
            # Generate optimized schedule
            if self.schedule_optimizer:
                schedule = await self.schedule_optimizer.create_schedule(
                    tasks, preferences, calendar_events
                )
            else:
                schedule = await self._create_basic_schedule(tasks, preferences)
            
            # Store the schedule
            self.active_schedules[user_id] = schedule
            
            # Generate response in Arabic first
            response_text = await self._generate_schedule_response(schedule, preferences.language_preference)
            
            # Provide TTS reminder if enabled
            if self.tts_engine and preferences.language_preference == "ar":
                audio_response = await self.tts_engine.synthesize(
                    response_text["ar"],
                    language="ar"
                )
            else:
                audio_response = None
            
            return {
                "success": True,
                "schedule": [asdict(block) for block in schedule],
                "response": response_text,
                "audio": audio_response,
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error planning workday: {str(e)}")
            return {
                "success": False,
                "error": f"خطأ في تخطيط يوم العمل: {str(e)}",
                "error_en": f"Error planning workday: {str(e)}"
            }

    async def _parse_tasks(self, tasks_input: str, context: Dict[str, Any]) -> List[Task]:
        """Parse tasks from user input using bilingual processing."""
        tasks = []
        
        if self.task_classifier:
            parsed_tasks = await self.task_classifier.extract_tasks(tasks_input)
            tasks.extend(parsed_tasks)
        else:
            # Fallback simple parsing
            task_lines = tasks_input.split('\n')
            for i, line in enumerate(task_lines):
                if line.strip():
                    task = Task(
                        id=f"task_{i}_{uuid.uuid4().hex[:8]}",
                        title_ar=line.strip(),
                        title_en=line.strip(),  # Same for now, would translate in real implementation
                        estimated_duration=60  # Default 1 hour
                    )
                    tasks.append(task)
        
        return tasks

    async def _get_calendar_conflicts(self, user_id: str, date: datetime.date) -> List[Dict]:
        """Get calendar events that may conflict with scheduling."""
        conflicts = []
        
        if self.calendar_api:
            try:
                events = await self.calendar_api.get_events(
                    user_id, 
                    start_date=date,
                    end_date=date + timedelta(days=1)
                )
                conflicts = events
            except Exception as e:
                self.logger.warning(f"Could not fetch calendar events: {str(e)}")
        
        return conflicts

    async def _create_basic_schedule(self, tasks: List[Task], preferences: UserPreferences) -> List[TimeBlock]:
        """Create a basic schedule when optimizer is not available."""
        schedule = []
        current_time = datetime.now().replace(
            hour=int(preferences.work_start_time.split(':')[0]),
            minute=int(preferences.work_start_time.split(':')[1]),
            second=0,
            microsecond=0
        )
        
        for task in tasks:
            # Create time block for task
            end_time = current_time + timedelta(minutes=task.estimated_duration)
            
            block = TimeBlock(
                id=f"block_{uuid.uuid4().hex[:8]}",
                start_time=current_time,
                end_time=end_time,
                task=task,
                block_type="work"
            )
            schedule.append(block)
            
            # Add break if needed
            if task.focus_type == FocusType.DEEP_WORK:
                break_end = end_time + timedelta(minutes=preferences.preferred_break_length)
                break_block = TimeBlock(
                    id=f"break_{uuid.uuid4().hex[:8]}",
                    start_time=end_time,
                    end_time=break_end,
                    block_type="break"
                )
                schedule.append(break_block)
                current_time = break_end
            else:
                current_time = end_time + timedelta(minutes=5)  # Short buffer
        
        return schedule

    async def _generate_schedule_response(self, schedule: List[TimeBlock], language: str) -> Dict[str, str]:
        """Generate human-readable schedule response."""
        if language == "ar":
            response_ar = "إليك جدولك المحسن لليوم:\n\n"
            response_en = "Here's your optimized schedule for today:\n\n"
            
            for block in schedule:
                time_str = f"{block.start_time.strftime('%H:%M')}–{block.end_time.strftime('%H:%M')}"
                
                if block.task:
                    response_ar += f"⏰ {time_str}: {block.task.title_ar}\n"
                    response_en += f"⏰ {time_str}: {block.task.title_en}\n"
                else:
                    if block.block_type == "break":
                        response_ar += f"☕ {time_str}: استراحة\n"
                        response_en += f"☕ {time_str}: Break\n"
                    else:
                        response_ar += f"🔄 {time_str}: وقت مرن\n"
                        response_en += f"🔄 {time_str}: Buffer time\n"
            
            response_ar += "\n💡 سأذكرك قبل كل كتلة زمنية بـ 5 دقائق."
            response_en += "\n💡 I'll remind you 5 minutes before each time block."
            
        else:
            response_en = "Here's your optimized schedule for today:\n\n"
            response_ar = "إليك جدولك المحسن لليوم:\n\n"
            
            for block in schedule:
                time_str = f"{block.start_time.strftime('%H:%M')}–{block.end_time.strftime('%H:%M')}"
                
                if block.task:
                    response_en += f"⏰ {time_str}: {block.task.title_en}\n"
                    response_ar += f"⏰ {time_str}: {block.task.title_ar}\n"
                else:
                    if block.block_type == "break":
                        response_en += f"☕ {time_str}: Break\n"
                        response_ar += f"☕ {time_str}: استراحة\n"
                    else:
                        response_en += f"🔄 {time_str}: Buffer time\n"
                        response_ar += f"🔄 {time_str}: وقت مرن\n"
            
            response_en += "\n💡 I'll remind you 5 minutes before each time block."
            response_ar += "\n💡 سأذكرك قبل كل كتلة زمنية بـ 5 دقائق."
        
        return {"ar": response_ar, "en": response_en}

    async def _adjust_schedule(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust schedule based on disruptions or user feedback."""
        user_id = context.get("user_id", "default")
        
        if self.disruption_handler:
            adjusted_schedule = await self.disruption_handler.handle_disruption(
                self.active_schedules.get(user_id, []),
                data
            )
            self.active_schedules[user_id] = adjusted_schedule
            
            return {
                "success": True,
                "message": "تم تعديل الجدول بنجاح",  # Arabic first
                "message_en": "Schedule adjusted successfully",
                "updated_schedule": [asdict(block) for block in adjusted_schedule]
            }
        
        return {
            "success": False,
            "error": "خدمة تعديل الجدول غير متاحة حالياً",
            "error_en": "Schedule adjustment service not available"
        }

    async def _track_focus_performance(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Track and analyze focus performance patterns."""
        user_id = context.get("user_id", "default")
        
        # Store completion data
        if user_id not in self.task_completion_history:
            self.task_completion_history[user_id] = []
        
        self.task_completion_history[user_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        })
        
        # Calculate performance metrics
        recent_history = self.task_completion_history[user_id][-10:]  # Last 10 entries
        completion_rate = sum(1 for entry in recent_history if entry["data"].get("completed", False)) / len(recent_history) if recent_history else 0
        
        self.completion_rates[user_id] = completion_rate
        
        return {
            "success": True,
            "completion_rate": completion_rate,
            "message": f"معدل الإنجاز الحالي: {completion_rate:.1%}",  # Arabic first
            "message_en": f"Current completion rate: {completion_rate:.1%}",
            "insights": await self._generate_performance_insights(user_id)
        }

    async def _provide_reminder(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide TTS reminder for upcoming time blocks."""
        message = data.get("message", "حان وقت الكتلة الزمنية التالية")  # Arabic default
        
        if self.tts_engine:
            try:
                audio_response = await self.tts_engine.synthesize(message, language="ar")
                return {
                    "success": True,
                    "message": message,
                    "audio": audio_response
                }
            except Exception as e:
                self.logger.error(f"TTS error: {str(e)}")
        
        return {
            "success": True,
            "message": message,
            "audio": None,
            "note": "الصوت غير متاح حالياً"  # Arabic note
        }

    async def _handle_general_query(self, input_data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries about time blocking."""
        response = {
            "success": True,
            "message": "مرحباً! أنا مساعدك لتخطيط الكتل الزمنية. يمكنني مساعدتك في:\n"
                      "• تخطيط يومك بكتل زمنية محسنة\n"
                      "• تعديل الجدول عند الحاجة\n"
                      "• تتبع أداء التركيز\n"
                      "• إرسال تذكيرات صوتية\n\n"
                      "ما الذي تود تخطيطه اليوم؟",
            "message_en": "Hello! I'm your time-blocking assistant. I can help you with:\n"
                         "• Planning your day with optimized time blocks\n"
                         "• Adjusting schedules when needed\n"
                         "• Tracking focus performance\n"
                         "• Providing voice reminders\n\n"
                         "What would you like to plan today?",
            "capabilities": [cap.name for cap in self.get_metadata().capabilities]
        }
        
        return response

    async def _generate_performance_insights(self, user_id: str) -> Dict[str, str]:
        """Generate insights about user's performance patterns."""
        completion_rate = self.completion_rates.get(user_id, 0)
        
        if completion_rate > 0.8:
            insight_ar = "أداؤك ممتاز! تحافظ على معدل إنجاز عالي."
            insight_en = "Excellent performance! You maintain a high completion rate."
        elif completion_rate > 0.6:
            insight_ar = "أداء جيد. حاول تقليل المشتتات في الكتل الزمنية القادمة."
            insight_en = "Good performance. Try reducing distractions in upcoming time blocks."
        else:
            insight_ar = "يمكن تحسين الأداء. فكر في تقليل طول الكتل الزمنية أو تقليل المهام المعقدة."
            insight_en = "Performance can be improved. Consider shorter time blocks or reducing complex tasks."
        
        return {"ar": insight_ar, "en": insight_en}

    def _get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences or create defaults."""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences()
        return self.user_preferences[user_id]

    async def cleanup(self) -> None:
        """Cleanup skill resources."""
        # Save any pending data
        if self.memory_manager:
            for user_id, history in self.task_completion_history.items():
                await self.memory_manager.store_user_data(
                    user_id, 
                    "time_block_history", 
                    history
                )
        
        self.logger.info("TimeBlockPlannerSkill cleanup completed")

    async def validate(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, (str, dict)):
            return True
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Check skill health."""
        health_status = {
            "status": "healthy",
            "active_users": len(self.user_preferences),
            "active_schedules": len(self.active_schedules),
            "components": {
                "task_classifier": self.task_classifier is not None,
                "schedule_optimizer": self.schedule_optimizer is not None,
                "disruption_handler": self.disruption_handler is not None,
                "bilingual_manager": self.bilingual_manager is not None,
                "calendar_api": self.calendar_api is not None,
                "tts_engine": self.tts_engine is not None
            }
        }
        
        return health_status