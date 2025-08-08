"""
Voice-Controlled Kanban Interface - واجهة كانبان الصوتية
Author: Drmusab
Last Modified: 2025-01-20

Implements a personal Kanban-style planner that can be controlled entirely through voice commands.
Supports Arabic voice commands and provides Arabic responses.
"""

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.observability.logging.config import get_logger
from src.processing.natural_language.bilingual_manager import BilingualManager
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.speech.text_to_speech import TextToSpeechEngine


class KanbanColumn(Enum):
    """Kanban board columns in Arabic."""
    BACKLOG = "قائمة المهام"  # Backlog
    TODO = "للقيام"  # To Do
    IN_PROGRESS = "قيد التنفيذ"  # In Progress
    REVIEW = "مراجعة"  # Review
    DONE = "مكتمل"  # Done


class TaskPriority(Enum):
    """Task priority levels in Arabic."""
    LOW = "منخفض"  # Low
    MEDIUM = "متوسط"  # Medium
    HIGH = "عالي"  # High
    URGENT = "عاجل"  # Urgent


@dataclass
class KanbanTask:
    """A task in the Kanban board."""
    id: str
    title: str
    description: str
    column: KanbanColumn
    priority: TaskPriority
    tags: List[str] = field(default_factory=list)
    estimated_minutes: Optional[int] = None
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: Optional[datetime] = None
    notes: str = ""


@dataclass
class KanbanBoard:
    """Kanban board with columns and tasks."""
    id: str
    name: str
    tasks: Dict[str, KanbanTask] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VoiceKanbanInterface:
    """Voice-controlled Kanban interface with Arabic support."""

    def __init__(self, container: Container):
        self.container = container
        self.config = container.resolve(ConfigLoader)
        self.event_bus = container.resolve(EventBus)
        self.logger = get_logger(__name__)
        
        # Language and speech processing
        self.bilingual_manager = container.resolve(BilingualManager)
        self.intent_manager = container.resolve(IntentManager)
        self.tts_engine = container.resolve(TextToSpeechEngine)
        
        # Kanban state
        self.boards: Dict[str, KanbanBoard] = {}
        self.current_board_id: Optional[str] = None
        
        # Initialize default board
        self._create_default_board()
        
        # Voice command patterns
        self._initialize_voice_patterns()

    def _create_default_board(self) -> None:
        """Create default personal Kanban board."""
        default_board = KanbanBoard(
            id="personal",
            name="اللوحة الشخصية"  # Personal Board
        )
        self.boards["personal"] = default_board
        self.current_board_id = "personal"

    def _initialize_voice_patterns(self) -> None:
        """Initialize voice command patterns for Arabic and English."""
        self.voice_patterns = {
            # Task creation commands
            "create_task": [
                # Arabic patterns
                r"أضف مهمة (.+)",
                r"إنشاء مهمة (.+)",
                r"مهمة جديدة (.+)",
                r"اكتب مهمة (.+)",
                # English patterns  
                r"add task (.+)",
                r"create task (.+)",
                r"new task (.+)",
                r"add (.+) to (.+)"
            ],
            
            # Move task commands
            "move_task": [
                # Arabic patterns
                r"انقل (.+) إلى (.+)",
                r"حرك (.+) إلى (.+)",
                r"ضع (.+) في (.+)",
                # English patterns
                r"move (.+) to (.+)",
                r"put (.+) in (.+)"
            ],
            
            # Status queries
            "show_board": [
                # Arabic patterns
                r"أظهر اللوحة",
                r"اعرض المهام",
                r"ما هي المهام",
                r"كيف حال المشاريع",
                # English patterns
                r"show board",
                r"show tasks",
                r"what are my tasks",
                r"board status"
            ],
            
            # Priority setting
            "set_priority": [
                # Arabic patterns
                r"اجعل (.+) (عاجل|عالي|متوسط|منخفض)",
                r"أولوية (.+) (عاجل|عالي|متوسط|منخفض)",
                # English patterns
                r"make (.+) (urgent|high|medium|low) priority",
                r"set (.+) priority to (urgent|high|medium|low)"
            ],
            
            # Complete task
            "complete_task": [
                # Arabic patterns
                r"أكملت (.+)",
                r"انتهيت من (.+)",
                r"(.+) مكتمل",
                # English patterns
                r"completed (.+)",
                r"finished (.+)",
                r"done with (.+)"
            ]
        }

    @handle_exceptions()
    async def process_voice_command(self, voice_text: str, language: str = "ar") -> str:
        """Process voice command and return response."""
        try:
            # Detect intent and extract entities
            intent_result = await self.intent_manager.analyze_intent(
                text=voice_text,
                language=language
            )
            
            intent = intent_result.get("intent", "unknown")
            entities = intent_result.get("entities", {})
            confidence = intent_result.get("confidence", 0.0)
            
            self.logger.info(f"Voice command - Intent: {intent}, Confidence: {confidence}")
            
            # Route to appropriate handler
            if intent == "create_task" or self._matches_pattern(voice_text, "create_task"):
                return await self._handle_create_task(voice_text, entities, language)
            elif intent == "move_task" or self._matches_pattern(voice_text, "move_task"):
                return await self._handle_move_task(voice_text, entities, language)
            elif intent == "show_board" or self._matches_pattern(voice_text, "show_board"):
                return await self._handle_show_board(language)
            elif intent == "set_priority" or self._matches_pattern(voice_text, "set_priority"):
                return await self._handle_set_priority(voice_text, entities, language)
            elif intent == "complete_task" or self._matches_pattern(voice_text, "complete_task"):
                return await self._handle_complete_task(voice_text, entities, language)
            else:
                return await self._handle_unknown_command(voice_text, language)
                
        except Exception as e:
            self.logger.error(f"Error processing voice command: {e}")
            if language == "ar":
                return "عذراً، حدث خطأ في معالجة الأمر الصوتي"
            else:
                return "Sorry, there was an error processing the voice command"

    def _matches_pattern(self, text: str, command_type: str) -> bool:
        """Check if text matches any pattern for the command type."""
        import re
        patterns = self.voice_patterns.get(command_type, [])
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    async def _handle_create_task(self, voice_text: str, entities: Dict, language: str) -> str:
        """Handle task creation command."""
        # Extract task details from voice text
        task_title = self._extract_task_title(voice_text, language)
        column = self._extract_column(voice_text, language)
        priority = self._extract_priority(voice_text, language)
        
        if not task_title:
            if language == "ar":
                return "لم أتمكن من فهم عنوان المهمة. يرجى المحاولة مرة أخرى"
            else:
                return "I couldn't understand the task title. Please try again"
        
        # Create new task
        task = KanbanTask(
            id=str(uuid.uuid4()),
            title=task_title,
            description="",
            column=column,
            priority=priority
        )
        
        # Add to current board
        current_board = self.boards[self.current_board_id]
        current_board.tasks[task.id] = task
        current_board.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Created task: {task_title}")
        
        if language == "ar":
            return f"تم إنشاء المهمة '{task_title}' في عمود '{column.value}' بأولوية '{priority.value}'"
        else:
            return f"Created task '{task_title}' in '{column.value}' column with '{priority.value}' priority"

    async def _handle_move_task(self, voice_text: str, entities: Dict, language: str) -> str:
        """Handle task movement command."""
        task_name = self._extract_task_name(voice_text, language)
        target_column = self._extract_target_column(voice_text, language)
        
        if not task_name or not target_column:
            if language == "ar":
                return "لم أتمكن من فهم المهمة أو العمود المستهدف"
            else:
                return "I couldn't understand the task or target column"
        
        # Find task by name
        task = self._find_task_by_name(task_name)
        if not task:
            if language == "ar":
                return f"لم أجد مهمة بالاسم '{task_name}'"
            else:
                return f"Couldn't find task named '{task_name}'"
        
        # Move task
        old_column = task.column.value
        task.column = target_column
        task.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Moved task '{task_name}' from '{old_column}' to '{target_column.value}'")
        
        if language == "ar":
            return f"تم نقل '{task_name}' من '{old_column}' إلى '{target_column.value}'"
        else:
            return f"Moved '{task_name}' from '{old_column}' to '{target_column.value}'"

    async def _handle_show_board(self, language: str) -> str:
        """Handle board display command."""
        current_board = self.boards[self.current_board_id]
        
        if not current_board.tasks:
            if language == "ar":
                return "لا توجد مهام في اللوحة حالياً"
            else:
                return "No tasks on the board currently"
        
        # Group tasks by column
        columns = {}
        for task in current_board.tasks.values():
            if task.column not in columns:
                columns[task.column] = []
            columns[task.column].append(task)
        
        # Build response
        if language == "ar":
            response = f"لوحة '{current_board.name}':\n\n"
            for column in KanbanColumn:
                tasks_in_column = columns.get(column, [])
                response += f"📋 {column.value} ({len(tasks_in_column)} مهام):\n"
                for task in tasks_in_column:
                    priority_icon = self._get_priority_icon(task.priority)
                    response += f"  {priority_icon} {task.title}\n"
                response += "\n"
        else:
            response = f"Board '{current_board.name}':\n\n"
            for column in KanbanColumn:
                tasks_in_column = columns.get(column, [])
                response += f"📋 {column.value} ({len(tasks_in_column)} tasks):\n"
                for task in tasks_in_column:
                    priority_icon = self._get_priority_icon(task.priority)
                    response += f"  {priority_icon} {task.title}\n"
                response += "\n"
        
        return response.strip()

    async def _handle_set_priority(self, voice_text: str, entities: Dict, language: str) -> str:
        """Handle priority setting command."""
        task_name = self._extract_task_name(voice_text, language)
        priority = self._extract_priority(voice_text, language)
        
        if not task_name:
            if language == "ar":
                return "لم أتمكن من فهم اسم المهمة"
            else:
                return "I couldn't understand the task name"
        
        task = self._find_task_by_name(task_name)
        if not task:
            if language == "ar":
                return f"لم أجد مهمة بالاسم '{task_name}'"
            else:
                return f"Couldn't find task named '{task_name}'"
        
        old_priority = task.priority.value
        task.priority = priority
        task.updated_at = datetime.now(timezone.utc)
        
        if language == "ar":
            return f"تم تغيير أولوية '{task_name}' من '{old_priority}' إلى '{priority.value}'"
        else:
            return f"Changed priority of '{task_name}' from '{old_priority}' to '{priority.value}'"

    async def _handle_complete_task(self, voice_text: str, entities: Dict, language: str) -> str:
        """Handle task completion command."""
        task_name = self._extract_task_name(voice_text, language)
        
        if not task_name:
            if language == "ar":
                return "لم أتمكن من فهم اسم المهمة"
            else:
                return "I couldn't understand the task name"
        
        task = self._find_task_by_name(task_name)
        if not task:
            if language == "ar":
                return f"لم أجد مهمة بالاسم '{task_name}'"
            else:
                return f"Couldn't find task named '{task_name}'"
        
        # Move to done column
        task.column = KanbanColumn.DONE
        task.updated_at = datetime.now(timezone.utc)
        
        if language == "ar":
            return f"ممتاز! تم تحريك '{task_name}' إلى عمود المكتمل 🎉"
        else:
            return f"Great! Moved '{task_name}' to Done column 🎉"

    async def _handle_unknown_command(self, voice_text: str, language: str) -> str:
        """Handle unknown commands."""
        if language == "ar":
            return """لم أتمكن من فهم الأمر. يمكنك قول:
• "أضف مهمة [اسم المهمة]"
• "انقل [المهمة] إلى [العمود]"
• "أظهر اللوحة"
• "اجعل [المهمة] عاجل"
• "أكملت [المهمة]" """
        else:
            return """I didn't understand that command. You can say:
• "Add task [task name]"
• "Move [task] to [column]"
• "Show board"
• "Make [task] urgent"
• "Completed [task]" """

    def _extract_task_title(self, text: str, language: str) -> Optional[str]:
        """Extract task title from voice text."""
        import re
        
        if language == "ar":
            patterns = [
                r"أضف مهمة (.+)",
                r"إنشاء مهمة (.+)",
                r"مهمة جديدة (.+)",
                r"اكتب مهمة (.+)"
            ]
        else:
            patterns = [
                r"add task (.+)",
                r"create task (.+)", 
                r"new task (.+)"
            ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_task_name(self, text: str, language: str) -> Optional[str]:
        """Extract task name from voice text."""
        import re
        
        if language == "ar":
            # Try to extract task name from various patterns
            patterns = [
                r"انقل (.+) إلى",
                r"حرك (.+) إلى", 
                r"ضع (.+) في",
                r"اجعل (.+) (عاجل|عالي|متوسط|منخفض)",
                r"أولوية (.+) (عاجل|عالي|متوسط|منخفض)",
                r"أكملت (.+)",
                r"انتهيت من (.+)",
                r"(.+) مكتمل"
            ]
        else:
            patterns = [
                r"move (.+) to",
                r"put (.+) in",
                r"make (.+) (urgent|high|medium|low)",
                r"set (.+) priority",
                r"completed (.+)",
                r"finished (.+)",
                r"done with (.+)"
            ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None

    def _extract_column(self, text: str, language: str) -> KanbanColumn:
        """Extract target column from voice text."""
        text_lower = text.lower()
        
        # Arabic column names
        if any(word in text_lower for word in ["قائمة", "مهام"]):
            return KanbanColumn.BACKLOG
        elif any(word in text_lower for word in ["للقيام", "انتظار"]):
            return KanbanColumn.TODO
        elif any(word in text_lower for word in ["تنفيذ", "عمل"]):
            return KanbanColumn.IN_PROGRESS
        elif any(word in text_lower for word in ["مراجعة", "فحص"]):
            return KanbanColumn.REVIEW
        elif any(word in text_lower for word in ["مكتمل", "انتهى"]):
            return KanbanColumn.DONE
        
        # English column names
        elif any(word in text_lower for word in ["backlog", "queue"]):
            return KanbanColumn.BACKLOG
        elif any(word in text_lower for word in ["todo", "to do"]):
            return KanbanColumn.TODO
        elif any(word in text_lower for word in ["progress", "working"]):
            return KanbanColumn.IN_PROGRESS
        elif any(word in text_lower for word in ["review", "check"]):
            return KanbanColumn.REVIEW
        elif any(word in text_lower for word in ["done", "completed"]):
            return KanbanColumn.DONE
        
        # Default to TODO
        return KanbanColumn.TODO

    def _extract_target_column(self, text: str, language: str) -> Optional[KanbanColumn]:
        """Extract target column from move command."""
        return self._extract_column(text, language)

    def _extract_priority(self, text: str, language: str) -> TaskPriority:
        """Extract priority from voice text."""
        text_lower = text.lower()
        
        # Arabic priorities
        if "عاجل" in text_lower:
            return TaskPriority.URGENT
        elif "عالي" in text_lower:
            return TaskPriority.HIGH
        elif "متوسط" in text_lower:
            return TaskPriority.MEDIUM
        elif "منخفض" in text_lower:
            return TaskPriority.LOW
        
        # English priorities
        elif "urgent" in text_lower:
            return TaskPriority.URGENT
        elif "high" in text_lower:
            return TaskPriority.HIGH
        elif "medium" in text_lower:
            return TaskPriority.MEDIUM
        elif "low" in text_lower:
            return TaskPriority.LOW
        
        # Default to medium
        return TaskPriority.MEDIUM

    def _find_task_by_name(self, name: str) -> Optional[KanbanTask]:
        """Find task by partial name match."""
        current_board = self.boards[self.current_board_id]
        name_lower = name.lower()
        
        # First try exact match
        for task in current_board.tasks.values():
            if task.title.lower() == name_lower:
                return task
        
        # Then try partial match
        for task in current_board.tasks.values():
            if name_lower in task.title.lower() or task.title.lower() in name_lower:
                return task
        
        return None

    def _get_priority_icon(self, priority: TaskPriority) -> str:
        """Get icon for task priority."""
        icons = {
            TaskPriority.URGENT: "🔴",
            TaskPriority.HIGH: "🟠", 
            TaskPriority.MEDIUM: "🟡",
            TaskPriority.LOW: "🟢"
        }
        return icons.get(priority, "⚪")

    def get_board_summary(self) -> Dict[str, Any]:
        """Get summary of current board."""
        current_board = self.boards[self.current_board_id]
        
        # Count tasks by column
        column_counts = {column: 0 for column in KanbanColumn}
        priority_counts = {priority: 0 for priority in TaskPriority}
        
        for task in current_board.tasks.values():
            column_counts[task.column] += 1
            priority_counts[task.priority] += 1
        
        return {
            "board_name": current_board.name,
            "total_tasks": len(current_board.tasks),
            "column_counts": {col.value: count for col, count in column_counts.items()},
            "priority_counts": {pri.value: count for pri, count in priority_counts.items()},
            "last_updated": current_board.updated_at.isoformat()
        }

    async def speak_response(self, text: str, language: str = "ar") -> None:
        """Convert text response to speech."""
        try:
            await self.tts_engine.speak(text, language=language)
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("VoiceKanbanInterface cleanup completed")