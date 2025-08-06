"""
Disruption Handler for Time-Block Planner
Author: Drmusab
Last Modified: 2025-01-20

Handles real-time schedule adjustments and disruptions with Arabic-first responses.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.observability.logging.config import get_logger
from .time_block_planner import TimeBlock, UserPreferences, FocusType


class DisruptionType:
    """Types of schedule disruptions."""
    OVERRUN = "overrun"  # Task taking longer than expected
    INTERRUPTION = "interruption"  # External interruption
    EARLY_COMPLETION = "early_completion"  # Task completed early
    EMERGENCY = "emergency"  # Emergency task insertion
    DISTRACTION = "distraction"  # User distraction detected
    EXTERNAL_MEETING = "external_meeting"  # New meeting added


class DisruptionHandler:
    """Handles schedule disruptions and real-time adjustments."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Disruption handling strategies
        self.strategies = {
            DisruptionType.OVERRUN: self._handle_overrun,
            DisruptionType.INTERRUPTION: self._handle_interruption,
            DisruptionType.EARLY_COMPLETION: self._handle_early_completion,
            DisruptionType.EMERGENCY: self._handle_emergency,
            DisruptionType.DISTRACTION: self._handle_distraction,
            DisruptionType.EXTERNAL_MEETING: self._handle_external_meeting
        }
        
        # Arabic responses for different disruption types
        self.arabic_responses = {
            DisruptionType.OVERRUN: {
                "message": "لاحظت أن المهمة تستغرق وقتاً أطول من المتوقع. هل تريد تمديد الوقت أم تأجيل المهام التالية؟",
                "options": ["تمديد الوقت", "تأجيل المهام", "إنهاء المهمة الآن"]
            },
            DisruptionType.INTERRUPTION: {
                "message": "تم اكتشاف مقاطعة. هل تريد أخذ استراحة قصيرة أم الاستمرار؟", 
                "options": ["استراحة قصيرة", "الاستمرار", "إعادة تنظيم الجدول"]
            },
            DisruptionType.DISTRACTION: {
                "message": "يبدو أنك مشتت. هل تريد أخذ استراحة للتركيز أم تقصير هذه الكتلة الزمنية؟",
                "options": ["استراحة للتركيز", "تقصير الكتلة", "تغيير المهمة"]
            },
            DisruptionType.EMERGENCY: {
                "message": "مهمة عاجلة! سأعيد ترتيب جدولك لاستيعابها.",
                "options": ["موافق", "تأجيل للاحقاً", "رفض"]
            }
        }
        
        # English responses (secondary)
        self.english_responses = {
            DisruptionType.OVERRUN: {
                "message": "Task is taking longer than expected. Extend time or postpone next tasks?",
                "options": ["Extend time", "Postpone tasks", "End task now"]
            },
            DisruptionType.INTERRUPTION: {
                "message": "Interruption detected. Take a short break or continue?",
                "options": ["Short break", "Continue", "Reorganize schedule"]
            },
            DisruptionType.DISTRACTION: {
                "message": "You seem distracted. Take a focus break or shorten this block?",
                "options": ["Focus break", "Shorten block", "Change task"]
            },
            DisruptionType.EMERGENCY: {
                "message": "Emergency task! Reorganizing your schedule to accommodate it.",
                "options": ["OK", "Postpone", "Decline"]
            }
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the disruption handler."""
        self.logger.info("DisruptionHandler initialized")

    async def handle_disruption(
        self, 
        current_schedule: List[TimeBlock], 
        disruption_data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle a schedule disruption and return adjusted schedule."""
        disruption_type = disruption_data.get("type", DisruptionType.INTERRUPTION)
        
        # Log the disruption
        self.logger.info(f"Handling disruption of type: {disruption_type}")
        
        # Route to appropriate handler
        handler = self.strategies.get(disruption_type, self._handle_generic_disruption)
        adjusted_schedule = await handler(current_schedule, disruption_data)
        
        return adjusted_schedule

    async def _handle_overrun(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle task overrun - task taking longer than expected."""
        current_time = datetime.now(timezone.utc)
        task_id = data.get("task_id")
        additional_minutes = data.get("additional_minutes", 30)
        user_choice = data.get("user_choice", "extend")  # extend, postpone, end
        
        # Find the current task block
        current_block = None
        current_index = -1
        
        for i, block in enumerate(schedule):
            if (block.task and 
                block.task.id == task_id and 
                block.start_time <= current_time <= block.end_time):
                current_block = block
                current_index = i
                break
        
        if not current_block:
            return schedule  # Task not found
        
        adjusted_schedule = schedule.copy()
        
        if user_choice == "extend":
            # Extend current task and shift subsequent tasks
            extension = timedelta(minutes=additional_minutes)
            current_block.end_time += extension
            
            # Shift all subsequent blocks
            for i in range(current_index + 1, len(adjusted_schedule)):
                adjusted_schedule[i].start_time += extension
                adjusted_schedule[i].end_time += extension
                
        elif user_choice == "postpone":
            # Move subsequent tasks and find new slot for overrunning task
            adjusted_schedule = self._postpone_subsequent_tasks(
                adjusted_schedule, current_index, additional_minutes
            )
            
        elif user_choice == "end":
            # End current task now and adjust subsequent tasks
            current_block.end_time = current_time
            # Add incomplete task to end of schedule
            remaining_time = current_block.task.estimated_duration - \
                           int((current_time - current_block.start_time).total_seconds() / 60)
            
            if remaining_time > 0:
                # Create a new block for the remaining work
                last_block = adjusted_schedule[-1]
                remainder_block = TimeBlock(
                    id=f"remainder_{uuid.uuid4().hex[:8]}",
                    start_time=last_block.end_time,
                    end_time=last_block.end_time + timedelta(minutes=remaining_time),
                    task=current_block.task,
                    block_type="work"
                )
                adjusted_schedule.append(remainder_block)
        
        return adjusted_schedule

    async def _handle_interruption(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle external interruption."""
        current_time = datetime.now(timezone.utc)
        interruption_duration = data.get("duration_minutes", 15)
        user_choice = data.get("user_choice", "short_break")
        
        # Find current block
        current_block = None
        current_index = -1
        
        for i, block in enumerate(schedule):
            if block.start_time <= current_time <= block.end_time:
                current_block = block
                current_index = i
                break
        
        if not current_block:
            return schedule
        
        adjusted_schedule = schedule.copy()
        interruption_delta = timedelta(minutes=interruption_duration)
        
        if user_choice == "short_break":
            # Insert break and shift subsequent tasks
            break_block = TimeBlock(
                id=f"interruption_{uuid.uuid4().hex[:8]}",
                start_time=current_time,
                end_time=current_time + interruption_delta,
                block_type="break"
            )
            
            # Adjust current block if needed
            if current_block.task:
                # Split current block
                remaining_time = current_block.end_time - current_time - interruption_delta
                if remaining_time.total_seconds() > 300:  # More than 5 minutes remaining
                    current_block.end_time = current_time
                    
                    continuation_block = TimeBlock(
                        id=f"continuation_{uuid.uuid4().hex[:8]}",
                        start_time=current_time + interruption_delta,
                        end_time=current_time + interruption_delta + remaining_time,
                        task=current_block.task,
                        block_type="work"
                    )
                    
                    adjusted_schedule.insert(current_index + 1, break_block)
                    adjusted_schedule.insert(current_index + 2, continuation_block)
                    
                    # Shift remaining blocks
                    for i in range(current_index + 3, len(adjusted_schedule)):
                        adjusted_schedule[i].start_time += interruption_delta
                        adjusted_schedule[i].end_time += interruption_delta
        
        elif user_choice == "reorganize":
            # Completely reorganize remaining schedule
            incomplete_tasks = self._extract_incomplete_tasks(schedule, current_time)
            remaining_time_slots = self._calculate_remaining_slots(schedule, current_time)
            adjusted_schedule = self._create_new_schedule_from_tasks(
                incomplete_tasks, remaining_time_slots
            )
        
        return adjusted_schedule

    async def _handle_early_completion(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle early task completion."""
        current_time = datetime.now(timezone.utc)
        task_id = data.get("task_id")
        time_saved_minutes = data.get("time_saved", 0)
        
        # Find completed task
        for i, block in enumerate(schedule):
            if (block.task and 
                block.task.id == task_id and 
                block.start_time <= current_time <= block.end_time):
                
                # Mark as completed early
                block.end_time = current_time
                
                # Shift subsequent tasks earlier
                time_saved = timedelta(minutes=time_saved_minutes)
                for j in range(i + 1, len(schedule)):
                    schedule[j].start_time -= time_saved
                    schedule[j].end_time -= time_saved
                
                # Add optional early break
                if time_saved_minutes >= 10:
                    break_block = TimeBlock(
                        id=f"early_break_{uuid.uuid4().hex[:8]}",
                        start_time=current_time,
                        end_time=current_time + timedelta(minutes=min(time_saved_minutes, 15)),
                        block_type="break"
                    )
                    schedule.insert(i + 1, break_block)
                
                break
        
        return schedule

    async def _handle_emergency(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle emergency task insertion."""
        current_time = datetime.now(timezone.utc)
        emergency_task = data.get("emergency_task")
        urgency = data.get("urgency", "high")  # high, medium, low
        estimated_duration = data.get("duration_minutes", 60)
        
        if not emergency_task:
            return schedule
        
        # Create emergency task block
        emergency_block = TimeBlock(
            id=f"emergency_{uuid.uuid4().hex[:8]}",
            start_time=current_time,
            end_time=current_time + timedelta(minutes=estimated_duration),
            task=emergency_task,
            block_type="work"
        )
        
        adjusted_schedule = [emergency_block]
        
        # Reschedule remaining tasks
        new_start_time = emergency_block.end_time
        
        for block in schedule:
            if block.start_time > current_time:
                # Shift future blocks
                time_shift = new_start_time - block.start_time
                block.start_time = new_start_time
                block.end_time += time_shift
                adjusted_schedule.append(block)
                new_start_time = block.end_time
        
        return adjusted_schedule

    async def _handle_distraction(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle user distraction during focused work."""
        current_time = datetime.now(timezone.utc)
        distraction_level = data.get("level", "medium")  # low, medium, high
        user_choice = data.get("user_choice", "focus_break")
        
        # Find current block
        current_block = None
        current_index = -1
        
        for i, block in enumerate(schedule):
            if block.start_time <= current_time <= block.end_time:
                current_block = block
                current_index = i
                break
        
        if not current_block or not current_block.task:
            return schedule
        
        adjusted_schedule = schedule.copy()
        
        if user_choice == "focus_break":
            # Insert 5-10 minute focus break
            break_duration = 5 if distraction_level == "low" else 10
            break_block = TimeBlock(
                id=f"focus_break_{uuid.uuid4().hex[:8]}",
                start_time=current_time,
                end_time=current_time + timedelta(minutes=break_duration),
                block_type="break"
            )
            
            # Extend current task to compensate
            current_block.end_time += timedelta(minutes=break_duration)
            
            # Shift subsequent blocks
            shift_time = timedelta(minutes=break_duration)
            for i in range(current_index + 1, len(adjusted_schedule)):
                adjusted_schedule[i].start_time += shift_time
                adjusted_schedule[i].end_time += shift_time
            
            adjusted_schedule.insert(current_index + 1, break_block)
            
        elif user_choice == "shorten_block":
            # Shorten current block and move to easier task
            if current_block.task.focus_type == FocusType.DEEP_WORK:
                # Reduce by 25%
                reduction = timedelta(minutes=current_block.task.estimated_duration * 0.25)
                current_block.end_time -= reduction
                
                # Shift subsequent blocks earlier
                for i in range(current_index + 1, len(adjusted_schedule)):
                    adjusted_schedule[i].start_time -= reduction
                    adjusted_schedule[i].end_time -= reduction
                    
        elif user_choice == "change_task":
            # Switch to an admin or communication task if available
            for i in range(current_index + 1, len(adjusted_schedule)):
                next_block = adjusted_schedule[i]
                if (next_block.task and 
                    next_block.task.focus_type in [FocusType.ADMIN, FocusType.COMMUNICATION]):
                    
                    # Swap tasks
                    current_block.task, next_block.task = next_block.task, current_block.task
                    break
        
        return adjusted_schedule

    async def _handle_external_meeting(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle external meeting insertion."""
        meeting_start = data.get("start_time")
        meeting_end = data.get("end_time")
        
        if not meeting_start or not meeting_end:
            return schedule
        
        # Convert to datetime objects if needed
        if isinstance(meeting_start, str):
            meeting_start = datetime.fromisoformat(meeting_start)
        if isinstance(meeting_end, str):
            meeting_end = datetime.fromisoformat(meeting_end)
        
        # Create meeting block
        meeting_block = TimeBlock(
            id=f"meeting_{uuid.uuid4().hex[:8]}",
            start_time=meeting_start,
            end_time=meeting_end,
            block_type="meeting"
        )
        
        # Remove conflicting blocks and reschedule
        adjusted_schedule = []
        conflicted_tasks = []
        
        for block in schedule:
            # Check for overlap
            if not (block.end_time <= meeting_start or block.start_time >= meeting_end):
                # Conflict detected
                if block.task:
                    conflicted_tasks.append(block.task)
            else:
                adjusted_schedule.append(block)
        
        # Add meeting block
        adjusted_schedule.append(meeting_block)
        
        # Reschedule conflicted tasks
        if conflicted_tasks:
            # Find available slots after meeting
            available_start = meeting_end
            for task in conflicted_tasks:
                task_block = TimeBlock(
                    id=f"rescheduled_{uuid.uuid4().hex[:8]}",
                    start_time=available_start,
                    end_time=available_start + timedelta(minutes=task.estimated_duration),
                    task=task,
                    block_type="work"
                )
                adjusted_schedule.append(task_block)
                available_start = task_block.end_time
        
        return sorted(adjusted_schedule, key=lambda x: x.start_time)

    async def _handle_generic_disruption(
        self, 
        schedule: List[TimeBlock], 
        data: Dict[str, Any]
    ) -> List[TimeBlock]:
        """Handle generic/unknown disruption types."""
        self.logger.warning(f"Unknown disruption type: {data.get('type', 'unknown')}")
        
        # Default: add a 10-minute buffer and shift subsequent tasks
        current_time = datetime.now(timezone.utc)
        buffer_duration = timedelta(minutes=10)
        
        for i, block in enumerate(schedule):
            if block.start_time > current_time:
                block.start_time += buffer_duration
                block.end_time += buffer_duration
        
        return schedule

    def _postpone_subsequent_tasks(
        self, 
        schedule: List[TimeBlock], 
        current_index: int, 
        delay_minutes: int
    ) -> List[TimeBlock]:
        """Postpone subsequent tasks by given delay."""
        delay = timedelta(minutes=delay_minutes)
        
        for i in range(current_index + 1, len(schedule)):
            schedule[i].start_time += delay
            schedule[i].end_time += delay
        
        return schedule

    def _extract_incomplete_tasks(self, schedule: List[TimeBlock], current_time: datetime) -> List:
        """Extract tasks that haven't been completed yet."""
        incomplete_tasks = []
        
        for block in schedule:
            if (block.task and 
                block.start_time > current_time):
                incomplete_tasks.append(block.task)
        
        return incomplete_tasks

    def _calculate_remaining_slots(
        self, 
        schedule: List[TimeBlock], 
        current_time: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Calculate remaining available time slots."""
        # This is a simplified implementation
        # In practice, would consider work hours, break requirements, etc.
        
        last_block = max(schedule, key=lambda x: x.end_time, default=None)
        if last_block:
            end_of_day = last_block.end_time.replace(hour=17, minute=0)  # Assume 5 PM end
            return [(current_time, end_of_day)]
        else:
            return [(current_time, current_time.replace(hour=17, minute=0))]

    def _create_new_schedule_from_tasks(
        self, 
        tasks: List, 
        time_slots: List[Tuple[datetime, datetime]]
    ) -> List[TimeBlock]:
        """Create new schedule from tasks and available slots."""
        new_schedule = []
        
        if not tasks or not time_slots:
            return new_schedule
        
        current_time = time_slots[0][0]
        
        for task in tasks:
            task_duration = timedelta(minutes=task.estimated_duration)
            block = TimeBlock(
                id=f"rescheduled_{uuid.uuid4().hex[:8]}",
                start_time=current_time,
                end_time=current_time + task_duration,
                task=task,
                block_type="work"
            )
            new_schedule.append(block)
            current_time = block.end_time + timedelta(minutes=5)  # 5-minute buffer
        
        return new_schedule

    async def suggest_adjustment(
        self, 
        disruption_type: str, 
        context: Dict[str, Any], 
        language: str = "ar"
    ) -> Dict[str, Any]:
        """Suggest adjustment for a disruption with localized responses."""
        responses = self.arabic_responses if language == "ar" else self.english_responses
        
        response_data = responses.get(disruption_type, {
            "message": "حدث تغيير في الجدول" if language == "ar" else "Schedule change occurred",
            "options": ["موافق" if language == "ar" else "OK"]
        })
        
        return {
            "type": disruption_type,
            "message": response_data["message"],
            "options": response_data["options"],
            "language": language,
            "context": context
        }