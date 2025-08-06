"""
Schedule Optimizer for Time-Block Planner
Author: Drmusab
Last Modified: 2025-01-20

Optimizes daily schedules using time-blocking techniques with intelligent task allocation.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.observability.logging.config import get_logger
from .time_block_planner import Task, TimeBlock, UserPreferences, FocusType, TaskComplexity


class ScheduleOptimizer:
    """Optimizes task scheduling using time-blocking principles."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Optimization parameters
        self.focus_decay_factor = 0.8  # Focus decreases over time
        self.context_switch_penalty = 15  # Minutes penalty for switching contexts
        self.optimal_block_ratios = {
            FocusType.DEEP_WORK: 0.4,    # 40% of work time
            FocusType.ADMIN: 0.25,       # 25% of work time
            FocusType.COMMUNICATION: 0.20, # 20% of work time
            FocusType.CREATIVE: 0.10,    # 10% of work time
            FocusType.LEARNING: 0.05     # 5% of work time
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the schedule optimizer."""
        self.logger.info("ScheduleOptimizer initialized")

    async def create_schedule(
        self, 
        tasks: List[Task], 
        preferences: UserPreferences, 
        calendar_events: List[Dict] = None
    ) -> List[TimeBlock]:
        """Create an optimized schedule for the given tasks."""
        if calendar_events is None:
            calendar_events = []
        
        # Sort tasks by priority and complexity
        sorted_tasks = self._prioritize_tasks(tasks)
        
        # Create time slots avoiding calendar conflicts
        available_slots = self._create_available_slots(preferences, calendar_events)
        
        # Optimize task allocation
        schedule = self._allocate_tasks_to_slots(sorted_tasks, available_slots, preferences)
        
        # Add breaks and buffer time
        optimized_schedule = self._add_breaks_and_buffers(schedule, preferences)
        
        return optimized_schedule

    def _prioritize_tasks(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by priority, complexity, and deadlines."""
        def task_score(task: Task) -> Tuple[int, int, int]:
            # Priority score (higher = more urgent)
            priority_scores = {
                "عاجل": 4, "عالي": 3, "متوسط": 2, "منخفض": 1
            }
            priority_score = priority_scores.get(task.priority.value, 2)
            
            # Deadline urgency (sooner = higher score)
            deadline_score = 0
            if task.deadline:
                days_until = (task.deadline - datetime.now(timezone.utc)).days
                if days_until <= 0:
                    deadline_score = 100  # Overdue
                elif days_until == 1:
                    deadline_score = 50   # Due tomorrow
                elif days_until <= 3:
                    deadline_score = 25   # Due this week
                else:
                    deadline_score = 10   # Due later
            
            # Complexity score (more complex = higher score for better time allocation)
            complexity_scores = {
                "معقد جداً": 4, "معقد": 3, "متوسط": 2, "بسيط": 1
            }
            complexity_score = complexity_scores.get(task.complexity.value, 2)
            
            return (-priority_score, -deadline_score, -complexity_score)
        
        return sorted(tasks, key=task_score)

    def _create_available_slots(
        self, 
        preferences: UserPreferences, 
        calendar_events: List[Dict]
    ) -> List[Tuple[datetime, datetime]]:
        """Create available time slots avoiding calendar conflicts."""
        # Start with work hours
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        work_start = today.replace(
            hour=int(preferences.work_start_time.split(':')[0]),
            minute=int(preferences.work_start_time.split(':')[1])
        )
        work_end = today.replace(
            hour=int(preferences.work_end_time.split(':')[0]),
            minute=int(preferences.work_end_time.split(':')[1])
        )
        
        # Create base slots
        available_slots = [(work_start, work_end)]
        
        # Remove calendar conflicts
        for event in calendar_events:
            event_start = self._parse_event_time(event.get('start_time'))
            event_end = self._parse_event_time(event.get('end_time'))
            
            if event_start and event_end:
                available_slots = self._subtract_time_range(
                    available_slots, 
                    event_start, 
                    event_end
                )
        
        # Filter out slots that are too small
        min_slot_size = timedelta(minutes=30)
        available_slots = [
            (start, end) for start, end in available_slots
            if end - start >= min_slot_size
        ]
        
        return available_slots

    def _parse_event_time(self, time_str: Optional[str]) -> Optional[datetime]:
        """Parse event time string to datetime."""
        if not time_str:
            return None
        
        try:
            # Handle different time formats
            if 'T' in time_str:
                return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            else:
                return datetime.strptime(time_str, '%H:%M').replace(
                    year=datetime.now().year,
                    month=datetime.now().month,
                    day=datetime.now().day
                )
        except Exception as e:
            self.logger.warning(f"Could not parse event time '{time_str}': {str(e)}")
            return None

    def _subtract_time_range(
        self, 
        slots: List[Tuple[datetime, datetime]], 
        conflict_start: datetime, 
        conflict_end: datetime
    ) -> List[Tuple[datetime, datetime]]:
        """Remove a time range from available slots."""
        new_slots = []
        
        for slot_start, slot_end in slots:
            # No overlap
            if conflict_end <= slot_start or conflict_start >= slot_end:
                new_slots.append((slot_start, slot_end))
                continue
            
            # Partial overlaps
            if conflict_start > slot_start:
                new_slots.append((slot_start, conflict_start))
            
            if conflict_end < slot_end:
                new_slots.append((conflict_end, slot_end))
        
        return new_slots

    def _allocate_tasks_to_slots(
        self, 
        tasks: List[Task], 
        available_slots: List[Tuple[datetime, datetime]], 
        preferences: UserPreferences
    ) -> List[TimeBlock]:
        """Allocate tasks to time slots optimally."""
        schedule = []
        remaining_tasks = tasks.copy()
        
        # Identify peak focus hours
        peak_hours = self._parse_peak_hours(preferences.focus_peak_hours)
        
        # First pass: Allocate high-priority and deep work tasks to peak hours
        remaining_tasks, peak_schedule = self._allocate_peak_tasks(
            remaining_tasks, available_slots, peak_hours, preferences
        )
        schedule.extend(peak_schedule)
        
        # Second pass: Allocate remaining tasks
        remaining_schedule = self._allocate_remaining_tasks(
            remaining_tasks, available_slots, preferences, schedule
        )
        schedule.extend(remaining_schedule)
        
        return sorted(schedule, key=lambda x: x.start_time)

    def _parse_peak_hours(self, peak_hours_list: List[str]) -> List[Tuple[int, int]]:
        """Parse peak focus hours from strings."""
        peaks = []
        for time_range in peak_hours_list:
            try:
                start_str, end_str = time_range.split('-')
                start_hour = int(start_str.split(':')[0])
                end_hour = int(end_str.split(':')[0])
                peaks.append((start_hour, end_hour))
            except (ValueError, IndexError):
                self.logger.warning(f"Could not parse peak hour range: {time_range}")
        return peaks

    def _allocate_peak_tasks(
        self, 
        tasks: List[Task], 
        available_slots: List[Tuple[datetime, datetime]], 
        peak_hours: List[Tuple[int, int]], 
        preferences: UserPreferences
    ) -> Tuple[List[Task], List[TimeBlock]]:
        """Allocate high-priority tasks to peak focus hours."""
        schedule = []
        remaining_tasks = []
        
        # Filter tasks that need peak focus
        peak_tasks = [
            task for task in tasks 
            if task.focus_type == FocusType.DEEP_WORK or 
               task.complexity.value in ["معقد", "معقد جداً"]
        ]
        other_tasks = [task for task in tasks if task not in peak_tasks]
        
        for task in peak_tasks:
            allocated = False
            
            for slot_start, slot_end in available_slots:
                # Check if slot overlaps with peak hours
                if self._is_peak_time(slot_start, slot_end, peak_hours):
                    task_duration = timedelta(minutes=task.estimated_duration)
                    
                    if slot_end - slot_start >= task_duration:
                        # Allocate task
                        block = TimeBlock(
                            id=f"block_{uuid.uuid4().hex[:8]}",
                            start_time=slot_start,
                            end_time=slot_start + task_duration,
                            task=task,
                            block_type="work"
                        )
                        schedule.append(block)
                        
                        # Update available slots
                        available_slots = self._subtract_time_range(
                            available_slots, 
                            slot_start, 
                            slot_start + task_duration
                        )
                        allocated = True
                        break
            
            if not allocated:
                remaining_tasks.append(task)
        
        remaining_tasks.extend(other_tasks)
        return remaining_tasks, schedule

    def _is_peak_time(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        peak_hours: List[Tuple[int, int]]
    ) -> bool:
        """Check if time slot overlaps with peak focus hours."""
        for peak_start, peak_end in peak_hours:
            slot_start_hour = start_time.hour
            slot_end_hour = end_time.hour
            
            # Check for overlap
            if not (slot_end_hour <= peak_start or slot_start_hour >= peak_end):
                return True
        return False

    def _allocate_remaining_tasks(
        self, 
        tasks: List[Task], 
        available_slots: List[Tuple[datetime, datetime]], 
        preferences: UserPreferences,
        existing_schedule: List[TimeBlock]
    ) -> List[TimeBlock]:
        """Allocate remaining tasks to available slots."""
        schedule = []
        
        for task in tasks:
            task_duration = timedelta(minutes=task.estimated_duration)
            allocated = False
            
            # Find best fitting slot
            best_slot = None
            best_score = float('-inf')
            
            for slot_start, slot_end in available_slots:
                if slot_end - slot_start >= task_duration:
                    # Calculate allocation score
                    score = self._calculate_allocation_score(
                        task, slot_start, preferences, existing_schedule + schedule
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_slot = (slot_start, slot_end)
            
            if best_slot:
                slot_start, slot_end = best_slot
                
                # Create time block
                block = TimeBlock(
                    id=f"block_{uuid.uuid4().hex[:8]}",
                    start_time=slot_start,
                    end_time=slot_start + task_duration,
                    task=task,
                    block_type="work"
                )
                schedule.append(block)
                
                # Update available slots
                available_slots = self._subtract_time_range(
                    available_slots, 
                    slot_start, 
                    slot_start + task_duration
                )
        
        return schedule

    def _calculate_allocation_score(
        self, 
        task: Task, 
        start_time: datetime, 
        preferences: UserPreferences,
        existing_blocks: List[TimeBlock]
    ) -> float:
        """Calculate score for allocating a task to a specific time."""
        score = 0
        
        # Time of day preference (higher score for morning for deep work)
        hour = start_time.hour
        if task.focus_type == FocusType.DEEP_WORK:
            if 9 <= hour <= 11:  # Morning peak
                score += 10
            elif 14 <= hour <= 16:  # Afternoon peak
                score += 5
        
        # Context switching penalty
        recent_blocks = [
            block for block in existing_blocks 
            if abs((block.end_time - start_time).total_seconds()) < 3600  # Within 1 hour
        ]
        
        for block in recent_blocks:
            if block.task and block.task.focus_type != task.focus_type:
                score -= 5  # Penalty for context switching
            elif block.task and block.task.focus_type == task.focus_type:
                score += 2  # Bonus for similar tasks
        
        # Deadline urgency
        if task.deadline:
            hours_until_deadline = (task.deadline - start_time).total_seconds() / 3600
            if hours_until_deadline < 24:
                score += 20  # High urgency bonus
            elif hours_until_deadline < 72:
                score += 10  # Medium urgency bonus
        
        return score

    def _add_breaks_and_buffers(
        self, 
        schedule: List[TimeBlock], 
        preferences: UserPreferences
    ) -> List[TimeBlock]:
        """Add breaks and buffer time to the schedule."""
        enhanced_schedule = []
        schedule = sorted(schedule, key=lambda x: x.start_time)
        
        for i, block in enumerate(schedule):
            enhanced_schedule.append(block)
            
            # Add break after deep work blocks
            if (block.task and 
                block.task.focus_type == FocusType.DEEP_WORK and 
                i < len(schedule) - 1):
                
                next_block = schedule[i + 1]
                break_start = block.end_time
                break_end = min(
                    break_start + timedelta(minutes=preferences.preferred_break_length),
                    next_block.start_time
                )
                
                if break_end - break_start >= timedelta(minutes=5):
                    break_block = TimeBlock(
                        id=f"break_{uuid.uuid4().hex[:8]}",
                        start_time=break_start,
                        end_time=break_end,
                        block_type="break"
                    )
                    enhanced_schedule.append(break_block)
        
        # Add buffer time
        buffer_enhanced = self._add_buffer_time(enhanced_schedule, preferences)
        
        return sorted(buffer_enhanced, key=lambda x: x.start_time)

    def _add_buffer_time(
        self, 
        schedule: List[TimeBlock], 
        preferences: UserPreferences
    ) -> List[TimeBlock]:
        """Add buffer time between blocks."""
        if not schedule:
            return schedule
        
        enhanced_schedule = []
        schedule = sorted(schedule, key=lambda x: x.start_time)
        
        for i, block in enumerate(schedule):
            enhanced_schedule.append(block)
            
            # Add buffer time between consecutive work blocks
            if i < len(schedule) - 1:
                next_block = schedule[i + 1]
                gap_duration = (next_block.start_time - block.end_time).total_seconds() / 60
                
                # If gap is small, add buffer time
                if 0 < gap_duration < 30:
                    buffer_duration = min(gap_duration * preferences.buffer_time_percentage, 10)
                    
                    if buffer_duration >= 2:  # Minimum 2 minutes buffer
                        buffer_block = TimeBlock(
                            id=f"buffer_{uuid.uuid4().hex[:8]}",
                            start_time=block.end_time,
                            end_time=block.end_time + timedelta(minutes=buffer_duration),
                            block_type="buffer"
                        )
                        enhanced_schedule.append(buffer_block)
        
        return enhanced_schedule

    async def reoptimize_schedule(
        self, 
        current_schedule: List[TimeBlock], 
        new_constraints: Dict[str, Any],
        preferences: UserPreferences
    ) -> List[TimeBlock]:
        """Reoptimize schedule based on new constraints or disruptions."""
        # Extract incomplete tasks
        current_time = datetime.now(timezone.utc)
        incomplete_tasks = []
        
        for block in current_schedule:
            if (block.task and 
                block.start_time > current_time and 
                not new_constraints.get("completed_tasks", {}).get(block.task.id, False)):
                incomplete_tasks.append(block.task)
        
        # Create new available slots from current time
        remaining_day_slots = self._create_remaining_day_slots(current_time, preferences)
        
        # Reoptimize
        new_schedule = self._allocate_tasks_to_slots(
            incomplete_tasks, 
            remaining_day_slots, 
            preferences
        )
        
        return self._add_breaks_and_buffers(new_schedule, preferences)

    def _create_remaining_day_slots(
        self, 
        current_time: datetime, 
        preferences: UserPreferences
    ) -> List[Tuple[datetime, datetime]]:
        """Create available slots for the remaining day."""
        work_end = current_time.replace(
            hour=int(preferences.work_end_time.split(':')[0]),
            minute=int(preferences.work_end_time.split(':')[1]),
            second=0,
            microsecond=0
        )
        
        if current_time >= work_end:
            return []  # No time remaining today
        
        return [(current_time, work_end)]