# Time-Block Efficiency Planner Skill Examples

This document demonstrates how to use the Time-Block Efficiency Planner skill with Arabic-first language support.

## Basic Usage Examples

### Arabic Examples (Primary Language)

```python
# Plan workday with Arabic input
arabic_input = "أريد تخطيط يومي: كتابة تقرير ساعتان عاجل، اجتماع فريق ساعة، إيميلات نصف ساعة"

result = await skill.execute(arabic_input, {
    "user_id": "user123",
    "session_id": "session456"
})

# Output:
# {
#   "success": true,
#   "schedule": [
#     {
#       "start_time": "09:00",
#       "end_time": "11:00", 
#       "task": "كتابة تقرير",
#       "priority": "عاجل",
#       "focus_type": "عمل عميق"
#     },
#     {
#       "start_time": "11:00",
#       "end_time": "11:15",
#       "block_type": "استراحة"
#     },
#     ...
#   ],
#   "response": {
#     "ar": "إليك جدولك المحسن لليوم:\n⏰ 09:00–11:00: كتابة تقرير...",
#     "en": "Here's your optimized schedule for today:\n⏰ 09:00–11:00: Write report..."
#   }
# }
```

### English Examples (Secondary Language)

```python
# Plan workday with English input
english_input = {
    "action": "plan_workday",
    "data": {
        "tasks": "Deep work 2 hours urgent, team meeting 1 hour, emails 30 minutes"
    }
}

result = await skill.execute(english_input, context)
```

### Complex Planning Examples

```python
# Arabic: Complex daily planning with priorities and time constraints
complex_arabic = """
كتابة مقترح المشروع ساعتان عاجل - معقد
اجتماع مع العميل ساعة مهم
مراجعة الإيميلات نصف ساعة عادي
تحضير العرض التقديمي ساعة مهم - إبداعي
استراحة غداء ساعة
متابعة المهام الإدارية 45 دقيقة بسيط
"""

result = await skill.execute(complex_arabic, context)
```

## Schedule Adjustment Examples

```python
# Handle task overrun
disruption_data = {
    "type": "overrun",
    "task_id": "task_123",
    "additional_minutes": 30,
    "user_choice": "extend"  # or "postpone", "end"
}

adjusted_schedule = await skill.execute({
    "action": "adjust_schedule",
    "data": disruption_data
}, context)
```

## Real-time Interaction Examples

```python
# Arabic interaction for distraction handling
distraction_input = "أشعر بالتشتت، هل يمكن أخذ استراحة قصيرة؟"

response = await skill.execute(distraction_input, context)
# Output: "يبدو أنك مشتت. هل تريد أخذ استراحة للتركيز أم تقصير هذه الكتلة الزمنية؟"
```

## Voice Reminder Examples

```python
# Request voice reminder
reminder_request = {
    "action": "provide_reminder",
    "data": {
        "message": "حان وقت كتلة العمل العميق. ركز وأغلق جميع المشتتات."
    }
}

audio_response = await skill.execute(reminder_request, context)
# Returns both text and TTS audio in Arabic
```

## Integration Examples

### Calendar Integration

```python
# Planning with calendar conflict detection
planning_with_calendar = {
    "action": "plan_workday",
    "data": {
        "tasks": "كتابة التقرير ساعتان، مراجعة المستندات ساعة",
        "check_calendar": True
    }
}

# The skill will automatically avoid scheduling conflicts
result = await skill.execute(planning_with_calendar, context)
```

### Performance Tracking

```python
# Track focus performance
focus_data = {
    "action": "track_focus",
    "data": {
        "task_id": "task_123",
        "completed": True,
        "actual_duration": 85,  # minutes
        "focus_rating": 8,  # 1-10 scale
        "interruptions": 2
    }
}

performance = await skill.execute(focus_data, context)
# Output: "معدل الإنجاز الحالي: 85%"
```

## Typical Daily Workflow

```python
# Morning: Plan the day
morning_planning = "تخطيط اليوم: كتابة المقال ساعتان، اجتماعات ساعة، إيميلات 30 دقيقة، مراجعة المشروع ساعة"
schedule = await skill.execute(morning_planning, context)

# Mid-day: Handle disruption
disruption = {
    "type": "interruption",
    "duration_minutes": 20,
    "user_choice": "short_break"
}
adjusted = await skill.execute({"action": "adjust_schedule", "data": disruption}, context)

# Evening: Track performance
performance_data = {
    "action": "track_focus",
    "data": {"completion_summary": "completed 3/4 tasks, good focus in morning"}
}
insights = await skill.execute(performance_data, context)
```

## Key Features Demonstrated

1. **Arabic-first Language Support**: All responses prioritize Arabic
2. **Intelligent Task Parsing**: Extracts duration, priority, and focus type
3. **Optimized Scheduling**: Places deep work in peak focus hours
4. **Real-time Adjustments**: Handles interruptions and overruns
5. **Bilingual Responses**: Provides both Arabic and English output
6. **Voice Integration**: TTS reminders in Arabic
7. **Performance Learning**: Tracks patterns for future optimization
8. **Calendar Integration**: Avoids scheduling conflicts