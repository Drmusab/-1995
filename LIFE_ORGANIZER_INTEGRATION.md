"""
Life Organizer Integration Guide - دليل دمج منظم الحياة
Author: Drmusab
Last Modified: 2025-01-20

Integration guide for adding the Life Organizer skill to the main AI assistant system.
"""

# Life Organizer Integration Guide

## Overview - النظرة العامة

The Life Organizer skill provides comprehensive life management capabilities with Arabic-first support. This guide shows how to integrate it with the existing AI assistant system.

## Skills Implemented - المهارات المطبقة

### 1. Core Components - المكونات الأساسية

```
src/skills/builtin/life_organizer/
├── __init__.py                          # Package initialization
├── life_organizer.py                    # Main skill orchestrator  
├── mood_energy_tracker.py               # Mood/energy tracking via voice+vision
├── voice_kanban_interface.py            # Voice-controlled Kanban board
└── adaptive_recommendation_engine.py    # Intelligent recommendations
```

### 2. Key Features - الميزات الرئيسية

#### Goal Decomposition - تحليل الأهداف
- Breaks down complex goals into actionable steps
- Integrates with existing `GoalDecomposer`
- Arabic-first goal parsing and step generation

#### Mood & Energy Tracking - تتبع المزاج والطاقة
- **Voice Analysis**: Uses sentiment analysis from speech
- **Vision Analysis**: Uses facial expression analysis
- **Combined Assessment**: Weighted confidence scoring
- **Arabic Emotional Context**: Cultural sensitivity in mood interpretation

#### Voice Kanban Interface - واجهة كانبان الصوتية
- **Arabic Commands**: "أضف مهمة", "أظهر اللوحة", "انقل المهمة"
- **English Commands**: "add task", "show board", "move task"
- **Task Management**: Create, move, prioritize, complete tasks
- **Board Visualization**: Multi-column Kanban view

#### Adaptive Recommendations - التوصيات التكيفية
- **Mood-Aware Scheduling**: Adjusts task timing based on energy levels
- **Cultural Context**: Arabic work patterns and break preferences
- **Intelligent Suggestions**: "You seem tired, reschedule deep work?"
- **Proactive Advice**: Recommends optimal task types for current state

## Integration Steps - خطوات الدمج

### 1. Register the Skill - تسجيل المهارة

Add to `src/skills/skill_registry.py`:

```python
from src.skills.builtin.life_organizer import LifeOrganizerSkill

# Register Life Organizer skill
skill_registry.register_skill(
    skill_id="life_organizer.main",
    skill_class=LifeOrganizerSkill,
    metadata={
        "name": "منظم الحياة الذكي",
        "description": "منظم شامل للحياة مع دعم الصوت والرؤية",
        "version": "1.0.0",
        "languages": ["ar", "en"],
        "capabilities": [
            "goal_decomposition",
            "mood_tracking", 
            "voice_control",
            "adaptive_scheduling"
        ]
    }
)
```

### 2. Update Skill Factory - تحديث مصنع المهارات

In `src/skills/skill_factory.py`:

```python
def create_life_organizer_skill(container: Container) -> LifeOrganizerSkill:
    """Create Life Organizer skill with dependencies."""
    return LifeOrganizerSkill(container)

# Add to skill factory mapping
SKILL_MAPPING = {
    # ... existing skills
    "life_organizer.main": create_life_organizer_skill,
}
```

### 3. Configure Dependencies - تهيئة التبعيات

Add to dependency injection configuration:

```python
# Required services for Life Organizer
container.register_singleton(SentimentAnalyzer)
container.register_singleton(ExpressionAnalyzer)
container.register_singleton(GoalDecomposer)
container.register_singleton(TimeBlockPlanner)
container.register_singleton(BilingualManager)
```

### 4. API Integration - دمج واجهة برمجة التطبيقات

Add endpoints in `src/api/routes/skills.py`:

```python
@router.post("/life-organizer/goal-decomposition")
async def decompose_goal(request: GoalDecompositionRequest):
    """Decompose goal into actionable steps."""
    skill = await get_skill("life_organizer.main")
    return await skill.process_request({
        "type": "goal_decomposition",
        "goal": request.goal,
        "language": request.language
    })

@router.post("/life-organizer/mood-tracking")
async def track_mood(request: MoodTrackingRequest):
    """Track mood and energy from voice/vision."""
    skill = await get_skill("life_organizer.main")
    return await skill.process_request({
        "type": "mood_tracking",
        "voice_text": request.voice_text,
        "image_data": request.image_data,
        "language": request.language
    })

@router.post("/life-organizer/voice-kanban")
async def voice_kanban(request: VoiceKanbanRequest):
    """Process voice Kanban command."""
    skill = await get_skill("life_organizer.main")
    return await skill.process_request({
        "type": "voice_kanban",
        "voice_command": request.voice_command,
        "language": request.language
    })
```

### 5. CLI Integration - دمج واجهة سطر الأوامر

Add commands in `src/cli.py`:

```python
@cli.command("life-organizer")
@click.option("--mode", type=click.Choice(["goal", "mood", "kanban", "schedule"]))
@click.option("--language", default="ar", type=click.Choice(["ar", "en"]))
@click.option("--input", help="Input text or command")
async def life_organizer(mode: str, language: str, input: str):
    """Life Organizer commands."""
    skill = await get_skill("life_organizer.main")
    
    request_type = {
        "goal": "goal_decomposition",
        "mood": "mood_tracking", 
        "kanban": "voice_kanban",
        "schedule": "adaptive_scheduling"
    }.get(mode, "general")
    
    result = await skill.process_request({
        "type": request_type,
        "input": input,
        "language": language
    })
    
    click.echo(result["response"])
```

## Usage Examples - أمثلة الاستخدام

### Arabic Voice Commands - أوامر صوتية عربية

```python
# Goal decomposition
await life_organizer.process_request({
    "type": "goal_decomposition",
    "goal": "تعلم الذكاء الاصطناعي خلال سنة",
    "language": "ar"
})

# Voice Kanban commands
await life_organizer.process_request({
    "type": "voice_kanban", 
    "voice_command": "أضف مهمة كتابة البحث عاجل",
    "language": "ar"
})

# Mood tracking with voice
await life_organizer.process_request({
    "type": "mood_tracking",
    "voice_text": "أشعر بالحماس والطاقة اليوم",
    "language": "ar"
})
```

### English Commands

```python
# Goal decomposition in English
await life_organizer.process_request({
    "type": "goal_decomposition",
    "goal": "Learn machine learning in 6 months",
    "language": "en"
})

# Voice Kanban in English
await life_organizer.process_request({
    "type": "voice_kanban",
    "voice_command": "add task review code high priority", 
    "language": "en"
})
```

## Configuration - التهيئة

### Environment Variables

```bash
# Life Organizer Configuration
LIFE_ORGANIZER_ENABLED=true
LIFE_ORGANIZER_DEFAULT_LANGUAGE=ar
LIFE_ORGANIZER_MOOD_TRACKING_ENABLED=true
LIFE_ORGANIZER_VOICE_KANBAN_ENABLED=true
LIFE_ORGANIZER_ADAPTIVE_SCHEDULING_ENABLED=true

# Mood tracking sensitivity
MOOD_TRACKER_CONFIDENCE_THRESHOLD=0.7
MOOD_TRACKER_HISTORY_LIMIT=50

# Kanban board settings
KANBAN_DEFAULT_BOARD_NAME="اللوحة الشخصية"
KANBAN_MAX_TASKS_PER_COLUMN=20

# Recommendation engine
RECOMMENDATION_ENGINE_MAX_SUGGESTIONS=5
RECOMMENDATION_ENGINE_URGENCY_THRESHOLD=3
```

### Skill Configuration File

Create `configs/skills/life_organizer.yaml`:

```yaml
life_organizer:
  enabled: true
  default_language: "ar"
  
  mood_tracking:
    enabled: true
    confidence_threshold: 0.7
    sources: ["voice", "vision"]
    history_limit: 50
    
  voice_kanban:
    enabled: true
    default_board: "اللوحة الشخصية"
    max_tasks_per_column: 20
    voice_patterns:
      arabic:
        - "أضف مهمة (.+)"
        - "أظهر اللوحة" 
        - "انقل (.+) إلى (.+)"
      english:
        - "add task (.+)"
        - "show board"
        - "move (.+) to (.+)"
        
  adaptive_recommendations:
    enabled: true
    max_suggestions: 5
    urgency_threshold: 3
    recommendation_rules:
      high_mood_high_energy:
        - type: "تحسين التوقيت"
          urgency: 1
      low_mood_low_energy:
        - type: "أخذ استراحة"
          urgency: 5
```

## Testing - الاختبار

Run the comprehensive test suite:

```bash
# Run minimal tests (no dependencies)
python test_life_organizer_minimal.py

# Run full integration tests (requires dependencies)
python test_life_organizer.py

# Run demo
python demo_life_organizer.py
```

## Monitoring & Observability - المراقبة والملاحظة

The Life Organizer skill provides comprehensive metrics:

### Metrics Exposed

- `life_organizer_requests_total{type, language}` - Total requests by type and language
- `life_organizer_mood_tracking_accuracy` - Mood tracking accuracy over time  
- `life_organizer_kanban_tasks_created` - Number of Kanban tasks created
- `life_organizer_recommendations_generated` - Number of recommendations generated
- `life_organizer_goal_decompositions` - Number of goals decomposed

### Health Checks

```python
# Health check endpoint
GET /health/skills/life-organizer

Response:
{
    "status": "healthy",
    "components": {
        "mood_tracker": "active",
        "kanban_interface": "active", 
        "recommendation_engine": "active"
    },
    "metrics": {
        "goals_count": 15,
        "daily_plans_count": 7,
        "active_tasks": 12
    }
}
```

## Security Considerations - اعتبارات الأمان

- **Voice Data**: Voice inputs are processed locally when possible
- **Vision Data**: Facial expressions analyzed without storing images
- **Personal Data**: Mood and task data encrypted at rest
- **Access Control**: Role-based access to life organizer features
- **Privacy**: User can disable mood tracking at any time

## Performance Optimizations - تحسينات الأداء

- **Caching**: Mood states and recommendations cached for 30 minutes
- **Async Processing**: All heavy operations use async/await
- **Memory Management**: Limited history storage (50 mood states, 100 recommendations)
- **Efficient Pattern Matching**: Compiled regex patterns for voice commands
- **Batch Processing**: Multiple recommendations generated in single call

## Troubleshooting - استكشاف الأخطاء

### Common Issues

1. **Mood tracking not working**
   - Check sentiment analyzer and expression analyzer dependencies
   - Verify voice/vision input format
   - Check confidence thresholds in configuration

2. **Voice commands not recognized**
   - Verify language setting matches input language
   - Check voice pattern regex in configuration
   - Ensure intent manager is properly initialized

3. **Recommendations not relevant**
   - Check mood/energy state accuracy
   - Review recommendation rules configuration
   - Verify user feedback integration

### Debug Commands

```bash
# Check skill status
ai-assistant life-organizer --mode status

# Test mood tracking
ai-assistant life-organizer --mode mood --input "أشعر بالسعادة"

# Test voice Kanban
ai-assistant life-organizer --mode kanban --input "أظهر اللوحة"

# Check recommendations
ai-assistant life-organizer --mode schedule
```

## Future Enhancements - التحسينات المستقبلية

1. **Machine Learning Integration**
   - Learn from user behavior patterns
   - Personalized recommendation algorithms
   - Predictive mood analysis

2. **Calendar Integration**
   - Sync with external calendars
   - Smart scheduling conflicts detection
   - Meeting mood preparation

3. **Team Collaboration**
   - Shared Kanban boards
   - Team mood analytics
   - Collaborative goal setting

4. **Cultural Adaptations**
   - Region-specific work patterns
   - Cultural holidays and breaks
   - Local productivity customs

---

This completes the Life Organizer integration guide. The skill is ready for production deployment with comprehensive Arabic support and intelligent mood-aware features.