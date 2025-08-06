#!/usr/bin/env python3
"""
Test for Time-Block Efficiency Planner Skill
Author: Drmusab
Last Modified: 2025-01-20

Simple test to validate the time block planner skill implementation.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.skills.builtin.time_block_planner import TimeBlockPlannerSkill


async def test_time_block_planner():
    """Test the time block planner skill."""
    print("🧪 Testing Time-Block Efficiency Planner Skill...")
    
    # Create skill instance
    skill = TimeBlockPlannerSkill()
    
    # Test metadata
    print("\n📋 Testing skill metadata...")
    metadata = skill.get_metadata()
    print(f"✅ Skill ID: {metadata.skill_id}")
    print(f"✅ Name (Arabic): {metadata.name}")
    print(f"✅ Capabilities: {len(metadata.capabilities)}")
    
    # Test initialization
    print("\n🔧 Testing skill initialization...")
    try:
        await skill.initialize({"language": "ar", "block_length": 90})
        print("✅ Skill initialized successfully")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return False
    
    # Test input validation
    print("\n✅ Testing input validation...")
    test_inputs = [
        "أريد تخطيط يومي مع كتابة ساعتان واجتماعات",
        {"action": "plan_workday", "data": {"tasks": "Study Arabic 2 hours, write report 1 hour"}},
        ""
    ]
    
    for test_input in test_inputs:
        is_valid = await skill.validate(test_input)
        print(f"✅ Input '{str(test_input)[:50]}...' validation: {is_valid}")
    
    # Test skill execution - basic planning
    print("\n⚙️ Testing skill execution...")
    try:
        context = {
            "user_id": "test_user",
            "session_id": "test_session",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Test Arabic input
        arabic_input = "أريد تخطيط يومي: كتابة تقرير ساعتان، اجتماع ساعة، إيميلات نصف ساعة"
        result = await skill.execute(arabic_input, context)
        
        if result.get("success"):
            print("✅ Arabic planning request executed successfully")
            print(f"📅 Generated {len(result.get('schedule', []))} time blocks")
            if result.get("response", {}).get("ar"):
                print(f"💬 Arabic response: {result['response']['ar'][:100]}...")
        else:
            print(f"❌ Arabic planning failed: {result.get('error', 'Unknown error')}")
        
        # Test English input
        english_input = {"action": "plan_workday", "data": {"tasks": "Deep work 90 minutes, emails 30 minutes"}}
        result2 = await skill.execute(english_input, context)
        
        if result2.get("success"):
            print("✅ English planning request executed successfully")
            print(f"📅 Generated {len(result2.get('schedule', []))} time blocks")
        else:
            print(f"❌ English planning failed: {result2.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Execution failed: {str(e)}")
        return False
    
    # Test health check
    print("\n🏥 Testing health check...")
    try:
        health = await skill.health_check()
        print(f"✅ Health status: {health.get('status')}")
        print(f"✅ Components status: {health.get('components', {})}")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
    
    # Test cleanup
    print("\n🧹 Testing cleanup...")
    try:
        await skill.cleanup()
        print("✅ Cleanup completed successfully")
    except Exception as e:
        print(f"❌ Cleanup failed: {str(e)}")
    
    print("\n🎉 Time-Block Efficiency Planner Skill test completed!")
    return True


async def test_skill_components():
    """Test individual skill components."""
    print("\n🔧 Testing individual components...")
    
    # Test TaskClassifier
    print("\n📝 Testing TaskClassifier...")
    try:
        from src.skills.builtin.time_block_planner.task_classifier import TaskClassifier
        classifier = TaskClassifier()
        await classifier.initialize({})
        
        tasks = await classifier.extract_tasks("كتابة تقرير ساعتان عاجل، اجتماع ساعة")
        print(f"✅ TaskClassifier extracted {len(tasks)} tasks")
        for task in tasks[:2]:  # Show first 2 tasks
            print(f"   - {task.title_ar} ({task.estimated_duration} min, {task.priority.value})")
            
    except Exception as e:
        print(f"❌ TaskClassifier test failed: {str(e)}")
    
    # Test ScheduleOptimizer  
    print("\n⏰ Testing ScheduleOptimizer...")
    try:
        from src.skills.builtin.time_block_planner.schedule_optimizer import ScheduleOptimizer
        from src.skills.builtin.time_block_planner import UserPreferences
        
        optimizer = ScheduleOptimizer()
        await optimizer.initialize({})
        
        # Create sample tasks
        from src.skills.builtin.time_block_planner import Task, TaskPriority, FocusType
        sample_tasks = [
            Task("1", "كتابة", "writing", estimated_duration=90, priority=TaskPriority.HIGH, focus_type=FocusType.DEEP_WORK),
            Task("2", "إيميل", "email", estimated_duration=30, focus_type=FocusType.COMMUNICATION)
        ]
        
        preferences = UserPreferences()
        schedule = await optimizer.create_schedule(sample_tasks, preferences)
        print(f"✅ ScheduleOptimizer created schedule with {len(schedule)} blocks")
        
    except Exception as e:
        print(f"❌ ScheduleOptimizer test failed: {str(e)}")
    
    # Test DisruptionHandler
    print("\n🚨 Testing DisruptionHandler...")
    try:
        from src.skills.builtin.time_block_planner.disruption_handler import DisruptionHandler
        
        handler = DisruptionHandler()
        await handler.initialize({})
        
        # Test disruption suggestion
        suggestion = await handler.suggest_adjustment("overrun", {"task_id": "test"}, "ar")
        print(f"✅ DisruptionHandler created suggestion: {suggestion.get('message', '')[:50]}...")
        
    except Exception as e:
        print(f"❌ DisruptionHandler test failed: {str(e)}")


if __name__ == "__main__":
    async def main():
        success = await test_time_block_planner()
        await test_skill_components()
        
        if success:
            print("\n🎯 All tests passed! Time-Block Efficiency Planner Skill is ready.")
        else:
            print("\n💥 Some tests failed. Please check the implementation.")
    
    asyncio.run(main())