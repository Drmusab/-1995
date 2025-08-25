"""
Coaching & Personal Development Skills for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 15:30:00 UTC

This module provides coaching and personal development skills for the AI assistant,
including personalized coaching, habit formation, study motivation, debate coaching,
charisma amplification, executive presence, and neuro-linguistic persuasion.
"""

import json
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, Union

import asyncio
import numpy as np

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ContextAdapted,
    FeedbackReceived,
    SkillExecutionCompleted,
    SkillExecutionFailed,
    SkillExecutionStarted,
    SkillRegistered,
)
from src.core.health_check import HealthCheck
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.external_apis.web_search import WebSearchAPI

# Integration components
from src.integrations.llm.model_router import ModelRouter
from src.integrations.storage.database import DatabaseManager

# Memory systems
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import EpisodicMemory, SemanticMemory, WorkingMemory

# Natural language processing
from src.processing.natural_language.intent_manager import IntentManager
from src.processing.natural_language.sentiment_analyzer import SentimentAnalyzer

# Personalization and learning
from src.personalization.learning.preference_learner import PreferenceLearner
from src.personalization.learning.reinforcement_learner import ReinforcementLearner

# Import from core_skills for base classes and types
from .core_skills import (
    BaseSkill,
    SkillCategory,
    SkillMetadata,
    SkillPriority,
    SkillResult,
    SkillError,
)

# Workflow and component management
from src.workflows.orchestrator import WorkflowOrchestrator

# Utilities
from src.utils.logger import get_logger


class PersonalizedCoachingAssistantSkill(BaseSkill):
    """Core coaching skill that provides personalized coaching assistance based on user goals and preferences."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="coaching.personalized_assistant",
            name="Personalized Coaching Assistant",
            description="Provides personalized coaching assistance based on user goals, preferences, and progress tracking",
            category=SkillCategory.COACHING,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (assess_goals, create_plan, track_progress, provide_guidance, adjust_plan)",
                    "required": True,
                },
                "user_id": {"type": "string", "description": "User identifier", "required": True},
                "goal_type": {
                    "type": "string", 
                    "description": "Type of goal (career, personal, health, skills, relationships)",
                    "required": False,
                },
                "goal_description": {
                    "type": "string",
                    "description": "Detailed description of the goal",
                    "required": False,
                },
                "progress_data": {
                    "type": "object",
                    "description": "Progress tracking data",
                    "required": False,
                },
                "guidance_request": {
                    "type": "string",
                    "description": "Specific guidance or advice needed",
                    "required": False,
                },
            },
            examples=[
                {"operation": "assess_goals", "user_id": "user_123", "goal_type": "career"},
                {
                    "operation": "create_plan",
                    "user_id": "user_123",
                    "goal_type": "skills",
                    "goal_description": "Learn Python programming within 6 months",
                },
                {
                    "operation": "track_progress",
                    "user_id": "user_123",
                    "progress_data": {"completed_tasks": 5, "total_tasks": 20},
                },
            ],
            required_components=["MemoryManager", "ModelRouter", "PreferenceLearner"],
            tags={"coaching", "goal_setting", "progress_tracking", "personalization"},
            is_stateful=True,
        )

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """Execute the personalized coaching assistant skill."""
        try:
            operation = input_data.get("operation")
            user_id = input_data.get("user_id")

            if not operation or not user_id:
                return SkillResult(
                    success=False,
                    data={},
                    message="Missing required parameters: operation and user_id",
                    errors=["operation and user_id are required"],
                )

            if operation == "assess_goals":
                return await self._assess_goals(user_id, input_data.get("goal_type"))
            elif operation == "create_plan":
                return await self._create_plan(user_id, input_data)
            elif operation == "track_progress":
                return await self._track_progress(user_id, input_data.get("progress_data", {}))
            elif operation == "provide_guidance":
                return await self._provide_guidance(user_id, input_data.get("guidance_request", ""))
            elif operation == "adjust_plan":
                return await self._adjust_plan(user_id, input_data)
            else:
                return SkillResult(
                    success=False,
                    data={},
                    message=f"Unknown operation: {operation}",
                    errors=[f"Unsupported operation: {operation}"],
                )

        except Exception as e:
            self.logger.error(f"Error in PersonalizedCoachingAssistantSkill: {str(e)}")
            return SkillResult(
                success=False,
                data={},
                message="Failed to execute coaching assistant",
                errors=[str(e)],
            )

    async def _assess_goals(self, user_id: str, goal_type: Optional[str] = None) -> SkillResult:
        """Assess user's current goals and situation."""
        try:
            # Retrieve user's goal history and preferences
            user_memories = await self.memory_manager.get_user_memories(user_id, limit=50)
            
            # Analyze existing goals and progress
            current_goals = []
            goal_patterns = {}
            
            for memory in user_memories:
                if isinstance(memory, dict) and "goals" in str(memory).lower():
                    current_goals.append(memory)
                    
                    # Extract goal type patterns
                    if "goal_type" in memory:
                        gtype = memory["goal_type"]
                        goal_patterns[gtype] = goal_patterns.get(gtype, 0) + 1

            # Generate assessment
            assessment = {
                "current_goals": current_goals,
                "goal_patterns": goal_patterns,
                "recommended_focus": self._recommend_focus_areas(goal_patterns, goal_type),
                "assessment_summary": f"Found {len(current_goals)} existing goals with focus on {max(goal_patterns, key=goal_patterns.get) if goal_patterns else 'no specific area'}",
            }

            return SkillResult(
                success=True,
                data=assessment,
                message="Goal assessment completed successfully",
                confidence=0.9,
                next_actions=["create_plan", "provide_guidance"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to assess goals",
                errors=[str(e)],
            )

    async def _create_plan(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Create a personalized coaching plan."""
        try:
            goal_type = input_data.get("goal_type", "general")
            goal_description = input_data.get("goal_description", "")
            
            # Generate coaching plan based on goal type and description
            plan = {
                "goal_id": str(uuid.uuid4()),
                "goal_type": goal_type,
                "description": goal_description,
                "milestones": self._generate_milestones(goal_type, goal_description),
                "timeline": self._generate_timeline(goal_type),
                "strategies": self._generate_strategies(goal_type),
                "success_metrics": self._generate_success_metrics(goal_type),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store the plan in memory
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=plan,
                memory_type="coaching_plan",
                metadata={"goal_type": goal_type, "plan_id": plan["goal_id"]},
            )

            return SkillResult(
                success=True,
                data=plan,
                message="Personalized coaching plan created successfully",
                confidence=0.95,
                next_actions=["track_progress", "provide_guidance"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to create coaching plan",
                errors=[str(e)],
            )

    async def _track_progress(self, user_id: str, progress_data: Dict[str, Any]) -> SkillResult:
        """Track user's progress against their goals."""
        try:
            # Retrieve current goals and plans
            user_memories = await self.memory_manager.get_user_memories(user_id, memory_type="coaching_plan")
            
            if not user_memories:
                return SkillResult(
                    success=False,
                    data={},
                    message="No coaching plans found for progress tracking",
                    errors=["No active coaching plans"],
                )

            # Calculate progress metrics
            progress_analysis = {
                "overall_progress": self._calculate_overall_progress(progress_data),
                "milestone_status": self._analyze_milestone_progress(progress_data),
                "recommendations": self._generate_progress_recommendations(progress_data),
                "next_steps": self._suggest_next_steps(progress_data),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store progress update
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=progress_analysis,
                memory_type="progress_update",
                metadata={"progress_date": datetime.now(timezone.utc).isoformat()},
            )

            return SkillResult(
                success=True,
                data=progress_analysis,
                message="Progress tracking completed successfully",
                confidence=0.9,
                next_actions=["provide_guidance", "adjust_plan"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to track progress",
                errors=[str(e)],
            )

    async def _provide_guidance(self, user_id: str, guidance_request: str) -> SkillResult:
        """Provide personalized coaching guidance."""
        try:
            # Retrieve user context and history
            user_memories = await self.memory_manager.get_user_memories(user_id, limit=20)
            
            # Generate contextual guidance
            guidance = {
                "guidance_type": self._determine_guidance_type(guidance_request),
                "recommendations": self._generate_guidance_recommendations(guidance_request, user_memories),
                "action_items": self._generate_action_items(guidance_request),
                "resources": self._suggest_resources(guidance_request),
                "follow_up": self._schedule_follow_up(guidance_request),
                "provided_at": datetime.now(timezone.utc).isoformat(),
            }

            return SkillResult(
                success=True,
                data=guidance,
                message="Personalized guidance provided successfully",
                confidence=0.85,
                next_actions=["track_progress"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to provide guidance",
                errors=[str(e)],
            )

    async def _adjust_plan(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Adjust the coaching plan based on progress and feedback."""
        try:
            # Retrieve current plan and progress
            plans = await self.memory_manager.get_user_memories(user_id, memory_type="coaching_plan")
            progress = await self.memory_manager.get_user_memories(user_id, memory_type="progress_update")
            
            if not plans:
                return SkillResult(
                    success=False,
                    data={},
                    message="No coaching plan found to adjust",
                    errors=["No active coaching plans"],
                )

            # Generate plan adjustments
            adjustments = {
                "plan_id": plans[0].get("goal_id") if plans else str(uuid.uuid4()),
                "adjustments_made": self._generate_plan_adjustments(plans[0], progress),
                "new_timeline": self._adjust_timeline(plans[0], progress),
                "updated_strategies": self._update_strategies(plans[0], progress),
                "rationale": self._explain_adjustments(plans[0], progress),
                "adjusted_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store the adjusted plan
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=adjustments,
                memory_type="plan_adjustment",
                metadata={"adjustment_date": datetime.now(timezone.utc).isoformat()},
            )

            return SkillResult(
                success=True,
                data=adjustments,
                message="Coaching plan adjusted successfully",
                confidence=0.9,
                next_actions=["track_progress", "provide_guidance"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to adjust plan",
                errors=[str(e)],
            )

    def _recommend_focus_areas(self, goal_patterns: Dict[str, int], goal_type: Optional[str]) -> List[str]:
        """Recommend focus areas based on goal patterns."""
        if goal_type:
            return [goal_type]
        
        if not goal_patterns:
            return ["career", "personal_development", "health"]
        
        # Return top 3 goal types
        sorted_goals = sorted(goal_patterns.items(), key=lambda x: x[1], reverse=True)
        return [goal for goal, _ in sorted_goals[:3]]

    def _generate_milestones(self, goal_type: str, description: str) -> List[Dict[str, Any]]:
        """Generate milestones for the goal."""
        base_milestones = {
            "career": [
                {"name": "Skills Assessment", "timeline": "Week 1", "description": "Evaluate current skills and identify gaps"},
                {"name": "Learning Plan", "timeline": "Week 2", "description": "Create detailed learning and development plan"},
                {"name": "Mid-term Review", "timeline": "Month 3", "description": "Review progress and adjust goals"},
                {"name": "Goal Achievement", "timeline": "Month 6", "description": "Achieve primary career objective"},
            ],
            "personal": [
                {"name": "Self-Assessment", "timeline": "Week 1", "description": "Reflect on current state and desired outcomes"},
                {"name": "Habit Formation", "timeline": "Month 1", "description": "Establish key supporting habits"},
                {"name": "Progress Check", "timeline": "Month 2", "description": "Evaluate progress and adjust approach"},
                {"name": "Goal Completion", "timeline": "Month 4", "description": "Achieve personal development goal"},
            ],
            "health": [
                {"name": "Baseline Assessment", "timeline": "Week 1", "description": "Establish current health metrics"},
                {"name": "Routine Establishment", "timeline": "Week 2", "description": "Create sustainable health routine"},
                {"name": "Monthly Check-in", "timeline": "Month 1", "description": "Review progress and adjust plan"},
                {"name": "Goal Achievement", "timeline": "Month 3", "description": "Reach health objectives"},
            ],
            "skills": [
                {"name": "Skill Gap Analysis", "timeline": "Week 1", "description": "Identify specific skills to develop"},
                {"name": "Learning Resources", "timeline": "Week 2", "description": "Gather learning materials and resources"},
                {"name": "Practice Phase", "timeline": "Month 2", "description": "Active skill development and practice"},
                {"name": "Mastery Demonstration", "timeline": "Month 4", "description": "Demonstrate skill proficiency"},
            ],
        }
        return base_milestones.get(goal_type, base_milestones["personal"])

    def _generate_timeline(self, goal_type: str) -> Dict[str, str]:
        """Generate timeline for the goal."""
        timelines = {
            "career": "6 months",
            "personal": "4 months", 
            "health": "3 months",
            "skills": "4 months",
            "relationships": "3 months",
        }
        return {
            "duration": timelines.get(goal_type, "4 months"),
            "start_date": datetime.now(timezone.utc).isoformat(),
            "target_completion": (datetime.now(timezone.utc) + timedelta(weeks=16)).isoformat(),
        }

    def _generate_strategies(self, goal_type: str) -> List[str]:
        """Generate strategies for the goal type."""
        strategies = {
            "career": [
                "Regular skill development sessions",
                "Networking and relationship building",
                "Performance tracking and feedback",
                "Mentorship engagement",
            ],
            "personal": [
                "Daily reflection and journaling",
                "Habit stacking techniques",
                "Regular self-assessment",
                "Support system engagement",
            ],
            "health": [
                "Consistent exercise routine",
                "Nutrition planning and tracking",
                "Sleep optimization",
                "Stress management techniques",
            ],
            "skills": [
                "Structured learning approach",
                "Practical application opportunities",
                "Regular practice sessions",
                "Peer learning and feedback",
            ],
        }
        return strategies.get(goal_type, strategies["personal"])

    def _generate_success_metrics(self, goal_type: str) -> List[str]:
        """Generate success metrics for the goal type."""
        metrics = {
            "career": [
                "Skill proficiency improvements",
                "Performance review scores",
                "Goal completion rate",
                "Professional network growth",
            ],
            "personal": [
                "Habit consistency rate",
                "Self-assessment scores",
                "Goal milestone completion",
                "Overall satisfaction rating",
            ],
            "health": [
                "Physical fitness metrics",
                "Energy level improvements",
                "Consistency in healthy habits",
                "Health indicator improvements",
            ],
            "skills": [
                "Skill assessment scores",
                "Project completion quality",
                "Learning milestone achievements",
                "Practical application success",
            ],
        }
        return metrics.get(goal_type, metrics["personal"])

    def _calculate_overall_progress(self, progress_data: Dict[str, Any]) -> float:
        """Calculate overall progress percentage."""
        completed = progress_data.get("completed_tasks", 0)
        total = progress_data.get("total_tasks", 1)
        return min(100.0, (completed / total) * 100) if total > 0 else 0.0

    def _analyze_milestone_progress(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze progress against milestones."""
        return {
            "milestones_completed": progress_data.get("milestones_completed", 0),
            "milestones_total": progress_data.get("milestones_total", 4),
            "current_milestone": progress_data.get("current_milestone", "Initial Planning"),
            "milestone_progress": progress_data.get("milestone_progress", 0.0),
        }

    def _generate_progress_recommendations(self, progress_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on progress."""
        progress_rate = self._calculate_overall_progress(progress_data)
        
        if progress_rate < 25:
            return [
                "Focus on establishing consistent daily habits",
                "Break down goals into smaller, manageable tasks",
                "Consider adjusting timeline if needed",
            ]
        elif progress_rate < 50:
            return [
                "Continue current momentum",
                "Identify and address any obstacles",
                "Celebrate small wins to maintain motivation",
            ]
        elif progress_rate < 75:
            return [
                "Maintain current strategies",
                "Begin preparing for goal completion",
                "Consider expanding scope if appropriate",
            ]
        else:
            return [
                "Focus on final implementation and refinement",
                "Document lessons learned",
                "Prepare for goal achievement celebration",
            ]

    def _suggest_next_steps(self, progress_data: Dict[str, Any]) -> List[str]:
        """Suggest specific next steps."""
        return [
            "Review and update progress tracking",
            "Schedule check-in with accountability partner",
            "Adjust strategies based on current results",
            "Plan upcoming milestone activities",
        ]

    def _determine_guidance_type(self, guidance_request: str) -> str:
        """Determine the type of guidance needed."""
        request_lower = guidance_request.lower()
        
        if any(word in request_lower for word in ["motivation", "inspire", "encourage"]):
            return "motivational"
        elif any(word in request_lower for word in ["strategy", "plan", "approach"]):
            return "strategic"
        elif any(word in request_lower for word in ["obstacle", "challenge", "stuck"]):
            return "problem_solving"
        elif any(word in request_lower for word in ["skill", "learn", "improve"]):
            return "skill_development"
        else:
            return "general"

    def _generate_guidance_recommendations(self, guidance_request: str, user_memories: List[Dict]) -> List[str]:
        """Generate specific guidance recommendations."""
        guidance_type = self._determine_guidance_type(guidance_request)
        
        base_recommendations = {
            "motivational": [
                "Focus on your progress so far, no matter how small",
                "Visualize your end goal and how achieving it will feel",
                "Connect with your deeper 'why' for this goal",
                "Celebrate recent wins and build momentum",
            ],
            "strategic": [
                "Break down large goals into smaller, actionable steps",
                "Identify the most impactful actions to focus on first",
                "Create accountability systems and checkpoints",
                "Consider multiple approaches and test what works best",
            ],
            "problem_solving": [
                "Identify the specific nature of the obstacle",
                "Brainstorm multiple solutions without judgment",
                "Seek input from others who've faced similar challenges",
                "Consider adjusting your approach or timeline",
            ],
            "skill_development": [
                "Focus on deliberate practice with specific feedback",
                "Find opportunities to apply skills in real situations",
                "Connect with mentors or peers in the skill area",
                "Track skill improvements over time",
            ],
            "general": [
                "Reflect on what's working well and what isn't",
                "Stay consistent with small daily actions",
                "Maintain a growth mindset and embrace learning",
                "Regular self-assessment and course correction",
            ],
        }
        
        return base_recommendations.get(guidance_type, base_recommendations["general"])

    def _generate_action_items(self, guidance_request: str) -> List[str]:
        """Generate specific action items."""
        return [
            "Schedule 15 minutes daily for goal-related activities",
            "Identify one small step to take today",
            "Connect with someone who can provide support or feedback",
            "Review and update your progress tracking system",
        ]

    def _suggest_resources(self, guidance_request: str) -> List[str]:
        """Suggest helpful resources."""
        return [
            "Goal-setting and achievement books/articles",
            "Relevant online courses or tutorials",
            "Professional communities or support groups",
            "Coaching or mentoring opportunities",
        ]

    def _schedule_follow_up(self, guidance_request: str) -> Dict[str, str]:
        """Schedule appropriate follow-up."""
        return {
            "next_check_in": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
            "frequency": "weekly",
            "focus": "Progress review and guidance adjustment",
        }

    def _generate_plan_adjustments(self, current_plan: Dict, progress_data: List[Dict]) -> List[str]:
        """Generate specific plan adjustments."""
        return [
            "Adjusted timeline based on current progress rate",
            "Modified strategies to address identified challenges",
            "Updated milestones to reflect new priorities",
            "Enhanced support systems and accountability measures",
        ]

    def _adjust_timeline(self, current_plan: Dict, progress_data: List[Dict]) -> Dict[str, str]:
        """Adjust timeline based on progress."""
        original_end = datetime.fromisoformat(current_plan.get("timeline", {}).get("target_completion", ""))
        
        # Extend by 2 weeks if progress is slow
        if progress_data and len(progress_data) > 0:
            latest_progress = progress_data[0] if isinstance(progress_data[0], dict) else {}
            progress_rate = latest_progress.get("overall_progress", 0)
            
            if progress_rate < 50:
                new_end = original_end + timedelta(weeks=2)
            else:
                new_end = original_end
        else:
            new_end = original_end

        return {
            "original_completion": original_end.isoformat(),
            "new_completion": new_end.isoformat(),
            "adjustment_reason": "Based on current progress rate and challenges identified",
        }

    def _update_strategies(self, current_plan: Dict, progress_data: List[Dict]) -> List[str]:
        """Update strategies based on what's working."""
        current_strategies = current_plan.get("strategies", [])
        
        # Add new strategies based on challenges
        additional_strategies = [
            "Increased accountability check-ins",
            "Smaller, more manageable daily goals",
            "Additional support resources and tools",
            "Enhanced motivation and reward systems",
        ]
        
        return current_strategies + additional_strategies[:2]

    def _explain_adjustments(self, current_plan: Dict, progress_data: List[Dict]) -> str:
        """Explain the rationale for plan adjustments."""
        return (
            "Plan adjustments made based on current progress analysis. "
            "Focus on maintaining momentum while addressing identified challenges. "
            "Timeline extended to ensure sustainable progress and goal achievement. "
            "Enhanced strategies added to improve success probability."
        )


class HabitFormationTrackerSkill(BaseSkill):
    """Skill for tracking and supporting habit formation with behavioral science principles."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="coaching.habit_formation_tracker",
            name="Habit Formation Tracker",
            description="Tracks habit formation progress using behavioral science principles and provides personalized support",
            category=SkillCategory.COACHING,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (create_habit, track_progress, analyze_patterns, suggest_improvements, get_insights)",
                    "required": True,
                },
                "user_id": {"type": "string", "description": "User identifier", "required": True},
                "habit_name": {
                    "type": "string",
                    "description": "Name of the habit to track",
                    "required": False,
                },
                "habit_frequency": {
                    "type": "string",
                    "description": "Desired frequency (daily, weekly, etc.)",
                    "required": False,
                },
                "completion_data": {
                    "type": "object",
                    "description": "Habit completion tracking data",
                    "required": False,
                },
            },
            examples=[
                {"operation": "create_habit", "user_id": "user_123", "habit_name": "morning_exercise", "habit_frequency": "daily"},
                {"operation": "track_progress", "user_id": "user_123", "completion_data": {"completed": True, "notes": "30-minute workout"}},
                {"operation": "analyze_patterns", "user_id": "user_123"},
            ],
            required_components=["MemoryManager", "ModelRouter"],
            tags={"habits", "tracking", "behavioral_science", "progress"},
            is_stateful=True,
        )

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """Execute the habit formation tracker skill."""
        try:
            operation = input_data.get("operation")
            user_id = input_data.get("user_id")

            if not operation or not user_id:
                return SkillResult(
                    success=False,
                    data={},
                    message="Missing required parameters: operation and user_id",
                    errors=["operation and user_id are required"],
                )

            if operation == "create_habit":
                return await self._create_habit(user_id, input_data)
            elif operation == "track_progress":
                return await self._track_progress(user_id, input_data.get("completion_data", {}))
            elif operation == "analyze_patterns":
                return await self._analyze_patterns(user_id)
            elif operation == "suggest_improvements":
                return await self._suggest_improvements(user_id)
            elif operation == "get_insights":
                return await self._get_insights(user_id)
            else:
                return SkillResult(
                    success=False,
                    data={},
                    message=f"Unknown operation: {operation}",
                    errors=[f"Unsupported operation: {operation}"],
                )

        except Exception as e:
            self.logger.error(f"Error in HabitFormationTrackerSkill: {str(e)}")
            return SkillResult(
                success=False,
                data={},
                message="Failed to execute habit formation tracker",
                errors=[str(e)],
            )

    async def _create_habit(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Create a new habit for tracking."""
        try:
            habit_name = input_data.get("habit_name", "")
            habit_frequency = input_data.get("habit_frequency", "daily")
            
            habit_data = {
                "habit_id": str(uuid.uuid4()),
                "name": habit_name,
                "frequency": habit_frequency,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "streak_count": 0,
                "total_completions": 0,
                "status": "active",
                "cue": input_data.get("cue", ""),
                "routine": input_data.get("routine", habit_name),
                "reward": input_data.get("reward", ""),
            }

            # Store habit in memory
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=habit_data,
                memory_type="habit_tracking",
                metadata={"habit_id": habit_data["habit_id"], "habit_name": habit_name},
            )

            return SkillResult(
                success=True,
                data=habit_data,
                message=f"Habit '{habit_name}' created successfully",
                confidence=0.95,
                next_actions=["track_progress"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to create habit",
                errors=[str(e)],
            )

    async def _track_progress(self, user_id: str, completion_data: Dict[str, Any]) -> SkillResult:
        """Track habit completion progress."""
        try:
            # Get user's active habits
            habits = await self.memory_manager.get_user_memories(user_id, memory_type="habit_tracking")
            
            if not habits:
                return SkillResult(
                    success=False,
                    data={},
                    message="No habits found for tracking",
                    errors=["No active habits to track"],
                )

            # Track completion for the most recent habit (or specified habit)
            habit = habits[0] if habits else {}
            habit_id = habit.get("habit_id", "")
            
            completion_entry = {
                "habit_id": habit_id,
                "completed": completion_data.get("completed", False),
                "completion_date": datetime.now(timezone.utc).isoformat(),
                "notes": completion_data.get("notes", ""),
                "quality_rating": completion_data.get("quality_rating", 5),
                "context": completion_data.get("context", {}),
            }

            # Store completion data
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=completion_entry,
                memory_type="habit_completion",
                metadata={"habit_id": habit_id, "date": completion_entry["completion_date"]},
            )

            # Update habit statistics
            updated_stats = await self._update_habit_stats(user_id, habit_id, completion_entry["completed"])

            return SkillResult(
                success=True,
                data={
                    "completion_recorded": completion_entry,
                    "updated_stats": updated_stats,
                },
                message="Habit progress tracked successfully",
                confidence=0.9,
                next_actions=["analyze_patterns", "get_insights"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to track habit progress",
                errors=[str(e)],
            )

    async def _analyze_patterns(self, user_id: str) -> SkillResult:
        """Analyze habit formation patterns."""
        try:
            # Get completion history
            completions = await self.memory_manager.get_user_memories(user_id, memory_type="habit_completion")
            
            if not completions:
                return SkillResult(
                    success=False,
                    data={},
                    message="No habit completion data found for analysis",
                    errors=["Insufficient data for pattern analysis"],
                )

            # Analyze patterns
            patterns = {
                "completion_rate": self._calculate_completion_rate(completions),
                "streak_analysis": self._analyze_streaks(completions),
                "time_patterns": self._analyze_time_patterns(completions),
                "quality_trends": self._analyze_quality_trends(completions),
                "contextual_factors": self._analyze_context_factors(completions),
                "behavioral_insights": self._generate_behavioral_insights(completions),
            }

            return SkillResult(
                success=True,
                data=patterns,
                message="Habit pattern analysis completed",
                confidence=0.85,
                next_actions=["suggest_improvements"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to analyze habit patterns",
                errors=[str(e)],
            )

    async def _suggest_improvements(self, user_id: str) -> SkillResult:
        """Suggest improvements based on habit patterns."""
        try:
            # Get recent patterns
            patterns_result = await self._analyze_patterns(user_id)
            
            if not patterns_result.success:
                return patterns_result

            patterns = patterns_result.data
            
            suggestions = {
                "habit_optimization": self._generate_habit_optimizations(patterns),
                "environmental_changes": self._suggest_environmental_changes(patterns),
                "motivation_strategies": self._suggest_motivation_strategies(patterns),
                "schedule_adjustments": self._suggest_schedule_adjustments(patterns),
                "accountability_measures": self._suggest_accountability_measures(patterns),
            }

            return SkillResult(
                success=True,
                data=suggestions,
                message="Habit improvement suggestions generated",
                confidence=0.8,
                next_actions=["track_progress"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to generate improvement suggestions",
                errors=[str(e)],
            )

    async def _get_insights(self, user_id: str) -> SkillResult:
        """Get behavioral insights and habit formation science."""
        try:
            # Get user's habit data
            habits = await self.memory_manager.get_user_memories(user_id, memory_type="habit_tracking")
            completions = await self.memory_manager.get_user_memories(user_id, memory_type="habit_completion")
            
            insights = {
                "habit_formation_stage": self._determine_formation_stage(completions),
                "behavioral_science_tips": self._get_science_based_tips(),
                "personalized_insights": self._generate_personalized_insights(habits, completions),
                "motivation_psychology": self._explain_motivation_psychology(),
                "next_level_strategies": self._suggest_advanced_strategies(completions),
            }

            return SkillResult(
                success=True,
                data=insights,
                message="Habit formation insights provided",
                confidence=0.9,
                next_actions=["suggest_improvements"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to generate insights",
                errors=[str(e)],
            )

    async def _update_habit_stats(self, user_id: str, habit_id: str, completed: bool) -> Dict[str, Any]:
        """Update habit statistics based on completion."""
        try:
            # Get completion history for this habit
            completions = await self.memory_manager.get_user_memories(
                user_id, 
                memory_type="habit_completion",
                filters={"habit_id": habit_id}
            )
            
            # Calculate stats
            total_completions = sum(1 for c in completions if c.get("completed", False))
            current_streak = self._calculate_current_streak(completions)
            longest_streak = self._calculate_longest_streak(completions)
            
            stats = {
                "total_completions": total_completions,
                "current_streak": current_streak,
                "longest_streak": longest_streak,
                "completion_rate": total_completions / len(completions) if completions else 0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            
            return stats
            
        except Exception:
            return {}

    def _calculate_completion_rate(self, completions: List[Dict]) -> float:
        """Calculate overall completion rate."""
        if not completions:
            return 0.0
        
        completed_count = sum(1 for c in completions if c.get("completed", False))
        return completed_count / len(completions) * 100

    def _analyze_streaks(self, completions: List[Dict]) -> Dict[str, Any]:
        """Analyze completion streaks."""
        if not completions:
            return {"current_streak": 0, "longest_streak": 0, "streak_analysis": "No data available"}
        
        # Sort by date
        sorted_completions = sorted(completions, key=lambda x: x.get("completion_date", ""))
        
        current_streak = self._calculate_current_streak(sorted_completions)
        longest_streak = self._calculate_longest_streak(sorted_completions)
        
        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "streak_analysis": f"Current streak: {current_streak} days, Best streak: {longest_streak} days",
        }

    def _calculate_current_streak(self, completions: List[Dict]) -> int:
        """Calculate current completion streak."""
        if not completions:
            return 0
        
        # Sort by date descending
        sorted_completions = sorted(completions, key=lambda x: x.get("completion_date", ""), reverse=True)
        
        streak = 0
        for completion in sorted_completions:
            if completion.get("completed", False):
                streak += 1
            else:
                break
        
        return streak

    def _calculate_longest_streak(self, completions: List[Dict]) -> int:
        """Calculate longest completion streak."""
        if not completions:
            return 0
        
        sorted_completions = sorted(completions, key=lambda x: x.get("completion_date", ""))
        
        max_streak = 0
        current_streak = 0
        
        for completion in sorted_completions:
            if completion.get("completed", False):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak

    def _analyze_time_patterns(self, completions: List[Dict]) -> Dict[str, Any]:
        """Analyze time-based completion patterns."""
        if not completions:
            return {"pattern": "No data available"}
        
        # Analyze day of week patterns
        day_completions = defaultdict(int)
        day_totals = defaultdict(int)
        
        for completion in completions:
            try:
                date_str = completion.get("completion_date", "")
                if date_str:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    day_name = date_obj.strftime("%A")
                    day_totals[day_name] += 1
                    if completion.get("completed", False):
                        day_completions[day_name] += 1
            except:
                continue
        
        # Calculate completion rates by day
        day_rates = {}
        for day in day_totals:
            rate = (day_completions[day] / day_totals[day]) * 100 if day_totals[day] > 0 else 0
            day_rates[day] = rate
        
        best_day = max(day_rates, key=day_rates.get) if day_rates else "Unknown"
        worst_day = min(day_rates, key=day_rates.get) if day_rates else "Unknown"
        
        return {
            "day_completion_rates": day_rates,
            "best_day": best_day,
            "worst_day": worst_day,
            "pattern_summary": f"Most successful on {best_day}, least successful on {worst_day}",
        }

    def _analyze_quality_trends(self, completions: List[Dict]) -> Dict[str, Any]:
        """Analyze quality rating trends."""
        if not completions:
            return {"trend": "No data available"}
        
        quality_ratings = [c.get("quality_rating", 5) for c in completions if c.get("completed", False)]
        
        if not quality_ratings:
            return {"trend": "No quality data available"}
        
        average_quality = sum(quality_ratings) / len(quality_ratings)
        
        # Simple trend analysis (last 5 vs first 5)
        if len(quality_ratings) >= 10:
            recent_avg = sum(quality_ratings[-5:]) / 5
            early_avg = sum(quality_ratings[:5]) / 5
            trend = "improving" if recent_avg > early_avg else "declining" if recent_avg < early_avg else "stable"
        else:
            trend = "insufficient data"
        
        return {
            "average_quality": average_quality,
            "trend": trend,
            "quality_summary": f"Average quality: {average_quality:.1f}/10, trend: {trend}",
        }

    def _analyze_context_factors(self, completions: List[Dict]) -> Dict[str, Any]:
        """Analyze contextual factors affecting completion."""
        context_success = defaultdict(lambda: {"completed": 0, "total": 0})
        
        for completion in completions:
            context = completion.get("context", {})
            completed = completion.get("completed", False)
            
            for factor, value in context.items():
                context_success[f"{factor}:{value}"]["total"] += 1
                if completed:
                    context_success[f"{factor}:{value}"]["completed"] += 1
        
        # Calculate success rates
        context_rates = {}
        for context_key, data in context_success.items():
            if data["total"] > 0:
                rate = (data["completed"] / data["total"]) * 100
                context_rates[context_key] = rate
        
        return {
            "context_success_rates": context_rates,
            "top_success_contexts": sorted(context_rates.items(), key=lambda x: x[1], reverse=True)[:3],
        }

    def _generate_behavioral_insights(self, completions: List[Dict]) -> List[str]:
        """Generate behavioral insights based on data."""
        insights = []
        
        completion_rate = self._calculate_completion_rate(completions)
        
        if completion_rate > 80:
            insights.append("Excellent habit consistency! You're in the automation phase.")
        elif completion_rate > 60:
            insights.append("Good progress. Focus on consistency to reach automation.")
        elif completion_rate > 40:
            insights.append("Building momentum. Consider environmental changes to boost success.")
        else:
            insights.append("Early stages. Focus on making the habit as easy as possible.")
        
        # Add more specific insights based on patterns
        if len(completions) > 21:  # 21 days is common habit formation timeframe
            insights.append("You've passed the initial 21-day mark. Keep building neural pathways!")
        
        if len(completions) > 66:  # Average time for habit automation
            insights.append("Approaching the average 66-day automation point. You're building strong neural pathways!")
        
        return insights

    def _determine_formation_stage(self, completions: List[Dict]) -> str:
        """Determine what stage of habit formation the user is in."""
        if not completions:
            return "initiation"
        
        days_tracked = len(completions)
        completion_rate = self._calculate_completion_rate(completions)
        
        if days_tracked < 21:
            return "initiation"
        elif days_tracked < 66:
            return "development"
        elif completion_rate > 80:
            return "maintenance"
        else:
            return "development"

    def _get_science_based_tips(self) -> List[str]:
        """Get evidence-based habit formation tips."""
        return [
            "Start with a 2-minute version of your habit to build consistency",
            "Use implementation intentions: 'After I [current habit], I will [new habit]'",
            "Focus on identity change: 'I am someone who [does this habit]'",
            "Make the habit obvious, attractive, easy, and satisfying",
            "Use habit stacking to link new habits to established ones",
            "Environment design matters more than willpower",
            "Celebrate small wins to activate reward pathways",
            "Expect setbacks and plan for recovery strategies",
        ]

    def _generate_personalized_insights(self, habits: List[Dict], completions: List[Dict]) -> List[str]:
        """Generate personalized insights based on user data."""
        insights = []
        
        if not completions:
            insights.append("Start tracking your habit completions to get personalized insights")
            return insights
        
        completion_rate = self._calculate_completion_rate(completions)
        
        if completion_rate > 80:
            insights.append("You're doing great! Your habit is becoming automatic.")
        elif completion_rate > 60:
            insights.append("You're building good momentum. Focus on consistency.")
        else:
            insights.append("Consider simplifying your habit to build consistency first.")
        
        # Add insights based on patterns
        streaks = self._analyze_streaks(completions)
        if streaks["current_streak"] > 7:
            insights.append("Your current streak shows strong commitment!")
        
        return insights

    def _explain_motivation_psychology(self) -> Dict[str, str]:
        """Explain the psychology behind habit motivation."""
        return {
            "intrinsic_motivation": "Focus on internal satisfaction and personal growth rather than external rewards",
            "dopamine_loops": "Small, frequent rewards help build stronger habit loops than large, infrequent ones",
            "identity_alignment": "Habits stick better when they align with your self-identity and values",
            "progress_visualization": "Seeing progress visually activates motivation and reward centers in the brain",
        }

    def _suggest_advanced_strategies(self, completions: List[Dict]) -> List[str]:
        """Suggest advanced habit formation strategies."""
        strategies = []
        
        completion_rate = self._calculate_completion_rate(completions)
        
        if completion_rate > 70:
            strategies.extend([
                "Try habit laddering to build related positive behaviors",
                "Experiment with keystone habits that trigger other positive changes",
                "Consider helping others develop similar habits for social reinforcement",
            ])
        else:
            strategies.extend([
                "Use temptation bundling: pair the habit with something you enjoy",
                "Create environmental constraints that make the habit easier",
                "Use social accountability and public commitments",
            ])
        
        strategies.extend([
            "Practice habit rehearsal and mental visualization",
            "Use the two-day rule: never miss twice in a row",
            "Track leading indicators, not just completion",
        ])
        
        return strategies

    def _generate_habit_optimizations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate specific habit optimizations based on patterns."""
        optimizations = []
        
        completion_rate = patterns.get("completion_rate", 0)
        
        if completion_rate < 50:
            optimizations.extend([
                "Reduce habit difficulty to build consistency first",
                "Focus on habit cues and environmental triggers",
                "Implement habit stacking with existing routines",
            ])
        elif completion_rate < 80:
            optimizations.extend([
                "Refine habit timing and context",
                "Add accountability measures",
                "Enhance reward systems",
            ])
        else:
            optimizations.extend([
                "Consider expanding or deepening the habit",
                "Add variation to prevent boredom",
                "Focus on maintaining long-term consistency",
            ])
        
        return optimizations

    def _suggest_environmental_changes(self, patterns: Dict[str, Any]) -> List[str]:
        """Suggest environmental changes to support habit formation."""
        return [
            "Optimize your physical environment to make the habit easier",
            "Remove obstacles and friction from your habit routine",
            "Create visual cues and reminders in your environment",
            "Design your space to support habit consistency",
            "Consider social environment changes for accountability",
        ]

    def _suggest_motivation_strategies(self, patterns: Dict[str, Any]) -> List[str]:
        """Suggest motivation strategies based on patterns."""
        return [
            "Connect habits to deeper values and long-term goals",
            "Use progress tracking and celebration of small wins",
            "Create a personal reward system for consistency",
            "Visualize the compound benefits of habit consistency",
            "Focus on identity change rather than outcome change",
        ]

    def _suggest_schedule_adjustments(self, patterns: Dict[str, Any]) -> List[str]:
        """Suggest schedule adjustments based on time patterns."""
        time_patterns = patterns.get("time_patterns", {})
        best_day = time_patterns.get("best_day", "")
        
        suggestions = [
            "Consider adjusting habit timing to your highest energy periods",
            "Build habits around existing consistent routines",
            "Use time blocking to protect habit practice time",
        ]
        
        if best_day:
            suggestions.append(f"Your {best_day} success suggests this timing works well")
        
        return suggestions

    def _suggest_accountability_measures(self, patterns: Dict[str, Any]) -> List[str]:
        """Suggest accountability measures to improve consistency."""
        return [
            "Find an accountability partner for regular check-ins",
            "Join a community or group with similar habits",
            "Use public commitment and social pressure positively",
            "Implement habit tracking tools and regular review",
            "Create consequences for breaking the habit chain",
        ]


class StudyMotivationCoachSkill(BaseSkill):
    """Skill for providing personalized study motivation and learning optimization."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="coaching.study_motivation_coach",
            name="Study Motivation Coach",
            description="Provides personalized study motivation, learning strategies, and academic coaching support",
            category=SkillCategory.COACHING,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (assess_learning_style, create_study_plan, motivate, track_study_progress, optimize_techniques)",
                    "required": True,
                },
                "user_id": {"type": "string", "description": "User identifier", "required": True},
                "subject": {
                    "type": "string",
                    "description": "Subject or topic being studied",
                    "required": False,
                },
                "study_data": {
                    "type": "object",
                    "description": "Study session data and progress",
                    "required": False,
                },
                "motivation_request": {
                    "type": "string",
                    "description": "Specific motivation or guidance needed",
                    "required": False,
                },
            },
            examples=[
                {"operation": "assess_learning_style", "user_id": "user_123"},
                {"operation": "create_study_plan", "user_id": "user_123", "subject": "Python Programming"},
                {"operation": "motivate", "user_id": "user_123", "motivation_request": "feeling overwhelmed with exam prep"},
            ],
            required_components=["MemoryManager", "ModelRouter", "PreferenceLearner"],
            tags={"study", "motivation", "learning", "academic_coaching"},
            is_stateful=True,
        )

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """Execute the study motivation coach skill."""
        try:
            operation = input_data.get("operation")
            user_id = input_data.get("user_id")

            if not operation or not user_id:
                return SkillResult(
                    success=False,
                    data={},
                    message="Missing required parameters: operation and user_id",
                    errors=["operation and user_id are required"],
                )

            if operation == "assess_learning_style":
                return await self._assess_learning_style(user_id)
            elif operation == "create_study_plan":
                return await self._create_study_plan(user_id, input_data)
            elif operation == "motivate":
                return await self._provide_motivation(user_id, input_data.get("motivation_request", ""))
            elif operation == "track_study_progress":
                return await self._track_study_progress(user_id, input_data.get("study_data", {}))
            elif operation == "optimize_techniques":
                return await self._optimize_study_techniques(user_id)
            else:
                return SkillResult(
                    success=False,
                    data={},
                    message=f"Unknown operation: {operation}",
                    errors=[f"Unsupported operation: {operation}"],
                )

        except Exception as e:
            self.logger.error(f"Error in StudyMotivationCoachSkill: {str(e)}")
            return SkillResult(
                success=False,
                data={},
                message="Failed to execute study motivation coach",
                errors=[str(e)],
            )

    async def _assess_learning_style(self, user_id: str) -> SkillResult:
        """Assess user's learning style and preferences."""
        try:
            # Retrieve user's learning history
            study_memories = await self.memory_manager.get_user_memories(user_id, limit=30)
            
            # Analyze learning patterns
            learning_patterns = self._analyze_learning_patterns(study_memories)
            
            assessment = {
                "primary_learning_style": learning_patterns.get("dominant_style", "multimodal"),
                "learning_preferences": learning_patterns.get("preferences", {}),
                "optimal_study_times": learning_patterns.get("peak_times", ["morning"]),
                "attention_span": learning_patterns.get("attention_span", "medium"),
                "preferred_study_methods": learning_patterns.get("methods", []),
                "learning_challenges": learning_patterns.get("challenges", []),
                "strengths": learning_patterns.get("strengths", []),
                "recommendations": self._generate_learning_recommendations(learning_patterns),
            }

            # Store assessment
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=assessment,
                memory_type="learning_assessment",
                metadata={"assessment_date": datetime.now(timezone.utc).isoformat()},
            )

            return SkillResult(
                success=True,
                data=assessment,
                message="Learning style assessment completed",
                confidence=0.85,
                next_actions=["create_study_plan", "optimize_techniques"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to assess learning style",
                errors=[str(e)],
            )

    async def _create_study_plan(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Create a personalized study plan."""
        try:
            subject = input_data.get("subject", "General Study")
            
            # Get learning assessment if available
            assessments = await self.memory_manager.get_user_memories(user_id, memory_type="learning_assessment")
            learning_style = assessments[0] if assessments else {}
            
            study_plan = {
                "plan_id": str(uuid.uuid4()),
                "subject": subject,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "study_schedule": self._generate_study_schedule(learning_style, subject),
                "learning_objectives": self._generate_learning_objectives(subject),
                "study_techniques": self._recommend_study_techniques(learning_style, subject),
                "assessment_strategy": self._create_assessment_strategy(subject),
                "motivation_milestones": self._create_motivation_milestones(subject),
                "resources": self._suggest_study_resources(subject),
                "progress_tracking": self._setup_progress_tracking(subject),
            }

            # Store study plan
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=study_plan,
                memory_type="study_plan",
                metadata={"subject": subject, "plan_id": study_plan["plan_id"]},
            )

            return SkillResult(
                success=True,
                data=study_plan,
                message=f"Study plan for {subject} created successfully",
                confidence=0.9,
                next_actions=["track_study_progress", "motivate"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to create study plan",
                errors=[str(e)],
            )

    async def _provide_motivation(self, user_id: str, motivation_request: str) -> SkillResult:
        """Provide personalized study motivation."""
        try:
            # Get study context and progress
            study_plans = await self.memory_manager.get_user_memories(user_id, memory_type="study_plan")
            study_progress = await self.memory_manager.get_user_memories(user_id, memory_type="study_progress")
            
            motivation_type = self._determine_motivation_type(motivation_request)
            
            motivation_response = {
                "motivation_type": motivation_type,
                "personalized_message": self._generate_motivational_message(motivation_type, study_progress),
                "actionable_steps": self._suggest_immediate_actions(motivation_type, motivation_request),
                "mindset_shifts": self._suggest_mindset_shifts(motivation_type),
                "study_techniques": self._recommend_motivation_techniques(motivation_type),
                "success_reminders": self._generate_success_reminders(study_progress),
                "goal_refocus": self._help_refocus_goals(study_plans, motivation_request),
                "provided_at": datetime.now(timezone.utc).isoformat(),
            }

            return SkillResult(
                success=True,
                data=motivation_response,
                message="Study motivation provided successfully",
                confidence=0.9,
                next_actions=["track_study_progress"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to provide motivation",
                errors=[str(e)],
            )

    async def _track_study_progress(self, user_id: str, study_data: Dict[str, Any]) -> SkillResult:
        """Track study session progress and performance."""
        try:
            progress_entry = {
                "session_id": str(uuid.uuid4()),
                "date": datetime.now(timezone.utc).isoformat(),
                "subject": study_data.get("subject", ""),
                "duration_minutes": study_data.get("duration_minutes", 0),
                "topics_covered": study_data.get("topics_covered", []),
                "difficulty_rating": study_data.get("difficulty_rating", 5),
                "comprehension_level": study_data.get("comprehension_level", 5),
                "focus_level": study_data.get("focus_level", 5),
                "techniques_used": study_data.get("techniques_used", []),
                "challenges_faced": study_data.get("challenges_faced", []),
                "achievements": study_data.get("achievements", []),
                "notes": study_data.get("notes", ""),
            }

            # Store progress
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=progress_entry,
                memory_type="study_progress",
                metadata={"session_date": progress_entry["date"], "subject": progress_entry["subject"]},
            )

            # Analyze progress trends
            all_progress = await self.memory_manager.get_user_memories(user_id, memory_type="study_progress")
            progress_analysis = self._analyze_study_progress(all_progress)

            return SkillResult(
                success=True,
                data={
                    "session_recorded": progress_entry,
                    "progress_analysis": progress_analysis,
                },
                message="Study progress tracked successfully",
                confidence=0.9,
                next_actions=["optimize_techniques", "motivate"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to track study progress",
                errors=[str(e)],
            )

    async def _optimize_study_techniques(self, user_id: str) -> SkillResult:
        """Optimize study techniques based on progress and effectiveness."""
        try:
            # Get study progress and learning assessment
            progress_data = await self.memory_manager.get_user_memories(user_id, memory_type="study_progress")
            assessments = await self.memory_manager.get_user_memories(user_id, memory_type="learning_assessment")
            
            if not progress_data:
                return SkillResult(
                    success=False,
                    data={},
                    message="No study progress data found for optimization",
                    errors=["Insufficient data for technique optimization"],
                )

            optimization_analysis = {
                "technique_effectiveness": self._analyze_technique_effectiveness(progress_data),
                "optimal_study_duration": self._determine_optimal_duration(progress_data),
                "best_study_times": self._identify_peak_performance_times(progress_data),
                "recommended_techniques": self._recommend_optimized_techniques(progress_data, assessments),
                "areas_for_improvement": self._identify_improvement_areas(progress_data),
                "personalized_strategies": self._generate_personalized_strategies(progress_data, assessments),
                "next_level_techniques": self._suggest_advanced_techniques(progress_data),
            }

            return SkillResult(
                success=True,
                data=optimization_analysis,
                message="Study technique optimization completed",
                confidence=0.85,
                next_actions=["create_study_plan", "track_study_progress"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to optimize study techniques",
                errors=[str(e)],
            )

    def _analyze_learning_patterns(self, study_memories: List[Dict]) -> Dict[str, Any]:
        """Analyze user's learning patterns from memory."""
        patterns = {
            "dominant_style": "visual",  # Default
            "preferences": {},
            "peak_times": ["morning"],
            "attention_span": "medium",
            "methods": ["reading", "note_taking"],
            "challenges": [],
            "strengths": [],
        }
        
        if not study_memories:
            return patterns
        
        # Analyze study sessions for patterns
        time_preferences = defaultdict(int)
        method_effectiveness = defaultdict(list)
        focus_levels = []
        
        for memory in study_memories:
            if isinstance(memory, dict):
                # Extract time patterns
                if "date" in memory:
                    try:
                        date_obj = datetime.fromisoformat(memory["date"].replace('Z', '+00:00'))
                        hour = date_obj.hour
                        if 6 <= hour < 12:
                            time_preferences["morning"] += 1
                        elif 12 <= hour < 18:
                            time_preferences["afternoon"] += 1
                        else:
                            time_preferences["evening"] += 1
                    except:
                        pass
                
                # Extract method effectiveness
                if "techniques_used" in memory and "comprehension_level" in memory:
                    for technique in memory["techniques_used"]:
                        method_effectiveness[technique].append(memory["comprehension_level"])
                
                # Extract focus levels
                if "focus_level" in memory:
                    focus_levels.append(memory["focus_level"])
        
        # Determine peak times
        if time_preferences:
            patterns["peak_times"] = [max(time_preferences, key=time_preferences.get)]
        
        # Determine attention span
        if focus_levels:
            avg_focus = sum(focus_levels) / len(focus_levels)
            if avg_focus > 7:
                patterns["attention_span"] = "high"
            elif avg_focus > 5:
                patterns["attention_span"] = "medium"
            else:
                patterns["attention_span"] = "low"
        
        # Determine effective methods
        effective_methods = []
        for method, scores in method_effectiveness.items():
            if scores and sum(scores) / len(scores) > 6:
                effective_methods.append(method)
        
        if effective_methods:
            patterns["methods"] = effective_methods
        
        return patterns

    def _generate_learning_recommendations(self, learning_patterns: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on patterns."""
        recommendations = []
        
        peak_times = learning_patterns.get("peak_times", ["morning"])
        attention_span = learning_patterns.get("attention_span", "medium")
        
        # Time-based recommendations
        if "morning" in peak_times:
            recommendations.append("Schedule demanding subjects during morning hours")
        if "evening" in peak_times:
            recommendations.append("Use evening sessions for review and practice")
        
        # Attention span recommendations
        if attention_span == "high":
            recommendations.append("Take advantage of long focus sessions for complex topics")
        elif attention_span == "low":
            recommendations.append("Use the Pomodoro Technique with 25-minute focused sessions")
        else:
            recommendations.append("Alternate between 45-minute study blocks and 15-minute breaks")
        
        recommendations.extend([
            "Vary study techniques to maintain engagement",
            "Create a consistent study environment",
            "Use active recall and spaced repetition",
            "Connect new information to existing knowledge",
        ])
        
        return recommendations

    def _generate_study_schedule(self, learning_style: Dict[str, Any], subject: str) -> Dict[str, Any]:
        """Generate a personalized study schedule."""
        peak_times = learning_style.get("optimal_study_times", ["morning"])
        attention_span = learning_style.get("attention_span", "medium")
        
        # Determine session length
        if attention_span == "high":
            session_length = 90
            break_length = 20
        elif attention_span == "low":
            session_length = 25
            break_length = 5
        else:
            session_length = 45
            break_length = 15
        
        schedule = {
            "daily_sessions": 2,
            "session_length_minutes": session_length,
            "break_length_minutes": break_length,
            "optimal_times": peak_times,
            "weekly_hours": 10,
            "rest_days": 1,
            "review_frequency": "daily",
            "assessment_frequency": "weekly",
        }
        
        return schedule

    def _generate_learning_objectives(self, subject: str) -> List[Dict[str, Any]]:
        """Generate learning objectives for the subject."""
        base_objectives = [
            {
                "objective": f"Understand fundamental concepts of {subject}",
                "timeline": "Week 1-2",
                "assessment": "Self-quiz and concept mapping",
            },
            {
                "objective": f"Apply {subject} principles to practical problems",
                "timeline": "Week 3-4",
                "assessment": "Practice exercises and projects",
            },
            {
                "objective": f"Analyze and evaluate {subject} applications",
                "timeline": "Week 5-6",
                "assessment": "Case studies and critical analysis",
            },
            {
                "objective": f"Synthesize and create using {subject} knowledge",
                "timeline": "Week 7-8",
                "assessment": "Original project or presentation",
            },
        ]
        
        return base_objectives

    def _recommend_study_techniques(self, learning_style: Dict[str, Any], subject: str) -> List[str]:
        """Recommend study techniques based on learning style and subject."""
        techniques = [
            "Active recall and self-testing",
            "Spaced repetition system",
            "Elaborative interrogation (asking why/how)",
            "Dual coding (visual and verbal)",
            "Interleaving different topics",
        ]
        
        # Add subject-specific techniques
        subject_lower = subject.lower()
        
        if any(word in subject_lower for word in ["math", "statistics", "physics"]):
            techniques.extend([
                "Problem-solving practice with worked examples",
                "Formula derivation and application",
                "Mathematical visualization and graphing",
            ])
        elif any(word in subject_lower for word in ["history", "literature", "social"]):
            techniques.extend([
                "Timeline creation and historical connections",
                "Narrative storytelling and character analysis",
                "Comparative analysis and critical thinking",
            ])
        elif any(word in subject_lower for word in ["programming", "coding", "computer"]):
            techniques.extend([
                "Hands-on coding practice and debugging",
                "Code reading and algorithm analysis",
                "Project-based learning and portfolio building",
            ])
        
        return techniques

    def _create_assessment_strategy(self, subject: str) -> Dict[str, Any]:
        """Create assessment strategy for the subject."""
        return {
            "formative_assessments": [
                "Daily self-quizzes",
                "Weekly practice tests",
                "Peer explanations",
                "Progress check-ins",
            ],
            "summative_assessments": [
                "Monthly comprehensive tests",
                "Project evaluations",
                "Practical applications",
                "Portfolio reviews",
            ],
            "self_reflection": [
                "Learning journal entries",
                "Difficulty ratings",
                "Strategy effectiveness reviews",
                "Goal adjustment sessions",
            ],
        }

    def _create_motivation_milestones(self, subject: str) -> List[Dict[str, Any]]:
        """Create motivation milestones for the subject."""
        return [
            {
                "milestone": "First week completion",
                "reward": "Celebrate establishing study routine",
                "timing": "Week 1",
            },
            {
                "milestone": "First concept mastery",
                "reward": "Treat yourself to something special",
                "timing": "Week 2-3",
            },
            {
                "milestone": "Mid-point progress",
                "reward": "Share achievement with friends/family",
                "timing": "Week 4",
            },
            {
                "milestone": "Advanced skill demonstration",
                "reward": "Plan a learning celebration",
                "timing": "Week 6-7",
            },
            {
                "milestone": "Course/goal completion",
                "reward": "Major celebration and reflection",
                "timing": "Week 8",
            },
        ]

    def _suggest_study_resources(self, subject: str) -> List[str]:
        """Suggest study resources for the subject."""
        general_resources = [
            "Khan Academy for visual learning",
            "Coursera for structured courses",
            "YouTube educational channels",
            "Library textbooks and journals",
            "Study groups and peer networks",
        ]
        
        subject_lower = subject.lower()
        
        if any(word in subject_lower for word in ["programming", "coding"]):
            general_resources.extend([
                "LeetCode for coding practice",
                "GitHub for project examples",
                "Stack Overflow for problem solving",
                "Documentation and official tutorials",
            ])
        elif any(word in subject_lower for word in ["language", "spanish", "french"]):
            general_resources.extend([
                "Duolingo for daily practice",
                "Language exchange platforms",
                "Podcasts and audio content",
                "Native speaker conversation groups",
            ])
        
        return general_resources

    def _setup_progress_tracking(self, subject: str) -> Dict[str, Any]:
        """Setup progress tracking system for the subject."""
        return {
            "daily_metrics": [
                "Study time completed",
                "Focus level rating",
                "Comprehension assessment",
                "Energy level tracking",
            ],
            "weekly_metrics": [
                "Concepts mastered",
                "Practice problems completed",
                "Assessment scores",
                "Goal progress percentage",
            ],
            "tracking_tools": [
                "Study log template",
                "Progress visualization charts",
                "Habit tracking calendar",
                "Performance dashboard",
            ],
        }

    def _determine_motivation_type(self, motivation_request: str) -> str:
        """Determine the type of motivation needed."""
        request_lower = motivation_request.lower()
        
        if any(word in request_lower for word in ["overwhelmed", "stress", "anxiety"]):
            return "stress_management"
        elif any(word in request_lower for word in ["procrastination", "delay", "avoid"]):
            return "procrastination_help"
        elif any(word in request_lower for word in ["boring", "uninteresting", "dull"]):
            return "engagement_boost"
        elif any(word in request_lower for word in ["difficult", "hard", "struggle"]):
            return "difficulty_support"
        elif any(word in request_lower for word in ["goal", "purpose", "why"]):
            return "goal_clarification"
        else:
            return "general_motivation"

    def _generate_motivational_message(self, motivation_type: str, study_progress: List[Dict]) -> str:
        """Generate personalized motivational message."""
        messages = {
            "stress_management": (
                "Feeling overwhelmed is normal when learning challenging material. "
                "Break your study goals into smaller, manageable chunks. "
                "Remember, progress happens one step at a time, and you're building valuable skills."
            ),
            "procrastination_help": (
                "Procrastination often comes from perfectionism or fear of failure. "
                "Start with just 5 minutes of study - momentum builds naturally. "
                "Your future self will thank you for taking action today."
            ),
            "engagement_boost": (
                "Learning becomes exciting when you connect it to your interests and goals. "
                "Try to find real-world applications or creative ways to explore the material. "
                "Curiosity is your best learning tool."
            ),
            "difficulty_support": (
                "Challenging material means you're growing your brain and building resilience. "
                "Every expert was once a beginner who kept going despite difficulties. "
                "Your persistence is developing both knowledge and character."
            ),
            "goal_clarification": (
                "Clear goals give direction and meaning to your efforts. "
                "Visualize how mastering this subject will impact your future. "
                "Connect your learning to your values and aspirations."
            ),
            "general_motivation": (
                "You've chosen to invest in yourself through learning - that's admirable. "
                "Every study session makes you more knowledgeable and capable. "
                "Keep building on your progress, one day at a time."
            ),
        }
        
        base_message = messages.get(motivation_type, messages["general_motivation"])
        
        # Add progress-specific encouragement if available
        if study_progress:
            progress_encouragement = (
                f" You've already completed {len(study_progress)} study sessions - "
                "that shows real commitment to your goals!"
            )
            return base_message + progress_encouragement
        
        return base_message

    def _suggest_immediate_actions(self, motivation_type: str, motivation_request: str) -> List[str]:
        """Suggest immediate actionable steps based on motivation type."""
        actions = {
            "stress_management": [
                "Take 5 deep breaths and remind yourself this is temporary",
                "Write down your specific concerns to clarify them",
                "Break your next study goal into 3 smaller steps",
                "Schedule a 10-minute break after 25 minutes of study",
            ],
            "procrastination_help": [
                "Commit to studying for just 2 minutes right now",
                "Remove distractions from your study space",
                "Set a specific time for your next study session",
                "Tell someone about your intention to study today",
            ],
            "engagement_boost": [
                "Find one interesting fact about your subject to share",
                "Connect today's study topic to something you enjoy",
                "Change your study location or method",
                "Set a fun reward for completing today's study goal",
            ],
            "difficulty_support": [
                "Identify the specific part you find most challenging",
                "Look for alternative explanations or examples online",
                "Reach out to a study partner or teacher for help",
                "Review easier material first to build confidence",
            ],
            "goal_clarification": [
                "Write down 3 reasons why this subject matters to you",
                "Visualize yourself successfully using this knowledge",
                "Connect this learning to your career or personal goals",
                "Share your learning journey with someone who supports you",
            ],
            "general_motivation": [
                "Review your recent progress and celebrate small wins",
                "Set one specific, achievable goal for today",
                "Remind yourself of your bigger learning objectives",
                "Do one small study task to build momentum",
            ],
        }
        
        return actions.get(motivation_type, actions["general_motivation"])

    def _suggest_mindset_shifts(self, motivation_type: str) -> List[str]:
        """Suggest helpful mindset shifts."""
        mindset_shifts = {
            "stress_management": [
                "From 'I must be perfect' to 'Progress over perfection'",
                "From 'I should know this already' to 'Learning takes time'",
                "From 'This is too much' to 'I can handle one step at a time'",
            ],
            "procrastination_help": [
                "From 'I need to feel motivated first' to 'Action creates motivation'",
                "From 'It has to be perfect' to 'Done is better than perfect'",
                "From 'I'll start tomorrow' to 'I can start small right now'",
            ],
            "engagement_boost": [
                "From 'This is boring' to 'How can I make this interesting?'",
                "From 'I have to study this' to 'I get to learn something new'",
                "From 'This isn't relevant' to 'How does this connect to my goals?'",
            ],
            "difficulty_support": [
                "From 'I'm not smart enough' to 'My brain is growing through challenge'",
                "From 'This is impossible' to 'This requires new strategies'",
                "From 'I should give up' to 'Difficulty means I'm learning'",
            ],
            "goal_clarification": [
                "From 'I don't know why I'm doing this' to 'This serves my bigger purpose'",
                "From 'Learning is a chore' to 'Learning is an investment in myself'",
                "From 'I'm behind others' to 'I'm on my own unique learning journey'",
            ],
            "general_motivation": [
                "From 'I don't feel like it' to 'I'll feel great after I start'",
                "From 'This is hard work' to 'This is building my capabilities'",
                "From 'I might fail' to 'I'm building resilience and knowledge'",
            ],
        }
        
        return mindset_shifts.get(motivation_type, mindset_shifts["general_motivation"])

    def _recommend_motivation_techniques(self, motivation_type: str) -> List[str]:
        """Recommend specific motivation techniques."""
        techniques = {
            "stress_management": [
                "Progressive muscle relaxation before study sessions",
                "Breathing exercises during breaks",
                "Positive self-talk and affirmations",
                "Time-blocking to manage workload",
            ],
            "procrastination_help": [
                "The 2-minute rule: if it takes less than 2 minutes, do it now",
                "Implementation intentions: 'After X, I will study Y'",
                "Temptation bundling: pair study with something enjoyable",
                "Public commitment and accountability partners",
            ],
            "engagement_boost": [
                "Gamification: create points and rewards for study tasks",
                "Variety: rotate between different study methods",
                "Social learning: study with others or teach someone else",
                "Real-world applications: find practical uses for knowledge",
            ],
            "difficulty_support": [
                "Scaffolding: break complex topics into simpler parts",
                "Multiple resources: use different explanations and examples",
                "Peer support: join study groups or online communities",
                "Growth mindset mantras: 'I can't do this YET'",
            ],
            "goal_clarification": [
                "Vision boarding: visualize your goals and achievements",
                "Values alignment: connect learning to personal values",
                "Future self meditation: imagine your successful future",
                "Progress tracking: celebrate milestones and improvements",
            ],
            "general_motivation": [
                "Regular progress reviews and celebrations",
                "Inspiring content: books, videos, or podcasts about learning",
                "Social support: share goals with friends and family",
                "Environmental design: create an inspiring study space",
            ],
        }
        
        return techniques.get(motivation_type, techniques["general_motivation"])

    def _generate_success_reminders(self, study_progress: List[Dict]) -> List[str]:
        """Generate reminders of past successes and progress."""
        if not study_progress:
            return [
                "Every expert was once a beginner",
                "You've learned complex things before",
                "Your brain is designed for learning and growth",
            ]
        
        total_sessions = len(study_progress)
        total_hours = sum(session.get("duration_minutes", 0) for session in study_progress) / 60
        
        reminders = [
            f"You've completed {total_sessions} study sessions - that's dedication!",
            f"You've invested {total_hours:.1f} hours in your learning - that's real commitment!",
            "Look how far you've come since you started",
            "Each study session makes you more knowledgeable",
        ]
        
        # Add specific achievements if available
        if study_progress:
            latest_session = study_progress[0]
            if latest_session.get("achievements"):
                reminders.append(f"Recent achievement: {latest_session['achievements'][0]}")
        
        return reminders

    def _help_refocus_goals(self, study_plans: List[Dict], motivation_request: str) -> Dict[str, Any]:
        """Help user refocus on their goals."""
        if not study_plans:
            return {
                "suggestion": "Take time to clarify your learning goals",
                "action": "Write down why this subject matters to you",
                "focus": "Connect learning to your values and aspirations",
            }
        
        current_plan = study_plans[0]
        subject = current_plan.get("subject", "your studies")
        
        return {
            "reminder": f"You're learning {subject} for important reasons",
            "objectives": current_plan.get("learning_objectives", [])[:2],
            "progress": "Every study session moves you closer to your goals",
            "refocus_action": "Review your learning objectives and visualize success",
        }

    def _analyze_study_progress(self, progress_data: List[Dict]) -> Dict[str, Any]:
        """Analyze study progress patterns and trends."""
        if not progress_data:
            return {"message": "Start tracking your study sessions to see progress analysis"}
        
        # Calculate metrics
        total_sessions = len(progress_data)
        total_minutes = sum(session.get("duration_minutes", 0) for session in progress_data)
        avg_focus = sum(session.get("focus_level", 5) for session in progress_data) / total_sessions
        avg_comprehension = sum(session.get("comprehension_level", 5) for session in progress_data) / total_sessions
        
        # Analyze trends (simple linear trend)
        recent_sessions = progress_data[:5] if len(progress_data) >= 5 else progress_data
        recent_focus = sum(session.get("focus_level", 5) for session in recent_sessions) / len(recent_sessions)
        
        trend = "stable"
        if len(progress_data) >= 5:
            early_focus = sum(session.get("focus_level", 5) for session in progress_data[-5:]) / 5
            if recent_focus > early_focus + 0.5:
                trend = "improving"
            elif recent_focus < early_focus - 0.5:
                trend = "declining"
        
        return {
            "total_sessions": total_sessions,
            "total_study_hours": total_minutes / 60,
            "average_focus_level": round(avg_focus, 1),
            "average_comprehension": round(avg_comprehension, 1),
            "focus_trend": trend,
            "consistency": "excellent" if total_sessions >= 10 else "building" if total_sessions >= 5 else "starting",
            "insights": self._generate_progress_insights(total_sessions, avg_focus, avg_comprehension, trend),
        }

    def _generate_progress_insights(self, total_sessions: int, avg_focus: float, avg_comprehension: float, trend: str) -> List[str]:
        """Generate insights based on progress analysis."""
        insights = []
        
        if total_sessions >= 20:
            insights.append("You've built a strong study habit! Consistency is key to mastery.")
        elif total_sessions >= 10:
            insights.append("Great progress on building consistent study habits.")
        else:
            insights.append("Keep building your study routine - consistency leads to success.")
        
        if avg_focus >= 7:
            insights.append("Excellent focus levels! Your concentration skills are strong.")
        elif avg_focus >= 5:
            insights.append("Good focus levels. Consider techniques to enhance concentration.")
        else:
            insights.append("Focus is an area for improvement. Try shorter sessions with breaks.")
        
        if avg_comprehension >= 7:
            insights.append("Strong comprehension scores indicate effective learning.")
        elif avg_comprehension >= 5:
            insights.append("Solid understanding. Consider varying study techniques.")
        else:
            insights.append("Consider adjusting study methods to improve comprehension.")
        
        if trend == "improving":
            insights.append("Your recent sessions show improvement - keep it up!")
        elif trend == "declining":
            insights.append("Recent sessions suggest you might need a strategy adjustment.")
        
        return insights

    def _analyze_technique_effectiveness(self, progress_data: List[Dict]) -> Dict[str, float]:
        """Analyze effectiveness of different study techniques."""
        technique_scores = defaultdict(list)
        
        for session in progress_data:
            techniques = session.get("techniques_used", [])
            comprehension = session.get("comprehension_level", 5)
            
            for technique in techniques:
                technique_scores[technique].append(comprehension)
        
        effectiveness = {}
        for technique, scores in technique_scores.items():
            if scores:
                effectiveness[technique] = sum(scores) / len(scores)
        
        return effectiveness

    def _determine_optimal_duration(self, progress_data: List[Dict]) -> Dict[str, Any]:
        """Determine optimal study session duration."""
        duration_performance = defaultdict(list)
        
        for session in progress_data:
            duration = session.get("duration_minutes", 0)
            focus = session.get("focus_level", 5)
            comprehension = session.get("comprehension_level", 5)
            
            # Group into duration ranges
            if duration <= 30:
                duration_range = "short (≤30 min)"
            elif duration <= 60:
                duration_range = "medium (31-60 min)"
            else:
                duration_range = "long (>60 min)"
            
            duration_performance[duration_range].append((focus + comprehension) / 2)
        
        optimal_range = "medium (31-60 min)"  # Default
        best_score = 0
        
        for duration_range, scores in duration_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    optimal_range = duration_range
        
        return {
            "optimal_range": optimal_range,
            "performance_by_duration": {k: sum(v)/len(v) if v else 0 for k, v in duration_performance.items()},
            "recommendation": f"Your optimal study duration appears to be {optimal_range}",
        }

    def _identify_peak_performance_times(self, progress_data: List[Dict]) -> Dict[str, Any]:
        """Identify when the user performs best."""
        time_performance = defaultdict(list)
        
        for session in progress_data:
            try:
                date_str = session.get("date", "")
                if date_str:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    hour = date_obj.hour
                    
                    if 6 <= hour < 12:
                        time_period = "morning"
                    elif 12 <= hour < 18:
                        time_period = "afternoon"
                    else:
                        time_period = "evening"
                    
                    focus = session.get("focus_level", 5)
                    comprehension = session.get("comprehension_level", 5)
                    time_performance[time_period].append((focus + comprehension) / 2)
            except:
                continue
        
        peak_time = "morning"  # Default
        best_score = 0
        
        for time_period, scores in time_performance.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    peak_time = time_period
        
        return {
            "peak_time": peak_time,
            "performance_by_time": {k: sum(v)/len(v) if v else 0 for k, v in time_performance.items()},
            "recommendation": f"Schedule your most challenging study sessions in the {peak_time}",
        }

    def _recommend_optimized_techniques(self, progress_data: List[Dict], assessments: List[Dict]) -> List[str]:
        """Recommend optimized study techniques based on data."""
        effectiveness = self._analyze_technique_effectiveness(progress_data)
        
        # Get top performing techniques
        top_techniques = sorted(effectiveness.items(), key=lambda x: x[1], reverse=True)[:3]
        
        recommendations = []
        
        if top_techniques:
            recommendations.append(f"Continue using {top_techniques[0][0]} - it's your most effective technique")
            if len(top_techniques) > 1:
                recommendations.append(f"Also prioritize {top_techniques[1][0]} for good results")
        
        # Add general optimized techniques
        recommendations.extend([
            "Implement spaced repetition for long-term retention",
            "Use active recall testing instead of passive review",
            "Combine visual and verbal learning (dual coding)",
            "Practice interleaving different topics in sessions",
        ])
        
        return recommendations

    def _identify_improvement_areas(self, progress_data: List[Dict]) -> List[str]:
        """Identify areas that need improvement."""
        improvements = []
        
        if not progress_data:
            return ["Start tracking study sessions to identify improvement areas"]
        
        avg_focus = sum(session.get("focus_level", 5) for session in progress_data) / len(progress_data)
        avg_comprehension = sum(session.get("comprehension_level", 5) for session in progress_data) / len(progress_data)
        
        if avg_focus < 6:
            improvements.append("Focus and concentration during study sessions")
        
        if avg_comprehension < 6:
            improvements.append("Comprehension and understanding of material")
        
        # Check for consistency
        recent_sessions = len([s for s in progress_data if 
            (datetime.now(timezone.utc) - datetime.fromisoformat(s.get("date", "").replace('Z', '+00:00'))).days <= 7
        ]) if progress_data else 0
        
        if recent_sessions < 3:
            improvements.append("Consistency in study schedule and routine")
        
        # Check for variety in techniques
        all_techniques = set()
        for session in progress_data:
            all_techniques.update(session.get("techniques_used", []))
        
        if len(all_techniques) < 3:
            improvements.append("Variety in study techniques and methods")
        
        return improvements if improvements else ["Keep up the great work! All areas are performing well."]

    def _generate_personalized_strategies(self, progress_data: List[Dict], assessments: List[Dict]) -> List[str]:
        """Generate personalized improvement strategies."""
        strategies = []
        
        # Based on focus levels
        if progress_data:
            avg_focus = sum(session.get("focus_level", 5) for session in progress_data) / len(progress_data)
            
            if avg_focus < 6:
                strategies.extend([
                    "Try the Pomodoro Technique with 25-minute focused sessions",
                    "Eliminate distractions and create a dedicated study space",
                    "Practice mindfulness meditation to improve concentration",
                ])
            else:
                strategies.extend([
                    "Experiment with longer study sessions to maximize focus",
                    "Use your high focus ability for the most challenging material",
                ])
        
        # Based on learning style assessment
        if assessments:
            learning_style = assessments[0]
            dominant_style = learning_style.get("primary_learning_style", "multimodal")
            
            if dominant_style == "visual":
                strategies.append("Create more diagrams, mind maps, and visual aids")
            elif dominant_style == "auditory":
                strategies.append("Use more discussion, audio content, and verbal repetition")
            elif dominant_style == "kinesthetic":
                strategies.append("Incorporate hands-on activities and movement into study")
        
        # General personalized strategies
        strategies.extend([
            "Track which techniques work best for you and use them more",
            "Adjust study timing to match your peak performance hours",
            "Set up rewards for achieving study milestones",
            "Regular review and adjustment of study strategies",
        ])
        
        return strategies

    def _suggest_advanced_techniques(self, progress_data: List[Dict]) -> List[str]:
        """Suggest advanced study techniques for experienced learners."""
        if len(progress_data) < 10:
            return ["Build consistency with basic techniques before advancing to complex methods"]
        
        return [
            "Feynman Technique: Explain concepts in simple terms to test understanding",
            "Method of Loci: Use spatial memory for complex information retention",
            "Deliberate Practice: Focus on specific weaknesses with immediate feedback",
            "Interleaved Practice: Mix different types of problems in single sessions",
            "Elaborative Encoding: Connect new information to existing knowledge networks",
            "Metacognitive Strategies: Think about how you think and learn",
            "Collaborative Learning: Teach others and learn from peer explanations",
            "Cross-Modal Learning: Engage multiple senses simultaneously",
        ]


class VirtualDebateCoachSkill(BaseSkill):
    """Skill for providing debate coaching, argument analysis, and persuasion training."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="coaching.virtual_debate_coach",
            name="Virtual Debate Coach",
            description="Provides debate coaching, argument analysis, logical reasoning training, and persuasion techniques",
            category=SkillCategory.COACHING,
            parameters={
                "operation": {
                    "type": "string",
                    "description": "Operation to perform (analyze_argument, coach_debate, practice_rebuttal, improve_rhetoric, assess_logic)",
                    "required": True,
                },
                "user_id": {"type": "string", "description": "User identifier", "required": True},
                "topic": {
                    "type": "string",
                    "description": "Debate topic or subject",
                    "required": False,
                },
                "argument": {
                    "type": "string",
                    "description": "Argument or position to analyze",
                    "required": False,
                },
                "stance": {
                    "type": "string",
                    "description": "Pro, con, or neutral stance",
                    "required": False,
                },
            },
            examples=[
                {"operation": "analyze_argument", "user_id": "user_123", "argument": "Climate change requires immediate action"},
                {"operation": "coach_debate", "user_id": "user_123", "topic": "Universal Basic Income"},
                {"operation": "practice_rebuttal", "user_id": "user_123", "argument": "Technology makes people lazy"},
            ],
            required_components=["MemoryManager", "ModelRouter"],
            tags={"debate", "argumentation", "logic", "rhetoric", "persuasion"},
            is_stateful=True,
        )

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """Execute the virtual debate coach skill."""
        try:
            operation = input_data.get("operation")
            user_id = input_data.get("user_id")

            if not operation or not user_id:
                return SkillResult(
                    success=False,
                    data={},
                    message="Missing required parameters: operation and user_id",
                    errors=["operation and user_id are required"],
                )

            if operation == "analyze_argument":
                return await self._analyze_argument(user_id, input_data)
            elif operation == "coach_debate":
                return await self._coach_debate(user_id, input_data)
            elif operation == "practice_rebuttal":
                return await self._practice_rebuttal(user_id, input_data)
            elif operation == "improve_rhetoric":
                return await self._improve_rhetoric(user_id, input_data)
            elif operation == "assess_logic":
                return await self._assess_logic(user_id, input_data)
            else:
                return SkillResult(
                    success=False,
                    data={},
                    message=f"Unknown operation: {operation}",
                    errors=[f"Unsupported operation: {operation}"],
                )

        except Exception as e:
            self.logger.error(f"Error in VirtualDebateCoachSkill: {str(e)}")
            return SkillResult(
                success=False,
                data={},
                message="Failed to execute virtual debate coach",
                errors=[str(e)],
            )

    async def _analyze_argument(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Analyze the structure and quality of an argument."""
        try:
            argument = input_data.get("argument", "")
            if not argument:
                return SkillResult(
                    success=False,
                    data={},
                    message="No argument provided for analysis",
                    errors=["argument parameter is required"],
                )

            analysis = {
                "argument_structure": self._analyze_argument_structure(argument),
                "logical_strength": self._assess_logical_strength(argument),
                "evidence_quality": self._evaluate_evidence(argument),
                "rhetorical_devices": self._identify_rhetorical_devices(argument),
                "logical_fallacies": self._detect_logical_fallacies(argument),
                "counterargument_vulnerabilities": self._find_vulnerabilities(argument),
                "improvement_suggestions": self._suggest_improvements(argument),
                "score": self._calculate_argument_score(argument),
            }

            # Store analysis for learning
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=analysis,
                memory_type="argument_analysis",
                metadata={"argument_snippet": argument[:100], "analysis_date": datetime.now(timezone.utc).isoformat()},
            )

            return SkillResult(
                success=True,
                data=analysis,
                message="Argument analysis completed",
                confidence=0.9,
                next_actions=["improve_rhetoric", "practice_rebuttal"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to analyze argument",
                errors=[str(e)],
            )

    async def _coach_debate(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Provide comprehensive debate coaching for a topic."""
        try:
            topic = input_data.get("topic", "")
            stance = input_data.get("stance", "pro")
            
            coaching = {
                "topic_analysis": self._analyze_debate_topic(topic),
                "key_arguments": self._generate_key_arguments(topic, stance),
                "counterarguments": self._anticipate_counterarguments(topic, stance),
                "evidence_sources": self._suggest_evidence_sources(topic),
                "rhetorical_strategies": self._recommend_rhetorical_strategies(topic, stance),
                "debate_structure": self._provide_debate_structure(),
                "practice_questions": self._generate_practice_questions(topic),
                "winning_techniques": self._share_winning_techniques(),
            }

            # Store coaching session
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                content=coaching,
                memory_type="debate_coaching",
                metadata={"topic": topic, "stance": stance, "coaching_date": datetime.now(timezone.utc).isoformat()},
            )

            return SkillResult(
                success=True,
                data=coaching,
                message=f"Debate coaching for '{topic}' completed",
                confidence=0.9,
                next_actions=["practice_rebuttal", "assess_logic"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to provide debate coaching",
                errors=[str(e)],
            )

    async def _practice_rebuttal(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Practice rebuttal techniques against opposing arguments."""
        try:
            opposing_argument = input_data.get("argument", "")
            
            rebuttal_training = {
                "opposing_argument": opposing_argument,
                "rebuttal_strategies": self._generate_rebuttal_strategies(opposing_argument),
                "counter_evidence": self._suggest_counter_evidence(opposing_argument),
                "logical_attacks": self._identify_logical_weaknesses(opposing_argument),
                "rhetorical_responses": self._craft_rhetorical_responses(opposing_argument),
                "practice_rebuttals": self._generate_practice_rebuttals(opposing_argument),
                "timing_strategies": self._provide_timing_strategies(),
                "delivery_tips": self._provide_delivery_tips(),
            }

            return SkillResult(
                success=True,
                data=rebuttal_training,
                message="Rebuttal practice session completed",
                confidence=0.85,
                next_actions=["coach_debate", "improve_rhetoric"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to provide rebuttal practice",
                errors=[str(e)],
            )

    async def _improve_rhetoric(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Improve rhetorical skills and persuasive communication."""
        try:
            # Get user's debate history for personalized advice
            debate_history = await self.memory_manager.get_user_memories(user_id, memory_type="debate_coaching")
            
            rhetoric_improvement = {
                "rhetorical_techniques": self._teach_rhetorical_techniques(),
                "persuasion_principles": self._explain_persuasion_principles(),
                "emotional_appeals": self._guide_emotional_appeals(),
                "credibility_building": self._teach_credibility_building(),
                "audience_analysis": self._guide_audience_analysis(),
                "language_power": self._teach_powerful_language(),
                "delivery_mastery": self._improve_delivery(),
                "practice_exercises": self._provide_rhetoric_exercises(),
                "personalized_feedback": self._generate_personalized_rhetoric_feedback(debate_history),
            }

            return SkillResult(
                success=True,
                data=rhetoric_improvement,
                message="Rhetoric improvement session completed",
                confidence=0.9,
                next_actions=["coach_debate", "practice_rebuttal"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to provide rhetoric improvement",
                errors=[str(e)],
            )

    async def _assess_logic(self, user_id: str, input_data: Dict[str, Any]) -> SkillResult:
        """Assess logical reasoning and identify areas for improvement."""
        try:
            argument = input_data.get("argument", "")
            
            logic_assessment = {
                "logical_structure": self._evaluate_logical_structure(argument),
                "reasoning_quality": self._assess_reasoning_quality(argument),
                "premise_validity": self._check_premise_validity(argument),
                "conclusion_support": self._evaluate_conclusion_support(argument),
                "logical_gaps": self._identify_logical_gaps(argument),
                "fallacy_detection": self._comprehensive_fallacy_check(argument),
                "logic_score": self._calculate_logic_score(argument),
                "improvement_recommendations": self._recommend_logic_improvements(argument),
            }

            return SkillResult(
                success=True,
                data=logic_assessment,
                message="Logic assessment completed",
                confidence=0.9,
                next_actions=["analyze_argument", "improve_rhetoric"],
            )

        except Exception as e:
            return SkillResult(
                success=False,
                data={},
                message="Failed to assess logic",
                errors=[str(e)],
            )

    def _analyze_argument_structure(self, argument: str) -> Dict[str, Any]:
        """Analyze the structure of an argument."""
        structure = {
            "has_clear_claim": "claim" in argument.lower() or "argue" in argument.lower(),
            "has_evidence": any(word in argument.lower() for word in ["because", "since", "evidence", "research", "study"]),
            "has_reasoning": any(word in argument.lower() for word in ["therefore", "thus", "consequently", "leads to"]),
            "structure_type": "deductive" if "therefore" in argument.lower() else "inductive",
            "complexity": "complex" if len(argument.split('.')) > 3 else "simple",
        }
        
        # Identify argument components
        components = []
        if structure["has_clear_claim"]:
            components.append("clear_claim")
        if structure["has_evidence"]:
            components.append("supporting_evidence")
        if structure["has_reasoning"]:
            components.append("logical_reasoning")
        
        structure["components"] = components
        structure["completeness"] = len(components) / 3  # Score out of 1
        
        return structure

    def _assess_logical_strength(self, argument: str) -> Dict[str, Any]:
        """Assess the logical strength of an argument."""
        strength_indicators = {
            "causal_reasoning": any(word in argument.lower() for word in ["cause", "effect", "leads to", "results in"]),
            "comparative_analysis": any(word in argument.lower() for word in ["compared to", "versus", "better than"]),
            "statistical_evidence": any(word in argument.lower() for word in ["percent", "%", "study", "research", "data"]),
            "expert_citation": any(word in argument.lower() for word in ["expert", "professor", "researcher", "according to"]),
            "logical_connectors": any(word in argument.lower() for word in ["therefore", "thus", "consequently", "because"]),
        }
        
        strength_score = sum(strength_indicators.values()) / len(strength_indicators)
        
        return {
            "indicators": strength_indicators,
            "strength_score": strength_score,
            "assessment": "strong" if strength_score > 0.6 else "moderate" if strength_score > 0.3 else "weak",
        }

    def _evaluate_evidence(self, argument: str) -> Dict[str, Any]:
        """Evaluate the quality of evidence in an argument."""
        evidence_types = {
            "statistical": any(word in argument.lower() for word in ["percent", "%", "statistics", "data", "number"]),
            "expert_testimony": any(word in argument.lower() for word in ["expert", "researcher", "professor", "study"]),
            "historical": any(word in argument.lower() for word in ["history", "past", "previously", "before"]),
            "anecdotal": any(word in argument.lower() for word in ["example", "instance", "case", "story"]),
            "scientific": any(word in argument.lower() for word in ["research", "study", "experiment", "science"]),
        }
        
        evidence_quality = sum(evidence_types.values())
        
        return {
            "evidence_types": evidence_types,
            "variety_score": evidence_quality / len(evidence_types),
            "quality_assessment": "high" if evidence_quality >= 3 else "moderate" if evidence_quality >= 2 else "low",
            "missing_evidence": [k for k, v in evidence_types.items() if not v],
        }

    def _identify_rhetorical_devices(self, argument: str) -> List[str]:
        """Identify rhetorical devices used in the argument."""
        devices = []
        
        # Check for common rhetorical devices
        if any(word in argument.lower() for word in ["imagine", "picture", "visualize"]):
            devices.append("imagery")
        if argument.count('?') > 0:
            devices.append("rhetorical_questions")
        if any(word in argument.lower() for word in ["we", "us", "our", "together"]):
            devices.append("inclusive_language")
        if any(word in argument.lower() for word in ["always", "never", "all", "every", "none"]):
            devices.append("absolutes")
        if any(word in argument.lower() for word in ["like", "as", "similar to"]):
            devices.append("analogies")
        
        return devices

    def _detect_logical_fallacies(self, argument: str) -> List[Dict[str, str]]:
        """Detect potential logical fallacies in the argument."""
        fallacies = []
        
        # Ad hominem
        if any(word in argument.lower() for word in ["stupid", "ignorant", "fool", "idiot"]):
            fallacies.append({
                "type": "ad_hominem",
                "description": "Attacking the person rather than the argument",
                "severity": "high"
            })
        
        # False dichotomy
        if any(phrase in argument.lower() for phrase in ["either...or", "only two", "must choose"]):
            fallacies.append({
                "type": "false_dichotomy",
                "description": "Presenting only two options when more exist",
                "severity": "medium"
            })
        
        # Appeal to emotion
        if any(word in argument.lower() for word in ["terrible", "horrible", "wonderful", "amazing"]) and "because" not in argument.lower():
            fallacies.append({
                "type": "appeal_to_emotion",
                "description": "Using emotion instead of logical reasoning",
                "severity": "medium"
            })
        
        # Slippery slope
        if any(phrase in argument.lower() for phrase in ["will lead to", "next thing", "eventually"]):
            fallacies.append({
                "type": "slippery_slope",
                "description": "Assuming one event will lead to extreme consequences",
                "severity": "medium"
            })
        
        return fallacies

    def _find_vulnerabilities(self, argument: str) -> List[str]:
        """Find potential vulnerabilities in the argument."""
        vulnerabilities = []
        
        # Weak evidence
        if "some say" in argument.lower() or "many believe" in argument.lower():
            vulnerabilities.append("Vague source attribution")
        
        # Overgeneralization
        if any(word in argument.lower() for word in ["always", "never", "all", "every"]):
            vulnerabilities.append("Overgeneralization")
        
        # Missing causation
        if "correlation" in argument.lower() and "causation" not in argument.lower():
            vulnerabilities.append("Correlation without proven causation")
        
        # Outdated information
        if any(year in argument for year in ["2010", "2011", "2012", "2013", "2014"]):
            vulnerabilities.append("Potentially outdated information")
        
        # Lack of opposing viewpoint
        if not any(word in argument.lower() for word in ["however", "although", "despite", "critics"]):
            vulnerabilities.append("Failure to address counterarguments")
        
        return vulnerabilities

    def _suggest_improvements(self, argument: str) -> List[str]:
        """Suggest specific improvements for the argument."""
        improvements = []
        
        # Check structure
        if "." not in argument or len(argument.split('.')) < 2:
            improvements.append("Break argument into clear, separate points")
        
        # Check evidence
        if not any(word in argument.lower() for word in ["study", "research", "data", "evidence"]):
            improvements.append("Add concrete evidence or research citations")
        
        # Check reasoning
        if not any(word in argument.lower() for word in ["because", "therefore", "thus", "since"]):
            improvements.append("Add clear logical connectors to show reasoning")
        
        # Check counterarguments
        if not any(word in argument.lower() for word in ["although", "however", "despite", "critics argue"]):
            improvements.append("Address potential counterarguments")
        
        # Check specificity
        if any(word in argument.lower() for word in ["some", "many", "most", "often"]):
            improvements.append("Replace vague terms with specific data or examples")
        
        return improvements

    def _calculate_argument_score(self, argument: str) -> Dict[str, float]:
        """Calculate overall argument quality score."""
        structure = self._analyze_argument_structure(argument)
        strength = self._assess_logical_strength(argument)
        evidence = self._evaluate_evidence(argument)
        fallacies = self._detect_logical_fallacies(argument)
        
        # Calculate component scores
        structure_score = structure["completeness"]
        logic_score = strength["strength_score"]
        evidence_score = evidence["variety_score"]
        fallacy_penalty = len(fallacies) * 0.1  # Deduct 0.1 for each fallacy
        
        overall_score = max(0, (structure_score + logic_score + evidence_score) / 3 - fallacy_penalty)
        
        return {
            "structure": structure_score,
            "logic": logic_score,
            "evidence": evidence_score,
            "fallacy_penalty": fallacy_penalty,
            "overall": overall_score,
            "grade": "A" if overall_score > 0.8 else "B" if overall_score > 0.6 else "C" if overall_score > 0.4 else "D",
        }

    def _analyze_debate_topic(self, topic: str) -> Dict[str, Any]:
        """Analyze a debate topic for key considerations."""
        return {
            "topic_type": self._classify_topic_type(topic),
            "stakeholders": self._identify_stakeholders(topic),
            "key_dimensions": self._identify_key_dimensions(topic),
            "complexity_level": self._assess_topic_complexity(topic),
            "controversy_level": self._assess_controversy_level(topic),
            "research_areas": self._suggest_research_areas(topic),
        }

    def _classify_topic_type(self, topic: str) -> str:
        """Classify the type of debate topic."""
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["should", "ought", "must"]):
            return "policy"
        elif any(word in topic_lower for word in ["is", "are", "exists"]):
            return "fact"
        elif any(word in topic_lower for word in ["good", "bad", "better", "worse", "moral"]):
            return "value"
        else:
            return "general"

    def _identify_stakeholders(self, topic: str) -> List[str]:
        """Identify key stakeholders in the debate topic."""
        topic_lower = topic.lower()
        stakeholders = ["general public"]
        
        if any(word in topic_lower for word in ["government", "policy", "law"]):
            stakeholders.extend(["government", "policymakers", "citizens"])
        if any(word in topic_lower for word in ["business", "economy", "market"]):
            stakeholders.extend(["businesses", "consumers", "workers"])
        if any(word in topic_lower for word in ["education", "school", "student"]):
            stakeholders.extend(["students", "teachers", "parents", "administrators"])
        if any(word in topic_lower for word in ["health", "medical", "patient"]):
            stakeholders.extend(["patients", "healthcare providers", "insurers"])
        
        return list(set(stakeholders))

    def _identify_key_dimensions(self, topic: str) -> List[str]:
        """Identify key dimensions to consider in the debate."""
        dimensions = ["logical", "ethical"]
        topic_lower = topic.lower()
        
        if any(word in topic_lower for word in ["cost", "money", "budget", "economic"]):
            dimensions.append("economic")
        if any(word in topic_lower for word in ["environment", "climate", "nature"]):
            dimensions.append("environmental")
        if any(word in topic_lower for word in ["social", "society", "community"]):
            dimensions.append("social")
        if any(word in topic_lower for word in ["legal", "law", "constitutional"]):
            dimensions.append("legal")
        if any(word in topic_lower for word in ["practical", "implementation", "feasible"]):
            dimensions.append("practical")
        
        return dimensions

    def _assess_topic_complexity(self, topic: str) -> str:
        """Assess the complexity level of the topic."""
        complexity_indicators = len(topic.split()) + topic.count(',') + topic.count('and')
        
        if complexity_indicators > 15:
            return "high"
        elif complexity_indicators > 8:
            return "medium"
        else:
            return "low"

    def _assess_controversy_level(self, topic: str) -> str:
        """Assess how controversial the topic is."""
        controversial_terms = ["abortion", "religion", "politics", "immigration", "gun", "climate"]
        topic_lower = topic.lower()
        
        if any(term in topic_lower for term in controversial_terms):
            return "high"
        elif any(word in topic_lower for word in ["should", "ban", "allow", "prohibit"]):
            return "medium"
        else:
            return "low"

    def _suggest_research_areas(self, topic: str) -> List[str]:
        """Suggest areas for research on the topic."""
        return [
            "Historical context and precedents",
            "Current statistics and data",
            "Expert opinions and academic research",
            "International comparisons",
            "Economic impact analysis",
            "Social implications",
            "Implementation challenges",
            "Alternative solutions"
        ]

    def _generate_key_arguments(self, topic: str, stance: str) -> List[Dict[str, str]]:
        """Generate key arguments for a topic and stance."""
        # This would normally use more sophisticated analysis
        return [
            {
                "argument": f"Primary argument supporting {stance} position on {topic}",
                "evidence_type": "statistical",
                "strength": "high"
            },
            {
                "argument": f"Secondary moral argument for {stance} stance",
                "evidence_type": "ethical",
                "strength": "medium"
            },
            {
                "argument": f"Practical benefits of {stance} approach",
                "evidence_type": "practical",
                "strength": "high"
            }
        ]

    def _anticipate_counterarguments(self, topic: str, stance: str) -> List[Dict[str, str]]:
        """Anticipate likely counterarguments."""
        opposing_stance = "con" if stance == "pro" else "pro"
        
        return [
            {
                "counterargument": f"Main {opposing_stance} argument against {topic}",
                "likelihood": "high",
                "response_strategy": "Direct refutation with evidence"
            },
            {
                "counterargument": f"Cost/benefit challenge to {stance} position",
                "likelihood": "medium",
                "response_strategy": "Comparative analysis"
            },
            {
                "counterargument": f"Implementation difficulty argument",
                "likelihood": "medium",
                "response_strategy": "Practical solutions and examples"
            }
        ]

    def _suggest_evidence_sources(self, topic: str) -> List[str]:
        """Suggest types of evidence sources to research."""
        return [
            "Peer-reviewed academic journals",
            "Government reports and statistics",
            "Reputable news organizations",
            "Expert interviews and quotes",
            "Historical case studies",
            "International data and comparisons",
            "Think tank research",
            "Primary source documents"
        ]

    def _recommend_rhetorical_strategies(self, topic: str, stance: str) -> List[str]:
        """Recommend rhetorical strategies for the topic and stance."""
        return [
            "Appeal to shared values and common ground",
            "Use concrete examples and case studies",
            "Employ logical progression from premise to conclusion",
            "Address counterarguments preemptively",
            "Use credible source citations",
            "Apply appropriate emotional appeals",
            "Utilize comparative analysis",
            "Emphasize practical benefits"
        ]

    def _provide_debate_structure(self) -> Dict[str, List[str]]:
        """Provide recommended debate structure."""
        return {
            "opening": [
                "Hook: Attention-grabbing opening",
                "Context: Background information",
                "Thesis: Clear position statement",
                "Preview: Outline of main arguments"
            ],
            "body": [
                "Argument 1: Strongest point with evidence",
                "Argument 2: Supporting point with examples",
                "Argument 3: Additional support or moral dimension",
                "Address counterarguments"
            ],
            "closing": [
                "Summarize key arguments",
                "Reinforce thesis statement",
                "Call to action or final appeal",
                "Memorable concluding statement"
            ]
        }

    def _generate_practice_questions(self, topic: str) -> List[str]:
        """Generate practice questions for debate preparation."""
        return [
            f"What are the three strongest arguments for your position on {topic}?",
            f"How would you respond to the claim that {topic} is too expensive?",
            f"What evidence contradicts your position, and how do you address it?",
            f"Why is your solution better than the status quo?",
            f"What are the potential unintended consequences of your position?",
            f"How does your position align with fundamental values?",
            f"What would implementation of your position look like?",
            f"How do other countries/contexts handle this issue?"
        ]

    def _share_winning_techniques(self) -> List[str]:
        """Share debate techniques that increase chances of winning."""
        return [
            "Control the narrative: Frame the debate in terms favorable to your position",
            "Use the 'Yes, and...' technique: Acknowledge valid points while strengthening your case",
            "Employ strategic concessions: Give ground on minor points to strengthen major ones",
            "Master the art of reframing: Turn opponent's arguments to support your position",
            "Use evidence hierarchy: Lead with strongest evidence, support with secondary sources",
            "Practice active listening: Respond to actual arguments, not straw men",
            "Maintain composure: Stay calm and professional under pressure",
            "End strong: Finish with your most compelling point"
        ]

    def _generate_rebuttal_strategies(self, opposing_argument: str) -> List[Dict[str, str]]:
        """Generate specific rebuttal strategies for an opposing argument."""
        return [
            {
                "strategy": "Challenge the evidence",
                "description": "Question the quality, relevance, or interpretation of their evidence",
                "application": "Look for outdated data, biased sources, or misrepresented statistics"
            },
            {
                "strategy": "Expose logical fallacies",
                "description": "Identify and point out logical errors in their reasoning",
                "application": "Look for ad hominem attacks, false dichotomies, or slippery slope arguments"
            },
            {
                "strategy": "Provide counter-evidence",
                "description": "Present contradictory evidence or alternative interpretations",
                "application": "Use more recent studies, different methodologies, or broader datasets"
            },
            {
                "strategy": "Reframe the argument",
                "description": "Change the context or perspective to weaken their position",
                "application": "Shift focus to different criteria, values, or stakeholder perspectives"
            }
        ]

    def _suggest_counter_evidence(self, opposing_argument: str) -> List[str]:
        """Suggest types of counter-evidence to research."""
        return [
            "Alternative studies or research with different conclusions",
            "More recent data that contradicts their claims",
            "Expert opinions that disagree with their sources",
            "Examples of failed implementations of their proposal",
            "Unintended consequences from similar policies",
            "Cultural or contextual factors they haven't considered",
            "Economic analysis showing different outcomes",
            "Methodological flaws in their cited research"
        ]

    def _identify_logical_weaknesses(self, opposing_argument: str) -> List[str]:
        """Identify logical weaknesses in the opposing argument."""
        weaknesses = []
        
        fallacies = self._detect_logical_fallacies(opposing_argument)
        for fallacy in fallacies:
            weaknesses.append(f"Contains {fallacy['type']}: {fallacy['description']}")
        
        vulnerabilities = self._find_vulnerabilities(opposing_argument)
        weaknesses.extend(vulnerabilities)
        
        return weaknesses

    def _craft_rhetorical_responses(self, opposing_argument: str) -> List[str]:
        """Craft rhetorical responses to opposing arguments."""
        return [
            "While my opponent raises an interesting point, the evidence suggests otherwise...",
            "This argument might seem compelling at first glance, but a deeper analysis reveals...",
            "My opponent's position fails to consider the broader implications of...",
            "The fundamental flaw in this reasoning is the assumption that...",
            "Even if we accept my opponent's premise, the conclusion doesn't follow because...",
            "This argument creates a false choice between X and Y, when in reality...",
            "The evidence my opponent cites actually supports my position when we consider..."
        ]

    def _generate_practice_rebuttals(self, opposing_argument: str) -> List[str]:
        """Generate practice rebuttals for the opposing argument."""
        return [
            f"Challenge: {opposing_argument}",
            "Response 1: Question the underlying assumptions",
            "Response 2: Provide contradictory evidence",
            "Response 3: Point out logical inconsistencies",
            "Response 4: Reframe the issue from a different perspective",
            "Response 5: Show unintended consequences of their position"
        ]

    def _provide_timing_strategies(self) -> List[str]:
        """Provide timing strategies for effective rebuttals."""
        return [
            "Immediate response: Address the strongest point first",
            "Strategic delay: Let weak arguments hang for emphasis",
            "Grouped rebuttal: Address multiple weak points together",
            "Preemptive strike: Anticipate and address before they argue",
            "Closing emphasis: Save strongest rebuttal for final impact",
            "Time management: Allocate time based on argument strength"
        ]

    def _provide_delivery_tips(self) -> List[str]:
        """Provide tips for effective rebuttal delivery."""
        return [
            "Maintain confident body language and eye contact",
            "Use clear, measured speech pace",
            "Employ strategic pauses for emphasis",
            "Keep tone respectful but firm",
            "Use gestures to emphasize key points",
            "Avoid emotional reactions to provocative statements",
            "Stay organized and follow logical sequence",
            "End with confidence and conviction"
        ]

    def _teach_rhetorical_techniques(self) -> Dict[str, str]:
        """Teach classical rhetorical techniques."""
        return {
            "ethos": "Build credibility through expertise, character, and trustworthiness",
            "pathos": "Appeal to emotions through vivid imagery, stories, and shared values",
            "logos": "Use logical reasoning, evidence, and clear argumentation",
            "kairos": "Choose the right moment and context for maximum impact",
            "antithesis": "Use contrasting ideas to highlight differences",
            "repetition": "Repeat key phrases for emphasis and memorability",
            "rhetorical_questions": "Engage audience thinking without expecting answers",
            "metaphor": "Use comparisons to make complex ideas accessible"
        }

    def _explain_persuasion_principles(self) -> Dict[str, str]:
        """Explain key principles of persuasion."""
        return {
            "reciprocity": "People feel obligated to return favors and concessions",
            "commitment": "People want to be consistent with previous statements and beliefs",
            "social_proof": "People follow the actions of similar others",
            "authority": "People defer to recognized experts and leaders",
            "liking": "People are more easily persuaded by those they like",
            "scarcity": "People value things that are rare or limited",
            "contrast": "Present your position against a less favorable alternative"
        }

    def _guide_emotional_appeals(self) -> List[str]:
        """Guide effective use of emotional appeals."""
        return [
            "Use emotions to support, not replace, logical arguments",
            "Choose emotions appropriate to your audience and topic",
            "Tell compelling stories that illustrate your points",
            "Use vivid, sensory language to create emotional resonance",
            "Appeal to shared values and common experiences",
            "Balance positive and negative emotional appeals",
            "Avoid manipulation; aim for genuine emotional connection",
            "Consider timing: when emotions will be most effective"
        ]

    def _teach_credibility_building(self) -> List[str]:
        """Teach how to build credibility in arguments."""
        return [
            "Cite authoritative and recent sources",
            "Acknowledge limitations and counterarguments",
            "Use precise, accurate language",
            "Demonstrate deep knowledge of the topic",
            "Show fairness by representing opposing views accurately",
            "Use personal experience appropriately",
            "Maintain consistency in your reasoning",
            "Admit when you don't know something"
        ]

    def _guide_audience_analysis(self) -> List[str]:
        """Guide effective audience analysis for persuasion."""
        return [
            "Identify audience values, beliefs, and priorities",
            "Consider their knowledge level and expertise",
            "Understand their cultural and social context",
            "Recognize their potential biases and preconceptions",
            "Determine what motivates them to act",
            "Assess their relationship to the topic",
            "Consider their decision-making authority",
            "Adapt your language and examples to their experience"
        ]

    def _teach_powerful_language(self) -> List[str]:
        """Teach use of powerful, persuasive language."""
        return [
            "Use active voice for stronger impact",
            "Choose concrete, specific words over abstract ones",
            "Employ strong verbs instead of weak verb-adverb combinations",
            "Use parallel structure for emphasis and flow",
            "Eliminate unnecessary words and qualifiers",
            "Choose words with appropriate connotations",
            "Use figurative language sparingly but effectively",
            "Vary sentence length and structure for rhythm"
        ]

    def _improve_delivery(self) -> List[str]:
        """Improve speech delivery and presentation skills."""
        return [
            "Practice proper breathing for voice control",
            "Use strategic pauses for emphasis and clarity",
            "Vary your pace to maintain interest",
            "Project confidence through posture and gestures",
            "Make appropriate eye contact with audience",
            "Use vocal variety to convey emotion and meaning",
            "Practice smooth transitions between points",
            "End statements strongly without uptalk"
        ]

    def _provide_rhetoric_exercises(self) -> List[str]:
        """Provide practical rhetoric exercises."""
        return [
            "Practice the 'Rule of Three': group arguments in threes",
            "Exercise: Defend an unpopular position using only logos",
            "Exercise: Tell the same story using different emotional appeals",
            "Practice: Turn opponent's evidence to support your position",
            "Exercise: Write the same argument for different audiences",
            "Practice: Deliver the same point with different levels of intensity",
            "Exercise: Create analogies to explain complex concepts",
            "Practice: Anticipate and prepare for hostile questions"
        ]

    def _generate_personalized_rhetoric_feedback(self, debate_history: List[Dict]) -> List[str]:
        """Generate personalized rhetoric feedback based on user's history."""
        if not debate_history:
            return ["Build a debate history to receive personalized feedback"]
        
        feedback = []
        
        # Analyze patterns in user's debate history
        topics_covered = len(set(session.get("topic", "") for session in debate_history))
        if topics_covered < 3:
            feedback.append("Practice with more diverse topics to improve adaptability")
        
        # Check for coaching session frequency
        if len(debate_history) > 5:
            feedback.append("You're building good debate practice consistency!")
        else:
            feedback.append("Regular practice will help internalize these techniques")
        
        feedback.extend([
            "Focus on areas where you feel least confident",
            "Record yourself practicing to identify delivery improvements",
            "Seek opportunities to debate in real settings",
        ])
        
        return feedback

    def _evaluate_logical_structure(self, argument: str) -> Dict[str, Any]:
        """Evaluate the logical structure of an argument."""
        structure_analysis = self._analyze_argument_structure(argument)
        
        return {
            "has_premises": structure_analysis["has_evidence"],
            "has_conclusion": structure_analysis["has_clear_claim"],
            "logical_flow": structure_analysis["has_reasoning"],
            "structure_completeness": structure_analysis["completeness"],
            "structure_type": structure_analysis["structure_type"],
            "recommendations": self._recommend_structure_improvements(structure_analysis)
        }

    def _assess_reasoning_quality(self, argument: str) -> Dict[str, Any]:
        """Assess the quality of reasoning in an argument."""
        reasoning_indicators = {
            "causal_links": any(word in argument.lower() for word in ["causes", "leads to", "results in"]),
            "logical_progression": any(word in argument.lower() for word in ["first", "second", "finally", "therefore"]),
            "conditional_reasoning": any(word in argument.lower() for word in ["if", "then", "when", "unless"]),
            "comparative_reasoning": any(word in argument.lower() for word in ["better", "worse", "compared to"]),
        }
        
        quality_score = sum(reasoning_indicators.values()) / len(reasoning_indicators)
        
        return {
            "reasoning_types": reasoning_indicators,
            "quality_score": quality_score,
            "assessment": "strong" if quality_score > 0.6 else "adequate" if quality_score > 0.3 else "weak"
        }

    def _check_premise_validity(self, argument: str) -> Dict[str, Any]:
        """Check the validity of premises in the argument."""
        premise_issues = []
        
        # Check for unsupported claims
        if any(phrase in argument.lower() for phrase in ["everyone knows", "it's obvious", "clearly"]):
            premise_issues.append("Contains unsupported assertions")
        
        # Check for vague terms
        if any(word in argument.lower() for word in ["some", "many", "most", "often"]):
            premise_issues.append("Uses vague quantifiers")
        
        # Check for outdated information
        current_year = datetime.now().year
        for year in range(2000, current_year - 5):
            if str(year) in argument:
                premise_issues.append("May contain outdated information")
                break
        
        return {
            "validity_issues": premise_issues,
            "validity_score": max(0, 1 - len(premise_issues) * 0.2),
            "assessment": "valid" if not premise_issues else "questionable"
        }

    def _evaluate_conclusion_support(self, argument: str) -> Dict[str, Any]:
        """Evaluate how well the conclusion is supported."""
        support_indicators = {
            "explicit_conclusion": any(word in argument.lower() for word in ["therefore", "thus", "consequently", "in conclusion"]),
            "evidence_cited": any(word in argument.lower() for word in ["study", "research", "data", "evidence"]),
            "reasoning_provided": any(word in argument.lower() for word in ["because", "since", "due to"]),
            "logical_connection": self._check_logical_connection(argument),
        }
        
        support_score = sum(support_indicators.values()) / len(support_indicators)
        
        return {
            "support_elements": support_indicators,
            "support_score": support_score,
            "assessment": "well_supported" if support_score > 0.75 else "moderately_supported" if support_score > 0.5 else "poorly_supported"
        }

    def _check_logical_connection(self, argument: str) -> bool:
        """Check if there's a logical connection between premises and conclusion."""
        # Simplified check for logical connectors
        return any(word in argument.lower() for word in ["therefore", "thus", "because", "since", "leads to"])

    def _identify_logical_gaps(self, argument: str) -> List[str]:
        """Identify gaps in logical reasoning."""
        gaps = []
        
        # Missing causal explanation
        if "correlation" in argument.lower() and "causation" not in argument.lower():
            gaps.append("Correlation presented without establishing causation")
        
        # Unstated assumptions
        if "obviously" in argument.lower() or "clearly" in argument.lower():
            gaps.append("Contains unstated assumptions")
        
        # Missing intermediate steps
        sentences = argument.split('.')
        if len(sentences) >= 2:
            first_sentence = sentences[0].lower()
            last_sentence = sentences[-1].lower()
            if "therefore" in last_sentence and len(sentences) == 2:
                gaps.append("Missing intermediate logical steps")
        
        return gaps

    def _comprehensive_fallacy_check(self, argument: str) -> List[Dict[str, str]]:
        """Perform comprehensive logical fallacy detection."""
        return self._detect_logical_fallacies(argument)  # Reuse existing method

    def _calculate_logic_score(self, argument: str) -> float:
        """Calculate overall logic score."""
        structure = self._evaluate_logical_structure(argument)
        reasoning = self._assess_reasoning_quality(argument)
        premises = self._check_premise_validity(argument)
        conclusion = self._evaluate_conclusion_support(argument)
        gaps = self._identify_logical_gaps(argument)
        fallacies = self._comprehensive_fallacy_check(argument)
        
        # Weighted scoring
        logic_score = (
            structure["structure_completeness"] * 0.2 +
            reasoning["quality_score"] * 0.25 +
            premises["validity_score"] * 0.2 +
            conclusion["support_score"] * 0.25 +
            max(0, 1 - len(gaps) * 0.1) * 0.05 +
            max(0, 1 - len(fallacies) * 0.1) * 0.05
        )
        
        return min(1.0, max(0.0, logic_score))

    def _recommend_logic_improvements(self, argument: str) -> List[str]:
        """Recommend specific improvements to logical reasoning."""
        improvements = []
        
        # Check each component and suggest improvements
        structure = self._evaluate_logical_structure(argument)
        if structure["structure_completeness"] < 0.8:
            improvements.append("Strengthen argument structure with clear premises and conclusion")
        
        reasoning = self._assess_reasoning_quality(argument)
        if reasoning["quality_score"] < 0.6:
            improvements.append("Add more explicit logical connections between ideas")
        
        premises = self._check_premise_validity(argument)
        if premises["validity_score"] < 0.8:
            improvements.append("Support claims with credible evidence and sources")
        
        gaps = self._identify_logical_gaps(argument)
        if gaps:
            improvements.append("Address logical gaps: " + "; ".join(gaps))
        
        fallacies = self._comprehensive_fallacy_check(argument)
        if fallacies:
            improvements.append("Eliminate logical fallacies: " + "; ".join([f["type"] for f in fallacies]))
        
        return improvements if improvements else ["Logical reasoning appears sound"]

    def _recommend_structure_improvements(self, structure_analysis: Dict[str, Any]) -> List[str]:
        """Recommend improvements to argument structure."""
        improvements = []
        
        if not structure_analysis["has_clear_claim"]:
            improvements.append("Add a clear, explicit claim or thesis statement")
        
        if not structure_analysis["has_evidence"]:
            improvements.append("Include supporting evidence for your claims")
        
        if not structure_analysis["has_reasoning"]:
            improvements.append("Add logical reasoning connecting evidence to conclusions")
        
        if structure_analysis["complexity"] == "simple" and structure_analysis["completeness"] < 0.8:
            improvements.append("Develop more comprehensive argumentation")
        
        return improvements if improvements else ["Argument structure is well-developed"]