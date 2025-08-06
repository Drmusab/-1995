"""
Task Classifier for Time-Block Planner
Author: Drmusab
Last Modified: 2025-01-20

Parses user tasks and preferences with Arabic-first language support.
"""

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.observability.logging.config import get_logger
from .time_block_planner import Task, TaskPriority, TaskComplexity, FocusType


class TaskClassifier:
    """Classifies and extracts tasks from user input with bilingual support."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Arabic keywords for task classification
        self.arabic_keywords = {
            "priorities": {
                "عاجل": TaskPriority.URGENT,
                "مهم": TaskPriority.HIGH, 
                "ضروري": TaskPriority.HIGH,
                "عادي": TaskPriority.MEDIUM,
                "بسيط": TaskPriority.LOW,
                "منخفض": TaskPriority.LOW
            },
            "complexity": {
                "معقد": TaskComplexity.COMPLEX,
                "صعب": TaskComplexity.COMPLEX,
                "متوسط": TaskComplexity.MODERATE,
                "بسيط": TaskComplexity.SIMPLE,
                "سهل": TaskComplexity.SIMPLE,
                "معقد جداً": TaskComplexity.VERY_COMPLEX
            },
            "focus_types": {
                "كتابة": FocusType.DEEP_WORK,
                "تفكير": FocusType.DEEP_WORK,
                "دراسة": FocusType.LEARNING,
                "تعلم": FocusType.LEARNING,
                "إداري": FocusType.ADMIN,
                "إدارة": FocusType.ADMIN,
                "إيميل": FocusType.COMMUNICATION,
                "رسائل": FocusType.COMMUNICATION,
                "اجتماع": FocusType.COMMUNICATION,
                "إبداع": FocusType.CREATIVE,
                "تصميم": FocusType.CREATIVE
            },
            "time_patterns": {
                "ساعة": 60,
                "ساعتان": 120,
                "نصف ساعة": 30,
                "ربع ساعة": 15,
                "دقيقة": 1,
                "دقائق": 1
            }
        }
        
        # English keywords for fallback
        self.english_keywords = {
            "priorities": {
                "urgent": TaskPriority.URGENT,
                "important": TaskPriority.HIGH,
                "high": TaskPriority.HIGH,
                "medium": TaskPriority.MEDIUM,
                "low": TaskPriority.LOW,
                "normal": TaskPriority.MEDIUM
            },
            "complexity": {
                "complex": TaskComplexity.COMPLEX,
                "difficult": TaskComplexity.COMPLEX,
                "moderate": TaskComplexity.MODERATE,
                "simple": TaskComplexity.SIMPLE,
                "easy": TaskComplexity.SIMPLE,
                "very complex": TaskComplexity.VERY_COMPLEX
            },
            "focus_types": {
                "writing": FocusType.DEEP_WORK,
                "thinking": FocusType.DEEP_WORK,
                "study": FocusType.LEARNING,
                "learning": FocusType.LEARNING,
                "admin": FocusType.ADMIN,
                "administrative": FocusType.ADMIN,
                "email": FocusType.COMMUNICATION,
                "meeting": FocusType.COMMUNICATION,
                "creative": FocusType.CREATIVE,
                "design": FocusType.CREATIVE
            },
            "time_patterns": {
                "hour": 60,
                "hours": 60,
                "minute": 1,
                "minutes": 1,
                "half hour": 30,
                "quarter hour": 15
            }
        }

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the task classifier."""
        self.logger.info("TaskClassifier initialized")

    async def extract_tasks(self, input_text: str) -> List[Task]:
        """Extract tasks from user input text."""
        tasks = []
        
        # Split by common delimiters (Arabic and English)
        separators = ['\n', '،', ',', 'و', 'and', '؛', ';']
        task_segments = self._split_by_separators(input_text, separators)
        
        for i, segment in enumerate(task_segments):
            segment = segment.strip()
            if not segment:
                continue
                
            task = await self._parse_single_task(segment, i)
            if task:
                tasks.append(task)
        
        return tasks

    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        """Split text by multiple separators."""
        segments = [text]
        
        for sep in separators:
            new_segments = []
            for segment in segments:
                new_segments.extend(segment.split(sep))
            segments = new_segments
        
        return [s.strip() for s in segments if s.strip()]

    async def _parse_single_task(self, task_text: str, index: int) -> Optional[Task]:
        """Parse a single task from text."""
        try:
            # Extract basic information
            title_ar, title_en = self._extract_titles(task_text)
            
            # Extract duration
            duration = self._extract_duration(task_text)
            
            # Extract priority
            priority = self._extract_priority(task_text)
            
            # Extract complexity
            complexity = self._extract_complexity(task_text)
            
            # Extract focus type
            focus_type = self._extract_focus_type(task_text)
            
            # Extract deadline
            deadline = self._extract_deadline(task_text)
            
            task = Task(
                id=f"task_{index}_{uuid.uuid4().hex[:8]}",
                title_ar=title_ar,
                title_en=title_en,
                description_ar=task_text,
                description_en=task_text,  # Would translate in real implementation
                estimated_duration=duration,
                priority=priority,
                complexity=complexity,
                focus_type=focus_type,
                deadline=deadline
            )
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error parsing task '{task_text}': {str(e)}")
            return None

    def _extract_titles(self, text: str) -> tuple[str, str]:
        """Extract Arabic and English titles from text."""
        # Remove time and priority indicators
        clean_text = re.sub(r'\d+\s*(ساعة|ساعات|دقيقة|دقائق|hour|hours|minute|minutes)', '', text)
        clean_text = re.sub(r'(عاجل|مهم|urgent|important|high|low)', '', clean_text, flags=re.IGNORECASE)
        clean_text = clean_text.strip()
        
        # For now, return the same text for both languages
        # In a real implementation, we would detect language and translate
        return clean_text, clean_text

    def _extract_duration(self, text: str) -> int:
        """Extract estimated duration from text."""
        # Check Arabic time patterns
        for pattern, multiplier in self.arabic_keywords["time_patterns"].items():
            # Look for numbers before time words
            ar_pattern = rf'(\d+)\s*{re.escape(pattern)}'
            match = re.search(ar_pattern, text)
            if match:
                return int(match.group(1)) * multiplier
        
        # Check English time patterns
        for pattern, multiplier in self.english_keywords["time_patterns"].items():
            en_pattern = rf'(\d+)\s*{re.escape(pattern)}'
            match = re.search(en_pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1)) * multiplier
        
        # Default duration based on task complexity keywords
        if any(word in text for word in ["معقد", "complex", "difficult"]):
            return 120  # 2 hours for complex tasks
        elif any(word in text for word in ["كتابة", "دراسة", "writing", "study"]):
            return 90   # 1.5 hours for deep work
        else:
            return 60   # 1 hour default

    def _extract_priority(self, text: str) -> TaskPriority:
        """Extract task priority from text."""
        text_lower = text.lower()
        
        # Check Arabic priority keywords
        for keyword, priority in self.arabic_keywords["priorities"].items():
            if keyword in text:
                return priority
        
        # Check English priority keywords  
        for keyword, priority in self.english_keywords["priorities"].items():
            if keyword in text_lower:
                return priority
        
        return TaskPriority.MEDIUM  # Default

    def _extract_complexity(self, text: str) -> TaskComplexity:
        """Extract task complexity from text."""
        text_lower = text.lower()
        
        # Check Arabic complexity keywords
        for keyword, complexity in self.arabic_keywords["complexity"].items():
            if keyword in text:
                return complexity
        
        # Check English complexity keywords
        for keyword, complexity in self.english_keywords["complexity"].items():
            if keyword in text_lower:
                return complexity
        
        return TaskComplexity.MODERATE  # Default

    def _extract_focus_type(self, text: str) -> FocusType:
        """Extract focus type from text."""
        text_lower = text.lower()
        
        # Check Arabic focus type keywords
        for keyword, focus_type in self.arabic_keywords["focus_types"].items():
            if keyword in text:
                return focus_type
        
        # Check English focus type keywords
        for keyword, focus_type in self.english_keywords["focus_types"].items():
            if keyword in text_lower:
                return focus_type
        
        return FocusType.ADMIN  # Default

    def _extract_deadline(self, text: str) -> Optional[datetime]:
        """Extract deadline from text."""
        # Look for date patterns (simplified)
        today = datetime.now().date()
        
        # Arabic date keywords
        if "اليوم" in text or "today" in text.lower():
            return datetime.combine(today, datetime.min.time().replace(hour=17))
        elif "غداً" in text or "tomorrow" in text.lower():
            return datetime.combine(today + timedelta(days=1), datetime.min.time().replace(hour=17))
        elif "الأسبوع" in text or "week" in text.lower():
            return datetime.combine(today + timedelta(days=7), datetime.min.time().replace(hour=17))
        
        # Look for specific date patterns (MM/DD or DD/MM)
        date_pattern = r'(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?'
        match = re.search(date_pattern, text)
        if match:
            try:
                day, month = int(match.group(1)), int(match.group(2))
                year = int(match.group(3)) if match.group(3) else today.year
                
                # Adjust for 2-digit years
                if year < 100:
                    year += 2000
                
                deadline_date = datetime(year, month, day, 17, 0)  # Default to 5 PM
                return deadline_date
            except ValueError:
                pass
        
        return None

    async def classify_user_input(self, input_text: str) -> Dict[str, Any]:
        """Classify user input to determine task planning preferences."""
        classification = {
            "language": "ar" if self._is_arabic_text(input_text) else "en",
            "intent": "plan_tasks",
            "urgency": "normal",
            "session_type": "individual"
        }
        
        # Detect urgency
        urgent_keywords_ar = ["عاجل", "فوري", "سريع", "الآن"]
        urgent_keywords_en = ["urgent", "asap", "immediately", "now", "quickly"]
        
        if any(keyword in input_text for keyword in urgent_keywords_ar + urgent_keywords_en):
            classification["urgency"] = "high"
        
        # Detect session type
        if any(word in input_text for word in ["اجتماع", "meeting", "فريق", "team"]):
            classification["session_type"] = "team"
        
        return classification

    def _is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_char_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return arabic_char_count > len(text) * 0.3  # More than 30% Arabic characters