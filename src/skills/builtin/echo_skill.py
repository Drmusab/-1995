"""
Echo Skill - A simple built-in skill for testing
Author: Drmusab

This skill simply echoes back the input data.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from src.skills.skill_registry import SkillInterface, SkillMetadata, SkillType, SkillCapability


class EchoSkill(SkillInterface):
    """A simple echo skill that returns the input data."""
    
    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return SkillMetadata(
            skill_id="builtin.echo",
            name="Echo Skill",
            version="1.0.0",
            description="Echoes back the input data",
            author="Drmusab",
            skill_type=SkillType.BUILTIN,
            capabilities=[
                SkillCapability(
                    name="echo",
                    description="Echo input data",
                    input_types=["any"],
                    output_types=["any"]
                )
            ],
            tags=["utility", "testing", "echo"],
            resource_requirements={
                "cpu_percent": 1,
                "memory_mb": 10
            }
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the skill."""
        self.config = config
        self.initialized_at = datetime.now(timezone.utc)
        self.execution_count = 0
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute the skill by echoing the input."""
        self.execution_count += 1
        
        return {
            "echoed_data": input_data,
            "execution_count": self.execution_count,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def validate(self, input_data: Any) -> bool:
        """Validate input data (always valid for echo)."""
        return True
    
    async def cleanup(self) -> None:
        """Cleanup skill resources."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check skill health."""
        return {
            "status": "healthy",
            "initialized_at": self.initialized_at.isoformat() if hasattr(self, 'initialized_at') else None,
            "execution_count": getattr(self, 'execution_count', 0)
        }