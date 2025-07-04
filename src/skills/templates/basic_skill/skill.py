"""
{{ skill_name }} - {{ skill_description }}
Author: {{ author }}

This is a basic skill template.
"""

from datetime import datetime, timezone
from typing import Any, Dict

from src.skills.skill_registry import SkillInterface, SkillMetadata, SkillType, SkillCapability


class {{ skill_name }}(SkillInterface):
    """{{ skill_description }}"""
    
    def get_metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return SkillMetadata(
            skill_id="{{ skill_name.lower() }}",
            name="{{ skill_name }}",
            version="1.0.0",
            description="{{ skill_description }}",
            author="{{ author }}",
            skill_type=SkillType.CUSTOM,
            capabilities=[
                SkillCapability(
                    name="execute",
                    description="Execute the skill",
                    input_types=["any"],
                    output_types=["string"]
                )
            ]
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the skill."""
        self.config = config
        self.initialized_at = datetime.now(timezone.utc)
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute the skill."""
        return f"{{ skill_name }} executed with input: {input_data}"
    
    async def validate(self, input_data: Any) -> bool:
        """Validate input data."""
        return True
    
    async def cleanup(self) -> None:
        """Cleanup skill resources."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check skill health."""
        return {
            "status": "healthy",
            "initialized_at": self.initialized_at.isoformat() if hasattr(self, 'initialized_at') else None
        }