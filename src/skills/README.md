# Skills Module

A comprehensive skill management system for the AI Assistant that provides dynamic skill creation, lifecycle management, validation, and execution capabilities.

## Quick Start

```python
from src.skills import SkillFactory, SkillRegistry, SkillValidator

# Initialize components
container = Container()  # Your DI container
skill_registry = SkillRegistry(container)
skill_validator = SkillValidator(container)
skill_factory = SkillFactory(container)

# Initialize the factory
await skill_factory.initialize()

# Execute a skill
result = await skill_factory.execute_skill(
    "builtin.echo",
    {"message": "Hello World"},
    context
)
```

## Components

- **SkillFactory** - Core factory for skill creation and execution
- **SkillRegistry** - Manages skill registration and discovery
- **SkillValidator** - Provides comprehensive skill validation

## Features

✅ **Dynamic Skill Creation** - Create skills with dependency injection  
✅ **Lifecycle Management** - Complete skill lifecycle from creation to cleanup  
✅ **Skill Types** - Support for builtin, custom, meta, external, and template skills  
✅ **Security Validation** - Multi-layer security and safety checks  
✅ **Performance Monitoring** - Comprehensive metrics and performance tracking  
✅ **Circuit Breaker Pattern** - Automatic failure detection and recovery  
✅ **Hot Reload** - Update skills without system restart  
✅ **Skill Composition** - Chain multiple skills together  
✅ **Template System** - Create new skills from templates  
✅ **Event Integration** - Full integration with the event bus system  

## Architecture Integration

The skill factory integrates with:
- **Component Manager** - Dependency resolution and lifecycle
- **Workflow Orchestrator** - Skill execution in workflows  
- **Session Manager** - Session-aware skill execution
- **Memory Systems** - Skill state persistence
- **Learning Systems** - Skill optimization and adaptation

## Files

- `skill_factory.py` - Main factory implementation
- `skill_registry.py` - Skill registration and management
- `skill_validator.py` - Skill validation and security
- `builtin/` - Built-in skills directory
- `templates/` - Skill templates for scaffolding

## Testing

Run the standalone test to verify functionality:

```bash
python test_standalone.py
```

## Documentation

See [SKILL_FACTORY_DOCS.md](../SKILL_FACTORY_DOCS.md) for comprehensive documentation.

## Example Skill

```python
class MySkill(SkillInterface):
    def get_metadata(self):
        return SkillMetadata(
            skill_id="custom.my_skill",
            name="My Skill",
            version="1.0.0",
            description="Custom functionality",
            author="Your Name",
            skill_type=SkillType.CUSTOM,
            capabilities=[SkillCapability(...)]
        )
    
    async def initialize(self, config):
        # Initialize resources
        pass
    
    async def execute(self, input_data, context):
        # Implement skill logic
        return result
```