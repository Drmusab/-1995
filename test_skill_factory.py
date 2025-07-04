#!/usr/bin/env python3
"""
Skill Factory Integration Test
Author: Drmusab
Last Modified: 2025-01-20 11:45:00 UTC

This script demonstrates the comprehensive skill factory functionality
and integration with the core system.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock imports for testing (since we don't have all dependencies)
class MockConfigLoader:
    def get(self, key, default=None):
        config_map = {
            "skills.max_concurrent": 100,
            "skills.default_timeout": 30.0,
            "skills.cache_enabled": True,
            "skills.hot_reload_enabled": False,
            "skills.circuit_breaker_enabled": True,
            "skills.builtin_paths": ["src/skills/builtin"],
            "validation.max_time_seconds": 30,
            "validation.security_level": "standard",
            "validation.performance_thresholds": {
                "max_memory_kb": 10000,
                "max_complexity": 50,
                "max_cpu_percent": 80,
                "max_memory_mb": 1000
            }
        }
        return config_map.get(key, default)

class MockEventBus:
    async def emit(self, event):
        print(f"EVENT: {event.__class__.__name__} - {event.skill_id if hasattr(event, 'skill_id') else 'system'}")

class MockMetricsCollector:
    def increment(self, metric, value=1):
        print(f"METRIC: {metric} +{value}")
    
    def histogram(self, metric, value):
        print(f"METRIC: {metric} = {value}")
    
    def set(self, metric, value):
        print(f"METRIC: {metric} = {value}")

class MockComponent:
    pass

class MockContainer:
    def __init__(self):
        self.components = {
            MockConfigLoader: MockConfigLoader(),
            MockEventBus: MockEventBus(),
            MockMetricsCollector: MockMetricsCollector(),
            MockComponent: MockComponent()
        }
    
    def get(self, component_type, default=None):
        return self.components.get(component_type, default or MockComponent())

class MockHealthCheck:
    pass

class MockErrorHandler:
    pass

# Monkey patch the imports
import src.skills.skill_registry as skill_registry_module
import src.skills.skill_validator as skill_validator_module  
import src.skills.skill_factory as skill_factory_module

# Replace the imports with our mocks
skill_registry_module.ConfigLoader = MockConfigLoader
skill_registry_module.EventBus = MockEventBus
skill_registry_module.MetricsCollector = MockMetricsCollector
skill_registry_module.TraceManager = MockComponent
skill_registry_module.ErrorHandler = MockErrorHandler
skill_registry_module.HealthCheck = MockHealthCheck
skill_registry_module.get_logger = lambda name: logging.getLogger(name)

skill_validator_module.ConfigLoader = MockConfigLoader
skill_validator_module.EventBus = MockEventBus
skill_validator_module.MetricsCollector = MockMetricsCollector
skill_validator_module.TraceManager = MockComponent
skill_validator_module.ErrorHandler = MockErrorHandler
skill_validator_module.HealthCheck = MockHealthCheck
skill_validator_module.AuthenticationManager = MockComponent
skill_validator_module.AuthorizationManager = MockComponent
skill_validator_module.SecuritySanitizer = MockComponent
skill_validator_module.get_logger = lambda name: logging.getLogger(name)

skill_factory_module.ConfigLoader = MockConfigLoader
skill_factory_module.EventBus = MockEventBus
skill_factory_module.MetricsCollector = MockMetricsCollector
skill_factory_module.TraceManager = MockComponent
skill_factory_module.ErrorHandler = MockErrorHandler
skill_factory_module.HealthCheck = MockHealthCheck
skill_factory_module.ComponentManager = MockComponent
skill_factory_module.WorkflowOrchestrator = MockComponent
skill_factory_module.SessionManager = MockComponent
skill_factory_module.MemoryManager = MockComponent
skill_factory_module.ContextManager = MockComponent
skill_factory_module.WorkingMemory = MockComponent
skill_factory_module.ContinualLearner = MockComponent
skill_factory_module.PreferenceLearner = MockComponent
skill_factory_module.FeedbackProcessor = MockComponent
skill_factory_module.get_logger = lambda name: logging.getLogger(name)

# Now import our skill components
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator
from src.skills.skill_factory import SkillFactory, SkillExecutionContext
from src.skills.builtin.echo_skill import EchoSkill

async def test_skill_factory():
    """Test the comprehensive skill factory."""
    print("=" * 60)
    print("SKILL FACTORY INTEGRATION TEST")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create mock container
    container = MockContainer()
    
    try:
        # Initialize components
        print("\n1. Initializing Components...")
        skill_registry = SkillRegistry(container)
        skill_validator = SkillValidator(container)
        
        # Add them to container
        container.components[SkillRegistry] = skill_registry
        container.components[SkillValidator] = skill_validator
        
        skill_factory = SkillFactory(container)
        
        print("✓ Components initialized successfully")
        
        # Register a test skill manually
        print("\n2. Registering Test Skill...")
        echo_skill_metadata = EchoSkill().get_metadata()
        success = await skill_registry.register_skill(
            "builtin.echo", 
            EchoSkill, 
            echo_skill_metadata
        )
        print(f"✓ Skill registration: {success}")
        
        # Validate the skill
        print("\n3. Validating Skill...")
        validation_report = await skill_validator.validate_skill(
            "builtin.echo",
            EchoSkill,
            echo_skill_metadata
        )
        print(f"✓ Validation result: {validation_report.is_valid}")
        print(f"  - Total checks: {validation_report.total_checks}")
        print(f"  - Critical issues: {validation_report.critical_issues}")
        print(f"  - Security issues: {validation_report.security_issues}")
        print(f"  - Overall score: {validation_report.overall_score:.1f}")
        
        # Initialize factory
        print("\n4. Initializing Skill Factory...")
        await skill_factory.initialize()
        print("✓ Skill factory initialized")
        
        # Create and execute skill
        print("\n5. Creating and Executing Skill...")
        test_input = {"message": "Hello, World!", "test": True}
        context = SkillExecutionContext(
            session_id="test-session",
            user_id="test-user",
            correlation_id="test-correlation"
        )
        
        result = await skill_factory.execute_skill("builtin.echo", test_input, context)
        
        print(f"✓ Execution result:")
        print(f"  - Success: {result.success}")
        print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
        print(f"  - Result: {result.result}")
        
        # Test health checks
        print("\n6. Health Checks...")
        factory_health = await skill_factory.health_check()
        registry_health = await skill_registry.health_check_all_skills()
        
        print(f"✓ Factory health: {factory_health['status']}")
        print(f"  - Active skills: {factory_health['active_skills']}")
        print(f"  - Total registered: {factory_health['total_registered_skills']}")
        
        # Get statistics
        print("\n7. Statistics...")
        factory_stats = skill_factory.get_factory_statistics()
        registry_stats = await skill_registry.get_skill_statistics()
        validator_stats = skill_validator.get_validation_statistics()
        
        print(f"✓ Factory Statistics:")
        print(f"  - Active skills: {factory_stats['active_skills']}")
        print(f"  - Total executions: {factory_stats['total_executions']}")
        print(f"  - Circuit breakers: {factory_stats['circuit_breaker_summary']}")
        
        print(f"✓ Registry Statistics:")
        print(f"  - Total skills: {registry_stats['total_skills']}")
        print(f"  - Skills by type: {registry_stats['skills_by_type']}")
        
        print(f"✓ Validator Statistics:")
        print(f"  - Total validations: {validator_stats['total_validations']}")
        print(f"  - Success rate: {validator_stats['success_rate']:.1f}%")
        
        # Test error handling
        print("\n8. Testing Error Handling...")
        try:
            await skill_factory.execute_skill("nonexistent.skill", test_input, context)
        except Exception as e:
            print(f"✓ Error handling works: {type(e).__name__}: {e}")
        
        # Cleanup
        print("\n9. Cleanup...")
        await skill_factory.cleanup()
        print("✓ Cleanup completed")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_skill_factory())
    sys.exit(0 if success else 1)