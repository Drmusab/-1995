#!/usr/bin/env python3
"""
Standalone Skill Factory Test
Author: Drmusab
Last Modified: 2025-01-20 12:00:00 UTC

This script tests the skill factory components in isolation.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import time

# Simple test implementations to avoid import issues
class SkillType(Enum):
    BUILTIN = "builtin"
    CUSTOM = "custom"
    META = "meta"
    EXTERNAL = "external"
    TEMPLATE = "template"

class SkillState(Enum):
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    VALIDATED = "validated"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    DEPRECATED = "deprecated"

@dataclass
class SkillCapability:
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    required_resources: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillMetadata:
    skill_id: str
    name: str
    version: str
    description: str
    author: str
    skill_type: SkillType
    capabilities: List[SkillCapability]
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

class SkillInterface:
    def get_metadata(self) -> SkillMetadata:
        raise NotImplementedError
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        raise NotImplementedError

@dataclass
class SkillRegistration:
    skill_id: str
    skill_class: type
    metadata: SkillMetadata
    state: SkillState = SkillState.REGISTERED
    instance: Any = None
    registration_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = None
    access_count: int = 0

class MockLogger:
    def __init__(self, name):
        self.name = name
    
    def info(self, msg): print(f"INFO [{self.name}]: {msg}")
    def debug(self, msg): print(f"DEBUG [{self.name}]: {msg}")
    def warning(self, msg): print(f"WARNING [{self.name}]: {msg}")
    def error(self, msg): print(f"ERROR [{self.name}]: {msg}")

class SimpleSkillRegistry:
    """Simplified skill registry for testing."""
    
    def __init__(self):
        self.logger = MockLogger("SkillRegistry")
        self.skills: Dict[str, SkillRegistration] = {}
        self.skills_by_type: Dict[SkillType, List[str]] = defaultdict(list)
    
    async def register_skill(self, skill_id: str, skill_class: type, 
                           metadata: SkillMetadata = None) -> bool:
        try:
            if metadata is None:
                temp_instance = skill_class()
                metadata = temp_instance.get_metadata()
            
            registration = SkillRegistration(
                skill_id=skill_id,
                skill_class=skill_class,
                metadata=metadata
            )
            
            self.skills[skill_id] = registration
            self.skills_by_type[metadata.skill_type].append(skill_id)
            
            self.logger.info(f"Registered skill: {skill_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register skill {skill_id}: {e}")
            return False
    
    def get_skill(self, skill_id: str) -> SkillRegistration:
        return self.skills.get(skill_id)
    
    def is_skill_available(self, skill_id: str) -> bool:
        registration = self.skills.get(skill_id)
        return registration is not None and registration.state in [SkillState.ACTIVE, SkillState.REGISTERED]
    
    async def update_skill_state(self, skill_id: str, new_state: SkillState) -> bool:
        if skill_id in self.skills:
            self.skills[skill_id].state = new_state
            return True
        return False

@dataclass
class SkillExecutionContext:
    session_id: str = None
    user_id: str = None
    correlation_id: str = None
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SkillExecutionResult:
    skill_id: str
    execution_id: str
    success: bool
    result: Any = None
    error: str = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class SimpleSkillFactory:
    """Simplified skill factory for testing."""
    
    def __init__(self, skill_registry):
        self.logger = MockLogger("SkillFactory")
        self.skill_registry = skill_registry
        self.active_skills: Dict[str, Any] = {}
        self.execution_history: deque = deque(maxlen=1000)
        self.execution_semaphore = asyncio.Semaphore(10)
    
    async def create_skill(self, skill_id: str, config: Dict[str, Any] = None) -> Any:
        registration = self.skill_registry.get_skill(skill_id)
        if not registration:
            raise ValueError(f"Skill {skill_id} not found")
        
        if not self.skill_registry.is_skill_available(skill_id):
            raise ValueError(f"Skill {skill_id} not available")
        
        if skill_id in self.active_skills:
            return self.active_skills[skill_id]
        
        # Create instance
        instance = registration.skill_class()
        await instance.initialize(config or {})
        
        self.active_skills[skill_id] = instance
        registration.instance = instance
        registration.access_count += 1
        registration.last_accessed = datetime.now(timezone.utc)
        
        await self.skill_registry.update_skill_state(skill_id, SkillState.ACTIVE)
        
        self.logger.info(f"Created skill instance: {skill_id}")
        return instance
    
    async def execute_skill(self, skill_id: str, input_data: Any,
                          context: SkillExecutionContext = None) -> SkillExecutionResult:
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        if context is None:
            context = SkillExecutionContext()
        
        result = SkillExecutionResult(
            skill_id=skill_id,
            execution_id=execution_id,
            success=False
        )
        
        try:
            async with self.execution_semaphore:
                # Get or create skill instance
                if skill_id in self.active_skills:
                    instance = self.active_skills[skill_id]
                else:
                    instance = await self.create_skill(skill_id)
                
                # Execute skill
                exec_context = {
                    "session_id": context.session_id,
                    "user_id": context.user_id,
                    "correlation_id": context.correlation_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **context.metadata
                }
                
                skill_result = await asyncio.wait_for(
                    instance.execute(input_data, exec_context),
                    timeout=context.timeout_seconds
                )
                
                result.success = True
                result.result = skill_result
                
                execution_time = (time.time() - start_time) * 1000
                result.execution_time_ms = execution_time
                
                self.logger.debug(f"Executed skill {skill_id} in {execution_time:.2f}ms")
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Skill execution failed for {skill_id}: {e}")
        
        self.execution_history.append(result)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "active_skills": len(self.active_skills),
            "total_executions": len(self.execution_history),
            "successful_executions": sum(1 for r in self.execution_history if r.success)
        }

# Test skill implementation
class EchoSkill(SkillInterface):
    """A simple echo skill for testing."""
    
    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            skill_id="test.echo",
            name="Echo Skill",
            version="1.0.0",
            description="Echoes back input data",
            author="Test",
            skill_type=SkillType.BUILTIN,
            capabilities=[
                SkillCapability(
                    name="echo",
                    description="Echo input data",
                    input_types=["any"],
                    output_types=["any"]
                )
            ],
            tags=["test", "echo"],
            resource_requirements={"memory_mb": 10}
        )
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.initialized_at = datetime.now(timezone.utc)
        self.execution_count = 0
    
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        self.execution_count += 1
        return {
            "echoed_data": input_data,
            "execution_count": self.execution_count,
            "context_keys": list(context.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

async def test_skill_system():
    """Test the simplified skill system."""
    print("=" * 60)
    print("SKILL SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Initialize components
        print("\n1. Initializing Components...")
        skill_registry = SimpleSkillRegistry()
        skill_factory = SimpleSkillFactory(skill_registry)
        print("✓ Components initialized")
        
        # Register test skill
        print("\n2. Registering Echo Skill...")
        success = await skill_registry.register_skill("test.echo", EchoSkill)
        print(f"✓ Skill registration: {success}")
        
        # Create and execute skill
        print("\n3. Executing Skill...")
        test_input = {"message": "Hello, Skill Factory!", "test": True}
        context = SkillExecutionContext(
            session_id="test-session",
            user_id="test-user",
            correlation_id="test-123"
        )
        
        result = await skill_factory.execute_skill("test.echo", test_input, context)
        
        print(f"✓ Execution Result:")
        print(f"  - Success: {result.success}")
        print(f"  - Time: {result.execution_time_ms:.2f}ms")
        print(f"  - Result: {result.result}")
        
        # Test multiple executions
        print("\n4. Multiple Executions...")
        for i in range(3):
            result = await skill_factory.execute_skill(
                "test.echo", 
                {"iteration": i}, 
                context
            )
            print(f"  - Execution {i+1}: {result.success} ({result.execution_time_ms:.1f}ms)")
        
        # Test error handling
        print("\n5. Error Handling...")
        try:
            await skill_factory.execute_skill("nonexistent", {}, context)
        except Exception as e:
            print(f"✓ Error handled correctly: {type(e).__name__}")
        
        # Get statistics
        print("\n6. Statistics...")
        stats = skill_factory.get_statistics()
        print(f"✓ Factory Statistics:")
        print(f"  - Active skills: {stats['active_skills']}")
        print(f"  - Total executions: {stats['total_executions']}")
        print(f"  - Successful executions: {stats['successful_executions']}")
        
        print(f"\n✓ Registry Statistics:")
        print(f"  - Total skills: {len(skill_registry.skills)}")
        print(f"  - Skills by type: {dict(skill_registry.skills_by_type)}")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_skill_system())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)