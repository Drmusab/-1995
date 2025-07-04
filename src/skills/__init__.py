"""
Skills Module for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 11:30:00 UTC

This module provides comprehensive skill management capabilities including
skill factory, registry, validation, and execution framework.
"""

from .skill_factory import (
    SkillFactory,
    SkillExecutionMode,
    SkillCacheStrategy,
    CircuitBreakerState,
    SkillExecutionContext,
    SkillExecutionResult,
    SkillTemplate,
    CircuitBreaker,
    SkillComposition,
    SkillCreated,
    SkillExecutionStarted,
    SkillExecutionCompleted,
    SkillExecutionFailed,
    CircuitBreakerTriggered,
    SkillHotReloaded
)

from .skill_registry import (
    SkillRegistry,
    SkillInterface,
    SkillType,
    SkillState,
    SkillMetadata,
    SkillCapability,
    SkillRegistration,
    SkillRegistered,
    SkillUnregistered,
    SkillStateChanged
)

from .skill_validator import (
    SkillValidator,
    ValidationSeverity,
    ValidationType,
    ValidationResult,
    ValidationReport,
    ValidationRule,
    SkillValidationStarted,
    SkillValidationCompleted,
    SkillSecurityViolation
)

__all__ = [
    # Skill Factory
    'SkillFactory',
    'SkillExecutionMode',
    'SkillCacheStrategy', 
    'CircuitBreakerState',
    'SkillExecutionContext',
    'SkillExecutionResult',
    'SkillTemplate',
    'CircuitBreaker',
    'SkillComposition',
    'SkillCreated',
    'SkillExecutionStarted',
    'SkillExecutionCompleted',
    'SkillExecutionFailed',
    'CircuitBreakerTriggered',
    'SkillHotReloaded',
    
    # Skill Registry
    'SkillRegistry',
    'SkillInterface',
    'SkillType',
    'SkillState',
    'SkillMetadata',
    'SkillCapability',
    'SkillRegistration',
    'SkillRegistered',
    'SkillUnregistered',
    'SkillStateChanged',
    
    # Skill Validator
    'SkillValidator',
    'ValidationSeverity',
    'ValidationType',
    'ValidationResult',
    'ValidationReport',
    'ValidationRule',
    'SkillValidationStarted',
    'SkillValidationCompleted',
    'SkillSecurityViolation'
]

# Module version
__version__ = "1.0.0"