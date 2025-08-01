"""
Skills Module for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 11:30:00 UTC

This module provides comprehensive skill management capabilities including
skill factory, registry, validation, and execution framework.
"""

from .skill_factory import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerTriggered,
    SkillCacheStrategy,
    SkillComposition,
    SkillCreated,
    SkillExecutionCompleted,
    SkillExecutionContext,
    SkillExecutionFailed,
    SkillExecutionMode,
    SkillExecutionResult,
    SkillExecutionStarted,
    SkillFactory,
    SkillHotReloaded,
    SkillTemplate,
)
from .skill_registry import (
    SkillCapability,
    SkillInterface,
    SkillMetadata,
    SkillRegistered,
    SkillRegistration,
    SkillRegistry,
    SkillState,
    SkillStateChanged,
    SkillType,
    SkillUnregistered,
)
from .skill_validator import (
    SkillSecurityViolation,
    SkillValidationCompleted,
    SkillValidationStarted,
    SkillValidator,
    ValidationReport,
    ValidationResult,
    ValidationRule,
    ValidationSeverity,
    ValidationType,
)

__all__ = [
    # Skill Factory
    "SkillFactory",
    "SkillExecutionMode",
    "SkillCacheStrategy",
    "CircuitBreakerState",
    "SkillExecutionContext",
    "SkillExecutionResult",
    "SkillTemplate",
    "CircuitBreaker",
    "SkillComposition",
    "SkillCreated",
    "SkillExecutionStarted",
    "SkillExecutionCompleted",
    "SkillExecutionFailed",
    "CircuitBreakerTriggered",
    "SkillHotReloaded",
    # Skill Registry
    "SkillRegistry",
    "SkillInterface",
    "SkillType",
    "SkillState",
    "SkillMetadata",
    "SkillCapability",
    "SkillRegistration",
    "SkillRegistered",
    "SkillUnregistered",
    "SkillStateChanged",
    # Skill Validator
    "SkillValidator",
    "ValidationSeverity",
    "ValidationType",
    "ValidationResult",
    "ValidationReport",
    "ValidationRule",
    "SkillValidationStarted",
    "SkillValidationCompleted",
    "SkillSecurityViolation",
]

# Module version
__version__ = "1.0.0"
