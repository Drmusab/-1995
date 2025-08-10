"""
Simplified Configuration Validation (Legacy Compatibility Layer)
Author: Drmusab
Last Modified: 2025-01-13

This module provides a simplified configuration validation system that works with
the unified YAML-first configuration system while maintaining backward compatibility.
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Import the unified configuration system
from src.core.config.unified_config import get_unified_config


class ValidationLevel(Enum):
    """Configuration validation levels."""
    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"
    CUSTOM = "custom"


class ValidationType(Enum):
    """Types of validation checks."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_id: str
    section: str
    key_path: str
    severity: ValidationSeverity
    message: str
    actual_value: Any = None
    expected_value: Any = None
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    config_version: str
    environment: str
    results: List[ValidationResult] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    is_valid: bool = True
    can_start: bool = True
    recommendations: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    
    def __init__(self, message: str, section: Optional[str] = None, key_path: Optional[str] = None):
        super().__init__(message)
        self.section = section
        self.key_path = key_path
        self.timestamp = datetime.now(timezone.utc)


class EnhancedConfigValidator:
    """
    Simplified Configuration Validator using the unified configuration system.
    
    This validator provides basic configuration validation while delegating
    complex validation logic to the YAML-defined rules in the unified config.
    """
    
    def __init__(self, container=None):
        """Initialize the configuration validator."""
        self.container = container
        self.unified_config = get_unified_config()
        self.validation_level = ValidationLevel.STANDARD
        self.current_environment = os.getenv("ENVIRONMENT", "development")
        
        # Simple logging
        import logging
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> None:
        """Initialize the configuration validator."""
        await self.unified_config.initialize()
        self.logger.info("ConfigValidator initialized")
    
    async def validate_all_configurations(
        self, validation_level: Optional[ValidationLevel] = None
    ) -> ValidationReport:
        """
        Validate all system configurations using the unified config system.
        
        Args:
            validation_level: Override default validation level
            
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        validation_id = f"validation_{int(start_time)}"
        
        # Use the unified configuration validation
        errors = self.unified_config.validate_configuration()
        
        # Create validation report
        report = ValidationReport(
            validation_id=validation_id,
            config_version="unified",
            environment=self.current_environment,
            validation_time=time.time() - start_time
        )
        
        # Convert errors to validation results
        for i, error in enumerate(errors):
            result = ValidationResult(
                rule_id=f"unified_rule_{i}",
                section="configuration",
                key_path="unknown",
                severity=ValidationSeverity.ERROR,
                message=error
            )
            report.results.append(result)
        
        # Update counts
        report.total_issues = len(errors)
        report.error_issues = len(errors)
        report.is_valid = len(errors) == 0
        report.can_start = len(errors) == 0
        
        # Generate recommendations
        if report.error_issues > 0:
            report.recommendations.append("Fix configuration errors before starting the system")
        
        self.logger.info(f"Configuration validation completed: {validation_id} ({report.total_issues} issues)")
        
        return report
    
    async def validate_section(
        self, section: str, config_data: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Validate a specific configuration section.
        
        Args:
            section: Section name to validate
            config_data: Optional specific configuration data
            
        Returns:
            List of validation results
        """
        results = []
        
        # Basic section validation
        section_config = self.unified_config.get_section(section)
        
        if not section_config:
            results.append(ValidationResult(
                rule_id="section_missing",
                section=section,
                key_path=section,
                severity=ValidationSeverity.WARNING,
                message=f"Configuration section '{section}' is missing or empty"
            ))
        
        return results
    
    async def validate_plugin_config(
        self, plugin_id: str, plugin_config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate plugin configuration.
        
        Args:
            plugin_id: Plugin identifier
            plugin_config: Plugin configuration
            
        Returns:
            Validation report for the plugin
        """
        validation_id = f"plugin_{plugin_id}_{int(time.time())}"
        
        report = ValidationReport(
            validation_id=validation_id,
            config_version="plugin",
            environment=self.current_environment,
        )
        
        # Basic plugin validation
        required_keys = ["name", "version", "type"]
        for key in required_keys:
            if key not in plugin_config:
                report.results.append(ValidationResult(
                    rule_id="plugin_missing_key",
                    section="plugin",
                    key_path=f"plugin.{key}",
                    severity=ValidationSeverity.ERROR,
                    message=f"Required plugin configuration key '{key}' is missing"
                ))
        
        # Update counts
        report.total_issues = len(report.results)
        report.error_issues = len([r for r in report.results if r.severity == ValidationSeverity.ERROR])
        report.is_valid = report.error_issues == 0
        
        return report
    
    async def validate_skill_config(
        self, skill_id: str, skill_config: Dict[str, Any]
    ) -> ValidationReport:
        """
        Validate skill configuration.
        
        Args:
            skill_id: Skill identifier
            skill_config: Skill configuration
            
        Returns:
            Validation report for the skill
        """
        validation_id = f"skill_{skill_id}_{int(time.time())}"
        
        report = ValidationReport(
            validation_id=validation_id,
            config_version="skill",
            environment=self.current_environment,
        )
        
        # Basic skill validation
        required_keys = ["name", "version", "capabilities"]
        for key in required_keys:
            if key not in skill_config:
                report.results.append(ValidationResult(
                    rule_id="skill_missing_key",
                    section="skill",
                    key_path=f"skill.{key}",
                    severity=ValidationSeverity.ERROR,
                    message=f"Required skill configuration key '{key}' is missing"
                ))
        
        # Validate capabilities
        if "capabilities" in skill_config:
            capabilities = skill_config["capabilities"]
            if not isinstance(capabilities, list) or not capabilities:
                report.results.append(ValidationResult(
                    rule_id="skill_invalid_capabilities",
                    section="skill",
                    key_path="skill.capabilities",
                    severity=ValidationSeverity.ERROR,
                    message="Skill must have at least one capability defined"
                ))
        
        # Update counts
        report.total_issues = len(report.results)
        report.error_issues = len([r for r in report.results if r.severity == ValidationSeverity.ERROR])
        report.is_valid = report.error_issues == 0
        
        return report
    
    def get_validation_report(self, validation_id: str) -> Optional[ValidationReport]:
        """Get a specific validation report."""
        # In the simplified version, we don't cache reports
        return None
    
    def get_latest_validation_report(self) -> Optional[ValidationReport]:
        """Get the latest validation report."""
        # In the simplified version, we don't cache reports
        return None
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "validation_level": self.validation_level.value,
            "environment": self.current_environment,
            "active_rules": "managed_by_unified_config",
        }
    
    def set_validation_level(self, level: ValidationLevel) -> None:
        """Set the validation level."""
        self.validation_level = level
        self.logger.info(f"Validation level set to: {level.value}")
    
    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        self.logger.info("ConfigValidator cleanup completed")


# Backward compatibility functions
async def create_enhanced_config_validator(container=None) -> EnhancedConfigValidator:
    """Create an enhanced configuration validator."""
    validator = EnhancedConfigValidator(container)
    await validator.initialize()
    return validator