"""
Advanced Skill Validator for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 10:45:00 UTC

This module provides comprehensive skill validation capabilities including
security checks, compatibility verification, performance validation, and
business rule enforcement.
"""

import asyncio
import hashlib
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from pathlib import Path

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    BaseEvent, EventCategory, EventPriority, EventSeverity
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Security imports
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.sanitization import SecuritySanitizer

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Skills
from src.skills.skill_registry import SkillInterface, SkillMetadata, SkillType


class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class ValidationType(Enum):
    """Types of validation checks."""
    INTERFACE = "interface"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPATIBILITY = "compatibility"
    BUSINESS_RULES = "business_rules"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    RESOURCE_LIMITS = "resource_limits"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    rule_id: str
    rule_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: float = 0.0


@dataclass
class ValidationReport:
    """Comprehensive validation report for a skill."""
    skill_id: str
    validation_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    results: List[ValidationResult] = field(default_factory=list)
    overall_score: float = 0.0
    is_valid: bool = True
    critical_issues: int = 0
    security_issues: int = 0
    performance_issues: int = 0
    warnings: int = 0
    total_checks: int = 0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRule:
    """Represents a validation rule."""
    rule_id: str
    name: str
    description: str
    validation_type: ValidationType
    severity: ValidationSeverity
    enabled: bool = True
    validator_function: Callable = None
    configuration: Dict[str, Any] = field(default_factory=dict)


# Validation Events
@dataclass
class SkillValidationStarted(BaseEvent):
    """Event fired when skill validation starts."""
    skill_id: str
    validation_id: str
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillValidationCompleted(BaseEvent):
    """Event fired when skill validation completes."""
    skill_id: str
    validation_id: str
    is_valid: bool
    issues_found: int
    execution_time_ms: float
    category: EventCategory = EventCategory.SKILL


@dataclass
class SkillSecurityViolation(BaseEvent):
    """Event fired when a security violation is detected."""
    skill_id: str
    violation_type: str
    severity: ValidationSeverity
    details: Dict[str, Any]
    category: EventCategory = EventCategory.SECURITY
    priority: EventPriority = EventPriority.HIGH


class SkillValidator:
    """
    Comprehensive skill validation system with security, performance,
    and compatibility checks.
    
    Features:
    - Interface validation
    - Security and safety checks
    - Performance validation
    - Compatibility verification
    - Business rule enforcement
    - Custom validation rules
    - Detailed reporting
    """
    
    def __init__(self, container: Container):
        """Initialize the skill validator."""
        self.container = container
        self.logger = get_logger(__name__)
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)
        
        # Security components
        self.auth_manager = container.get(AuthenticationManager, None)
        self.authz_manager = container.get(AuthorizationManager, None)
        self.sanitizer = container.get(SecuritySanitizer, None)
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Validation rules
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.validation_profiles: Dict[str, List[str]] = {}
        
        # Configuration
        self.max_validation_time = self.config.get("validation.max_time_seconds", 30)
        self.security_level = self.config.get("validation.security_level", "standard")
        self.performance_thresholds = self.config.get("validation.performance_thresholds", {})
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # State
        self.validation_history: List[ValidationReport] = []
        self.is_healthy = True
        
        self.logger.info("SkillValidator initialized successfully")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules."""
        # Interface validation rules
        self.add_rule(ValidationRule(
            rule_id="interface_001",
            name="Required Methods",
            description="Check if skill implements required interface methods",
            validation_type=ValidationType.INTERFACE,
            severity=ValidationSeverity.ERROR,
            validator_function=self._validate_required_methods
        ))
        
        self.add_rule(ValidationRule(
            rule_id="interface_002",
            name="Method Signatures",
            description="Validate method signatures match interface",
            validation_type=ValidationType.INTERFACE,
            severity=ValidationSeverity.ERROR,
            validator_function=self._validate_method_signatures
        ))
        
        # Security validation rules
        self.add_rule(ValidationRule(
            rule_id="security_001",
            name="Dangerous Imports",
            description="Check for dangerous imports",
            validation_type=ValidationType.SECURITY,
            severity=ValidationSeverity.CRITICAL,
            validator_function=self._validate_dangerous_imports
        ))
        
        self.add_rule(ValidationRule(
            rule_id="security_002",
            name="File System Access",
            description="Check for unrestricted file system access",
            validation_type=ValidationType.SECURITY,
            severity=ValidationSeverity.SECURITY,
            validator_function=self._validate_file_access
        ))
        
        self.add_rule(ValidationRule(
            rule_id="security_003",
            name="Network Access",
            description="Check for network access patterns",
            validation_type=ValidationType.SECURITY,
            severity=ValidationSeverity.WARNING,
            validator_function=self._validate_network_access
        ))
        
        # Performance validation rules
        self.add_rule(ValidationRule(
            rule_id="performance_001",
            name="Memory Usage",
            description="Check estimated memory usage",
            validation_type=ValidationType.PERFORMANCE,
            severity=ValidationSeverity.WARNING,
            validator_function=self._validate_memory_usage
        ))
        
        self.add_rule(ValidationRule(
            rule_id="performance_002",
            name="Execution Time",
            description="Check estimated execution time",
            validation_type=ValidationType.PERFORMANCE,
            severity=ValidationSeverity.WARNING,
            validator_function=self._validate_execution_time
        ))
        
        # Compatibility validation rules
        self.add_rule(ValidationRule(
            rule_id="compatibility_001",
            name="Python Version",
            description="Check Python version compatibility",
            validation_type=ValidationType.COMPATIBILITY,
            severity=ValidationSeverity.ERROR,
            validator_function=self._validate_python_version
        ))
        
        self.add_rule(ValidationRule(
            rule_id="compatibility_002",
            name="Dependency Versions",
            description="Check dependency version compatibility",
            validation_type=ValidationType.COMPATIBILITY,
            severity=ValidationSeverity.WARNING,
            validator_function=self._validate_dependency_versions
        ))
        
        # Business rules
        self.add_rule(ValidationRule(
            rule_id="business_001",
            name="Resource Limits",
            description="Check resource usage limits",
            validation_type=ValidationType.BUSINESS_RULES,
            severity=ValidationSeverity.WARNING,
            validator_function=self._validate_resource_limits
        ))
        
        # Configuration validation rules
        self.add_rule(ValidationRule(
            rule_id="config_001",
            name="Required Configuration",
            description="Check required configuration parameters",
            validation_type=ValidationType.CONFIGURATION,
            severity=ValidationSeverity.ERROR,
            validator_function=self._validate_required_config
        ))
        
        # Create validation profiles
        self.validation_profiles = {
            "basic": ["interface_001", "interface_002"],
            "security": ["security_001", "security_002", "security_003"],
            "performance": ["performance_001", "performance_002"],
            "compatibility": ["compatibility_001", "compatibility_002"],
            "full": list(self.validation_rules.keys())
        }
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.validation_rules[rule.rule_id] = rule
        self.logger.debug(f"Added validation rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule."""
        if rule_id in self.validation_rules:
            del self.validation_rules[rule_id]
            self.logger.debug(f"Removed validation rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a validation rule."""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a validation rule."""
        if rule_id in self.validation_rules:
            self.validation_rules[rule_id].enabled = False
            return True
        return False
    
    @handle_exceptions
    async def validate_skill(self, skill_id: str, skill_class: Type,
                           metadata: SkillMetadata, profile: str = "full") -> ValidationReport:
        """
        Validate a skill against all applicable rules.
        
        Args:
            skill_id: Skill identifier
            skill_class: Skill class to validate
            metadata: Skill metadata
            profile: Validation profile to use
            
        Returns:
            Comprehensive validation report
        """
        validation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Fire validation started event
        await self.event_bus.emit(SkillValidationStarted(
            skill_id=skill_id,
            validation_id=validation_id
        ))
        
        # Create validation report
        report = ValidationReport(
            skill_id=skill_id,
            validation_id=validation_id
        )
        
        # Get rules for profile
        rule_ids = self.validation_profiles.get(profile, [])
        if not rule_ids:
            rule_ids = list(self.validation_rules.keys())
        
        # Run validation rules
        for rule_id in rule_ids:
            if rule_id not in self.validation_rules:
                continue
                
            rule = self.validation_rules[rule_id]
            if not rule.enabled:
                continue
            
            try:
                result = await self._run_validation_rule(
                    rule, skill_id, skill_class, metadata
                )
                report.results.append(result)
                
            except Exception as e:
                # Add error result
                error_result = ValidationResult(
                    rule_id=rule_id,
                    rule_name=rule.name,
                    validation_type=rule.validation_type,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Validation rule failed: {str(e)}"
                )
                report.results.append(error_result)
                self.logger.error(f"Validation rule {rule_id} failed: {str(e)}")
        
        # Calculate report metrics
        self._calculate_report_metrics(report)
        
        # Calculate execution time
        report.execution_time_ms = (time.time() - start_time) * 1000
        
        # Store in history
        self.validation_history.append(report)
        if len(self.validation_history) > 1000:
            self.validation_history.pop(0)
        
        # Update metrics
        self.metrics.increment("skill_validator.validations.total")
        self.metrics.histogram("skill_validator.execution_time_ms", report.execution_time_ms)
        
        if not report.is_valid:
            self.metrics.increment("skill_validator.validations.failed")
            
        if report.security_issues > 0:
            self.metrics.increment("skill_validator.security_violations")
            
            # Fire security violation event
            await self.event_bus.emit(SkillSecurityViolation(
                skill_id=skill_id,
                violation_type="validation_failed",
                severity=ValidationSeverity.SECURITY,
                details={"issues": report.security_issues, "report_id": validation_id}
            ))
        
        # Fire validation completed event
        await self.event_bus.emit(SkillValidationCompleted(
            skill_id=skill_id,
            validation_id=validation_id,
            is_valid=report.is_valid,
            issues_found=report.critical_issues + report.security_issues,
            execution_time_ms=report.execution_time_ms
        ))
        
        self.logger.info(f"Validation completed for skill {skill_id}: "
                        f"valid={report.is_valid}, issues={len(report.results)}")
        
        return report
    
    async def _run_validation_rule(self, rule: ValidationRule, skill_id: str,
                                 skill_class: Type, metadata: SkillMetadata) -> ValidationResult:
        """Run a single validation rule."""
        start_time = time.time()
        
        try:
            # Run the validator function
            if rule.validator_function:
                passed, message, details = await rule.validator_function(
                    skill_id, skill_class, metadata, rule.configuration
                )
            else:
                passed, message, details = True, "No validator function", {}
            
            # Create result
            result = ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            return result
            
        except Exception as e:
            # Return error result
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Validation error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_report_metrics(self, report: ValidationReport):
        """Calculate metrics for a validation report."""
        report.total_checks = len(report.results)
        
        for result in report.results:
            if result.severity == ValidationSeverity.CRITICAL:
                report.critical_issues += 1
            elif result.severity == ValidationSeverity.SECURITY:
                report.security_issues += 1
            elif result.severity == ValidationSeverity.WARNING:
                report.warnings += 1
            
            if result.validation_type == ValidationType.PERFORMANCE and not result.passed:
                report.performance_issues += 1
        
        # Calculate overall score (0-100)
        if report.total_checks > 0:
            failed_count = sum(1 for r in report.results if not r.passed)
            report.overall_score = ((report.total_checks - failed_count) / report.total_checks) * 100
        else:
            report.overall_score = 100
        
        # Determine if skill is valid
        report.is_valid = (report.critical_issues == 0 and report.security_issues == 0)
    
    # Individual validation rule implementations
    async def _validate_required_methods(self, skill_id: str, skill_class: Type, 
                                       metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate that skill implements required methods."""
        required_methods = ['get_metadata', 'initialize', 'execute']
        missing_methods = []
        
        for method_name in required_methods:
            if not hasattr(skill_class, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            return False, f"Missing required methods: {', '.join(missing_methods)}", {
                "missing_methods": missing_methods
            }
        
        return True, "All required methods implemented", {}
    
    async def _validate_method_signatures(self, skill_id: str, skill_class: Type,
                                        metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate method signatures match interface."""
        issues = []
        
        # Check get_metadata method
        if hasattr(skill_class, 'get_metadata'):
            sig = inspect.signature(skill_class.get_metadata)
            if len(sig.parameters) > 1:  # Only 'self' parameter expected
                issues.append("get_metadata should only have 'self' parameter")
        
        # Check initialize method
        if hasattr(skill_class, 'initialize'):
            sig = inspect.signature(skill_class.initialize)
            if len(sig.parameters) != 2:  # 'self' and 'config' parameters expected
                issues.append("initialize should have 'self' and 'config' parameters")
        
        # Check execute method
        if hasattr(skill_class, 'execute'):
            sig = inspect.signature(skill_class.execute)
            if len(sig.parameters) != 3:  # 'self', 'input_data', 'context' parameters expected
                issues.append("execute should have 'self', 'input_data', and 'context' parameters")
        
        if issues:
            return False, f"Method signature issues: {'; '.join(issues)}", {
                "issues": issues
            }
        
        return True, "All method signatures valid", {}
    
    async def _validate_dangerous_imports(self, skill_id: str, skill_class: Type,
                                        metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Check for dangerous imports."""
        dangerous_modules = [
            'os', 'sys', 'subprocess', 'shutil', 'pickle', 'eval', 'exec',
            'importlib', '__import__', 'compile', 'globals', 'locals'
        ]
        
        # Get source code
        try:
            source = inspect.getsource(skill_class)
        except OSError:
            return False, "Cannot access source code for security validation", {}
        
        found_dangerous = []
        
        for module in dangerous_modules:
            if f"import {module}" in source or f"from {module}" in source:
                found_dangerous.append(module)
        
        if found_dangerous:
            return False, f"Dangerous imports detected: {', '.join(found_dangerous)}", {
                "dangerous_imports": found_dangerous
            }
        
        return True, "No dangerous imports detected", {}
    
    async def _validate_file_access(self, skill_id: str, skill_class: Type,
                                  metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Check for file system access patterns."""
        try:
            source = inspect.getsource(skill_class)
        except OSError:
            return True, "Cannot access source code for file access validation", {}
        
        file_access_patterns = [
            r'open\s*\(',
            r'file\s*\(',
            r'Path\s*\(',
            r'\.read\s*\(',
            r'\.write\s*\(',
            r'\.delete\s*\(',
            r'\.unlink\s*\('
        ]
        
        found_patterns = []
        
        for pattern in file_access_patterns:
            if re.search(pattern, source):
                found_patterns.append(pattern)
        
        if found_patterns:
            return False, f"File access patterns detected: {len(found_patterns)} patterns", {
                "patterns_found": len(found_patterns)
            }
        
        return True, "No file access patterns detected", {}
    
    async def _validate_network_access(self, skill_id: str, skill_class: Type,
                                     metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Check for network access patterns."""
        try:
            source = inspect.getsource(skill_class)
        except OSError:
            return True, "Cannot access source code for network access validation", {}
        
        network_patterns = [
            r'requests\.',
            r'urllib\.',
            r'http\.',
            r'socket\.',
            r'aiohttp\.',
            r'websocket'
        ]
        
        found_patterns = []
        
        for pattern in network_patterns:
            if re.search(pattern, source):
                found_patterns.append(pattern)
        
        if found_patterns:
            return False, f"Network access patterns detected: {len(found_patterns)} patterns", {
                "patterns_found": len(found_patterns)
            }
        
        return True, "No network access patterns detected", {}
    
    async def _validate_memory_usage(self, skill_id: str, skill_class: Type,
                                   metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate estimated memory usage."""
        # Simple heuristic based on source code size
        try:
            source = inspect.getsource(skill_class)
            estimated_memory_kb = len(source) * 0.1  # Rough estimate
            
            max_memory_kb = self.performance_thresholds.get("max_memory_kb", 10000)
            
            if estimated_memory_kb > max_memory_kb:
                return False, f"Estimated memory usage too high: {estimated_memory_kb:.1f}KB", {
                    "estimated_memory_kb": estimated_memory_kb,
                    "max_allowed_kb": max_memory_kb
                }
            
            return True, f"Memory usage within limits: {estimated_memory_kb:.1f}KB", {
                "estimated_memory_kb": estimated_memory_kb
            }
            
        except OSError:
            return True, "Cannot estimate memory usage", {}
    
    async def _validate_execution_time(self, skill_id: str, skill_class: Type,
                                     metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate estimated execution time."""
        # Simple heuristic based on complexity
        try:
            source = inspect.getsource(skill_class)
            complexity_score = source.count('for') + source.count('while') + source.count('if')
            
            max_complexity = self.performance_thresholds.get("max_complexity", 50)
            
            if complexity_score > max_complexity:
                return False, f"Code complexity too high: {complexity_score}", {
                    "complexity_score": complexity_score,
                    "max_allowed": max_complexity
                }
            
            return True, f"Code complexity within limits: {complexity_score}", {
                "complexity_score": complexity_score
            }
            
        except OSError:
            return True, "Cannot estimate execution complexity", {}
    
    async def _validate_python_version(self, skill_id: str, skill_class: Type,
                                     metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate Python version compatibility."""
        import sys
        
        current_version = sys.version_info
        
        # Check if skill specifies version requirements
        if hasattr(metadata, 'python_version_min'):
            min_version = tuple(map(int, metadata.python_version_min.split('.')))
            if current_version < min_version:
                return False, f"Python version too old: {current_version} < {min_version}", {
                    "current_version": str(current_version),
                    "required_min": metadata.python_version_min
                }
        
        if hasattr(metadata, 'python_version_max'):
            max_version = tuple(map(int, metadata.python_version_max.split('.')))
            if current_version > max_version:
                return False, f"Python version too new: {current_version} > {max_version}", {
                    "current_version": str(current_version),
                    "required_max": metadata.python_version_max
                }
        
        return True, f"Python version compatible: {current_version}", {
            "current_version": str(current_version)
        }
    
    async def _validate_dependency_versions(self, skill_id: str, skill_class: Type,
                                          metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate dependency versions."""
        # For now, just check that dependencies are listed
        if not metadata.dependencies:
            return True, "No dependencies specified", {}
        
        # In a real implementation, we would check actual package versions
        return True, f"Dependencies specified: {len(metadata.dependencies)}", {
            "dependency_count": len(metadata.dependencies)
        }
    
    async def _validate_resource_limits(self, skill_id: str, skill_class: Type,
                                      metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate resource usage limits."""
        resource_reqs = metadata.resource_requirements
        
        max_cpu = self.performance_thresholds.get("max_cpu_percent", 80)
        max_memory = self.performance_thresholds.get("max_memory_mb", 1000)
        
        issues = []
        
        if 'cpu_percent' in resource_reqs and resource_reqs['cpu_percent'] > max_cpu:
            issues.append(f"CPU requirement too high: {resource_reqs['cpu_percent']}%")
        
        if 'memory_mb' in resource_reqs and resource_reqs['memory_mb'] > max_memory:
            issues.append(f"Memory requirement too high: {resource_reqs['memory_mb']}MB")
        
        if issues:
            return False, f"Resource limits exceeded: {'; '.join(issues)}", {
                "issues": issues
            }
        
        return True, "Resource requirements within limits", {}
    
    async def _validate_required_config(self, skill_id: str, skill_class: Type,
                                      metadata: SkillMetadata, config: Dict[str, Any]) -> tuple:
        """Validate required configuration."""
        if not metadata.configuration_schema:
            return True, "No configuration schema specified", {}
        
        required_fields = metadata.configuration_schema.get('required', [])
        
        if not required_fields:
            return True, "No required configuration fields", {}
        
        return True, f"Configuration schema has {len(required_fields)} required fields", {
            "required_fields": required_fields
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = len(self.validation_history)
        
        if total_validations == 0:
            return {
                "total_validations": 0,
                "success_rate": 0,
                "average_score": 0,
                "common_issues": []
            }
        
        successful = sum(1 for r in self.validation_history if r.is_valid)
        success_rate = (successful / total_validations) * 100
        
        avg_score = sum(r.overall_score for r in self.validation_history) / total_validations
        
        # Count common issues
        issue_counts = {}
        for report in self.validation_history:
            for result in report.results:
                if not result.passed:
                    issue_counts[result.rule_id] = issue_counts.get(result.rule_id, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_validations": total_validations,
            "success_rate": success_rate,
            "average_score": avg_score,
            "common_issues": common_issues,
            "enabled_rules": sum(1 for r in self.validation_rules.values() if r.enabled),
            "total_rules": len(self.validation_rules)
        }
    
    def get_validation_report(self, validation_id: str) -> Optional[ValidationReport]:
        """Get a specific validation report."""
        for report in self.validation_history:
            if report.validation_id == validation_id:
                return report
        return None
    
    def get_skill_validation_history(self, skill_id: str) -> List[ValidationReport]:
        """Get validation history for a specific skill."""
        return [r for r in self.validation_history if r.skill_id == skill_id]