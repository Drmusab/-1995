"""
Advanced Logging Filters for AI Assistant
Author: Drmusab
Last Modified: 2025-06-26 12:38:56 UTC

This module provides comprehensive logging filters for the AI assistant,
enabling intelligent log filtering, sensitive data masking, context-aware
filtering, performance-based filtering, and dynamic filter management.
"""

import re
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Pattern, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json
import hashlib
import weakref
from abc import ABC, abstractmethod

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    LogFilterApplied, LogDataMasked, LogRateLimited, 
    SensitiveDataDetected, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.security.sanitization import SecuritySanitizer

# Memory systems for context-aware filtering
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager

# Learning for adaptive filtering
from src.learning.continual_learning import ContinualLearner
from src.learning.preference_learning import PreferenceLearner

# Observability
from src.observability.monitoring.metrics import MetricsCollector


class FilterAction(Enum):
    """Actions that can be taken by filters."""
    ALLOW = "allow"
    DENY = "deny"
    MASK = "mask"
    TRANSFORM = "transform"
    THROTTLE = "throttle"
    REDIRECT = "redirect"
    ANNOTATE = "annotate"


class SensitivityLevel(Enum):
    """Levels of data sensitivity."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    SECRET = 4


class FilterPriority(Enum):
    """Filter execution priorities."""
    CRITICAL = 0      # Security and compliance filters
    HIGH = 1          # Performance and rate limiting
    NORMAL = 2        # Standard filtering
    LOW = 3           # Decorative filters
    BACKGROUND = 4    # Analytics and learning


@dataclass
class FilterRule:
    """Configuration for a specific filter rule."""
    rule_id: str
    name: str
    pattern: Optional[Union[str, Pattern]] = None
    condition: Optional[Callable[[logging.LogRecord], bool]] = None
    action: FilterAction = FilterAction.ALLOW
    priority: FilterPriority = FilterPriority.NORMAL
    enabled: bool = True
    
    # Masking configuration
    mask_pattern: Optional[str] = None
    mask_replacement: str = "[MASKED]"
    preserve_length: bool = False
    preserve_format: bool = True
    
    # Rate limiting configuration
    rate_limit: Optional[int] = None  # messages per time window
    time_window: float = 60.0  # seconds
    
    # Context configuration
    requires_context: bool = False
    context_keys: Set[str] = field(default_factory=set)
    user_specific: bool = False
    session_specific: bool = False
    
    # Metadata
    description: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Set[str] = field(default_factory=set)


@dataclass
class FilterResult:
    """Result of applying a filter."""
    action_taken: FilterAction
    rule_id: str
    original_message: str
    filtered_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    sensitivity_detected: Optional[SensitivityLevel] = None
    context_used: Dict[str, Any] = field(default_factory=dict)


class LogFilter(ABC):
    """Abstract base class for all log filters."""
    
    def __init__(self, rule: FilterRule):
        self.rule = rule
        self.enabled = rule.enabled
        self.metrics: Dict[str, int] = defaultdict(int)
        self.last_activity = time.time()
    
    @abstractmethod
    def should_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Determine if this filter should be applied to the record."""
        pass
    
    @abstractmethod
    def apply_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterResult:
        """Apply the filter to the log record."""
        pass
    
    def is_enabled(self) -> bool:
        """Check if filter is enabled."""
        return self.enabled and self.rule.enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get filter performance metrics."""
        return {
            "rule_id": self.rule.rule_id,
            "name": self.rule.name,
            "enabled": self.enabled,
            "metrics": dict(self.metrics),
            "last_activity": self.last_activity
        }


class SensitiveDataFilter(LogFilter):
    """Filter for detecting and masking sensitive data."""
    
    # Common sensitive data patterns
    SENSITIVE_PATTERNS = {
        "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
        "ssn": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
        "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
        "api_key": re.compile(r'\b[A-Za-z0-9]{32,}\b'),
        "password": re.compile(r'password["\'\s]*[:=]["\'\s]*[^\s"\']+', re.IGNORECASE),
        "token": re.compile(r'token["\'\s]*[:=]["\'\s]*[^\s"\']+', re.IGNORECASE),
        "secret": re.compile(r'secret["\'\s]*[:=]["\'\s]*[^\s"\']+', re.IGNORECASE),
        "ip_address": re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        "session_id": re.compile(r'session[_-]?id["\'\s]*[:=]["\'\s]*[^\s"\']+', re.IGNORECASE),
        "user_id": re.compile(r'user[_-]?id["\'\s]*[:=]["\'\s]*[^\s"\']+', re.IGNORECASE)
    }
    
    def __init__(self, rule: FilterRule, security_sanitizer: Optional[SecuritySanitizer] = None):
        super().__init__(rule)
        self.security_sanitizer = security_sanitizer
        self.custom_patterns: Dict[str, Pattern] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile additional patterns from rule configuration."""
        if self.rule.pattern:
            if isinstance(self.rule.pattern, str):
                self.custom_patterns["custom"] = re.compile(self.rule.pattern)
            else:
                self.custom_patterns["custom"] = self.rule.pattern
    
    def should_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Check if record contains sensitive data."""
        message = self._get_full_message(record)
        
        # Check built-in patterns
        for pattern_name, pattern in self.SENSITIVE_PATTERNS.items():
            if pattern.search(message):
                return True
        
        # Check custom patterns
        for pattern_name, pattern in self.custom_patterns.items():
            if pattern.search(message):
                return True
        
        # Use security sanitizer if available
        if self.security_sanitizer:
            try:
                risk_level = self.security_sanitizer.assess_risk(message)
                if risk_level > 0.5:  # Threshold for sensitive data
                    return True
            except Exception:
                pass  # Continue with other checks
        
        return False
    
    def apply_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterResult:
        """Apply sensitive data masking."""
        start_time = time.time()
        original_message = self._get_full_message(record)
        filtered_message = original_message
        detected_types = []
        sensitivity = SensitivityLevel.PUBLIC
        
        # Apply masking for each pattern type
        for pattern_name, pattern in {**self.SENSITIVE_PATTERNS, **self.custom_patterns}.items():
            matches = list(pattern.finditer(filtered_message))
            if matches:
                detected_types.append(pattern_name)
                
                # Determine sensitivity level
                if pattern_name in ["ssn", "credit_card", "password", "secret"]:
                    sensitivity = max(sensitivity, SensitivityLevel.SECRET)
                elif pattern_name in ["api_key", "token", "session_id"]:
                    sensitivity = max(sensitivity, SensitivityLevel.RESTRICTED)
                elif pattern_name in ["email", "phone", "user_id"]:
                    sensitivity = max(sensitivity, SensitivityLevel.CONFIDENTIAL)
                else:
                    sensitivity = max(sensitivity, SensitivityLevel.INTERNAL)
                
                # Apply masking
                for match in reversed(matches):  # Reverse to maintain indices
                    replacement = self._generate_mask(match.group(), pattern_name)
                    filtered_message = (
                        filtered_message[:match.start()] + 
                        replacement + 
                        filtered_message[match.end():]
                    )
        
        # Update the log record
        self._update_record_message(record, filtered_message)
        
        # Update metrics
        self.metrics["messages_masked"] += 1
        self.metrics["sensitive_patterns_detected"] += len(detected_types)
        self.last_activity = time.time()
        
        processing_time = time.time() - start_time
        
        return FilterResult(
            action_taken=FilterAction.MASK,
            rule_id=self.rule.rule_id,
            original_message=original_message,
            filtered_message=filtered_message,
            metadata={
                "detected_types": detected_types,
                "patterns_matched": len(detected_types)
            },
            processing_time=processing_time,
            sensitivity_detected=sensitivity
        )
    
    def _generate_mask(self, original: str, pattern_type: str) -> str:
        """Generate appropriate mask for detected sensitive data."""
        if self.rule.preserve_length:
            if self.rule.preserve_format and pattern_type == "credit_card":
                # Keep format like **** **** **** 1234
                if len(original) >= 4:
                    return "*" * (len(original) - 4) + original[-4:]
            elif self.rule.preserve_format and pattern_type == "phone":
                # Keep format like (***) ***-1234
                if len(original) >= 4:
                    return re.sub(r'\d', '*', original[:-4]) + original[-4:]
            else:
                return "*" * len(original)
        else:
            return self.rule.mask_replacement
    
    def _get_full_message(self, record: logging.LogRecord) -> str:
        """Get the complete message including formatted parts."""
        try:
            message = record.getMessage()
            
            # Include exception info if present
            if record.exc_info:
                import traceback
                message += "\n" + "".join(traceback.format_exception(*record.exc_info))
            
            return message
        except Exception:
            return str(record.msg)
    
    def _update_record_message(self, record: logging.LogRecord, new_message: str) -> None:
        """Update the log record with filtered message."""
        record.msg = new_message
        record.args = ()  # Clear args to prevent re-formatting


class RateLimitFilter(LogFilter):
    """Filter for rate limiting log messages."""
    
    def __init__(self, rule: FilterRule):
        super().__init__(rule)
        self.message_counts: Dict[str, deque] = defaultdict(lambda: deque())
        self.blocked_until: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def should_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Check if message should be rate limited."""
        if not self.rule.rate_limit:
            return False
        
        key = self._get_rate_limit_key(record, context)
        current_time = time.time()
        
        with self.lock:
            # Check if currently blocked
            if key in self.blocked_until and current_time < self.blocked_until[key]:
                return True
            
            # Clean old entries
            message_times = self.message_counts[key]
            cutoff_time = current_time - self.rule.time_window
            
            while message_times and message_times[0] < cutoff_time:
                message_times.popleft()
            
            # Check rate limit
            if len(message_times) >= self.rule.rate_limit:
                # Block for remaining window time
                self.blocked_until[key] = current_time + self.rule.time_window
                return True
            
            # Add current message
            message_times.append(current_time)
            return False
    
    def apply_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterResult:
        """Apply rate limiting."""
        start_time = time.time()
        original_message = record.getMessage()
        
        key = self._get_rate_limit_key(record, context)
        
        # Update metrics
        self.metrics["messages_rate_limited"] += 1
        self.last_activity = time.time()
        
        processing_time = time.time() - start_time
        
        return FilterResult(
            action_taken=FilterAction.THROTTLE,
            rule_id=self.rule.rule_id,
            original_message=original_message,
            metadata={
                "rate_limit_key": key,
                "current_rate": len(self.message_counts[key])
            },
            processing_time=processing_time
        )
    
    def _get_rate_limit_key(self, record: logging.LogRecord, context: Dict[str, Any]) -> str:
        """Generate rate limiting key based on context."""
        key_parts = []
        
        # Include logger name
        key_parts.append(record.name)
        
        # Include log level
        key_parts.append(record.levelname)
        
        # Include user/session if configured
        if self.rule.user_specific and "user_id" in context:
            key_parts.append(f"user:{context['user_id']}")
        
        if self.rule.session_specific and "session_id" in context:
            key_parts.append(f"session:{context['session_id']}")
        
        # Include message pattern for similar messages
        message_hash = hashlib.md5(record.msg.encode()).hexdigest()[:8]
        key_parts.append(f"msg:{message_hash}")
        
        return "|".join(key_parts)


class ContextualFilter(LogFilter):
    """Filter based on contextual information."""
    
    def __init__(self, rule: FilterRule, context_manager: Optional[ContextManager] = None):
        super().__init__(rule)
        self.context_manager = context_manager
        self.context_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_cache_cleanup = time.time()
    
    def should_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Check if context-based filtering should be applied."""
        if not self.rule.requires_context:
            return self._evaluate_condition(record, context)
        
        # Check if required context keys are present
        for key in self.rule.context_keys:
            if key not in context:
                return False
        
        # Get additional context if available
        enhanced_context = self._get_enhanced_context(context)
        
        return self._evaluate_condition(record, enhanced_context)
    
    def apply_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterResult:
        """Apply contextual filtering."""
        start_time = time.time()
        original_message = record.getMessage()
        
        enhanced_context = self._get_enhanced_context(context)
        action = self._determine_action(record, enhanced_context)
        
        filtered_message = original_message
        if action == FilterAction.TRANSFORM:
            filtered_message = self._transform_message(record, enhanced_context)
        elif action == FilterAction.ANNOTATE:
            filtered_message = self._annotate_message(record, enhanced_context)
        
        # Update the record if message was modified
        if filtered_message != original_message:
            record.msg = filtered_message
            record.args = ()
        
        # Update metrics
        self.metrics[f"action_{action.value}"] += 1
        self.last_activity = time.time()
        
        processing_time = time.time() - start_time
        
        return FilterResult(
            action_taken=action,
            rule_id=self.rule.rule_id,
            original_message=original_message,
            filtered_message=filtered_message if filtered_message != original_message else None,
            context_used=enhanced_context,
            processing_time=processing_time
        )
    
    def _get_enhanced_context(self, base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced context information."""
        enhanced = base_context.copy()
        
        # Get session context if available
        if self.context_manager and "session_id" in base_context:
            session_id = base_context["session_id"]
            
            # Check cache first
            if session_id in self.context_cache:
                cache_entry = self.context_cache[session_id]
                if time.time() - cache_entry.get("timestamp", 0) < self.cache_ttl:
                    enhanced.update(cache_entry.get("data", {}))
                    return enhanced
            
            # Fetch from context manager
            try:
                session_context = self.context_manager.get_session_context(session_id)
                if session_context:
                    enhanced.update(session_context)
                    
                    # Cache the result
                    self.context_cache[session_id] = {
                        "data": session_context,
                        "timestamp": time.time()
                    }
            except Exception:
                pass  # Continue without enhanced context
        
        # Cleanup old cache entries periodically
        current_time = time.time()
        if current_time - self.last_cache_cleanup > self.cache_ttl:
            self._cleanup_cache()
            self.last_cache_cleanup = current_time
        
        return enhanced
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.context_cache.items()
            if current_time - entry.get("timestamp", 0) > self.cache_ttl
        ]
        for key in expired_keys:
            del self.context_cache[key]
    
    def _evaluate_condition(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Evaluate the filter condition."""
        if self.rule.condition:
            try:
                return self.rule.condition(record)
            except Exception:
                return False
        
        # Default pattern matching
        if self.rule.pattern:
            message = record.getMessage()
            if isinstance(self.rule.pattern, str):
                return self.rule.pattern in message
            else:
                return bool(self.rule.pattern.search(message))
        
        return True
    
    def _determine_action(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterAction:
        """Determine what action to take based on context."""
        # Check for emergency conditions
        if context.get("system_state") == "emergency":
            return FilterAction.ALLOW  # Always allow emergency logs
        
        # Check user preferences
        user_prefs = context.get("user_preferences", {})
        if user_prefs.get("detailed_logging", False):
            return FilterAction.ANNOTATE
        elif user_prefs.get("minimal_logging", False):
            return FilterAction.TRANSFORM
        
        # Check log level and component
        if record.levelno >= logging.ERROR:
            return FilterAction.ANNOTATE  # Add context to errors
        elif record.name.startswith("src.assistant"):
            return FilterAction.TRANSFORM  # Transform assistant logs
        
        return self.rule.action
    
    def _transform_message(self, record: logging.LogRecord, context: Dict[str, Any]) -> str:
        """Transform the log message based on context."""
        message = record.getMessage()
        
        # Add session information
        if "session_id" in context:
            session_id = context["session_id"][-8:]  # Last 8 chars
            message = f"[session:{session_id}] {message}"
        
        # Add user information
        if "user_id" in context:
            user_id = context["user_id"]
            message = f"[user:{user_id}] {message}"
        
        # Add component information
        if "component" in context:
            component = context["component"]
            message = f"[{component}] {message}"
        
        return message
    
    def _annotate_message(self, record: logging.LogRecord, context: Dict[str, Any]) -> str:
        """Annotate the log message with additional context."""
        message = record.getMessage()
        annotations = []
        
        # Add performance context
        if "processing_time" in context:
            processing_time = context["processing_time"]
            annotations.append(f"took:{processing_time:.3f}s")
        
        # Add memory context
        if "memory_usage" in context:
            memory_usage = context["memory_usage"]
            annotations.append(f"mem:{memory_usage:.1f}MB")
        
        # Add interaction context
        if "interaction_id" in context:
            interaction_id = context["interaction_id"][-8:]
            annotations.append(f"interaction:{interaction_id}")
        
        if annotations:
            message += f" [{', '.join(annotations)}]"
        
        return message


class PerformanceFilter(LogFilter):
    """Filter based on performance metrics and system load."""
    
    def __init__(self, rule: FilterRule, metrics: Optional[MetricsCollector] = None):
        super().__init__(rule)
        self.metrics_collector = metrics
        self.performance_history: deque = deque(maxlen=100)
        self.load_threshold = 0.8  # 80% system load threshold
    
    def should_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Check if performance-based filtering should be applied."""
        # Check system load
        current_load = self._get_system_load()
        
        # If system is under high load, filter non-essential logs
        if current_load > self.load_threshold:
            return record.levelno < logging.WARNING
        
        # Check message processing time
        processing_time = context.get("processing_time", 0)
        if processing_time > 1.0:  # Slow operations should be logged
            return False
        
        # Check if this is a performance-critical component
        if record.name.startswith(("src.processing", "src.memory")):
            return record.levelno < logging.INFO
        
        return False
    
    def apply_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterResult:
        """Apply performance-based filtering."""
        start_time = time.time()
        original_message = record.getMessage()
        
        system_load = self._get_system_load()
        
        # Determine action based on system state
        if system_load > 0.9:
            action = FilterAction.DENY
        elif system_load > self.load_threshold:
            action = FilterAction.THROTTLE
        else:
            action = FilterAction.ALLOW
        
        # Update metrics
        self.metrics["performance_filters_applied"] += 1
        self.performance_history.append({
            "timestamp": time.time(),
            "system_load": system_load,
            "action": action.value
        })
        self.last_activity = time.time()
        
        processing_time = time.time() - start_time
        
        return FilterResult(
            action_taken=action,
            rule_id=self.rule.rule_id,
            original_message=original_message,
            metadata={
                "system_load": system_load,
                "load_threshold": self.load_threshold
            },
            processing_time=processing_time
        )
    
    def _get_system_load(self) -> float:
        """Get current system load metric."""
        try:
            # Try to get from metrics collector
            if self.metrics_collector:
                return self.metrics_collector.get_gauge("system_cpu_usage") or 0.0
            
            # Fallback to basic CPU usage
            import psutil
            return psutil.cpu_percent(interval=0.1) / 100.0
        except Exception:
            return 0.0  # Default to no load if measurement fails


class AdaptiveFilter(LogFilter):
    """Adaptive filter that learns from patterns and user feedback."""
    
    def __init__(self, rule: FilterRule, learner: Optional[ContinualLearner] = None):
        super().__init__(rule)
        self.learner = learner
        self.pattern_history: deque = deque(maxlen=1000)
        self.feedback_history: deque = deque(maxlen=100)
        self.adaptation_threshold = 10  # Minimum samples for adaptation
    
    def should_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Use learned patterns to determine filtering."""
        # Extract features from log record
        features = self._extract_features(record, context)
        
        # Use learned model if available
        if self.learner and len(self.pattern_history) > self.adaptation_threshold:
            try:
                prediction = self.learner.predict_filter_action(features)
                return prediction.get("should_filter", False)
            except Exception:
                pass
        
        # Fallback to rule-based logic
        return self._rule_based_decision(record, context)
    
    def apply_filter(self, record: logging.LogRecord, context: Dict[str, Any]) -> FilterResult:
        """Apply adaptive filtering with learning."""
        start_time = time.time()
        original_message = record.getMessage()
        
        features = self._extract_features(record, context)
        action = self._determine_adaptive_action(record, context, features)
        
        # Store pattern for learning
        pattern_data = {
            "timestamp": time.time(),
            "features": features,
            "action": action.value,
            "context": context.copy()
        }
        self.pattern_history.append(pattern_data)
        
        # Apply action
        filtered_message = original_message
        if action == FilterAction.TRANSFORM:
            filtered_message = self._adaptive_transform(record, features)
        
        # Update record if needed
        if filtered_message != original_message:
            record.msg = filtered_message
            record.args = ()
        
        # Update metrics
        self.metrics[f"adaptive_{action.value}"] += 1
        self.last_activity = time.time()
        
        processing_time = time.time() - start_time
        
        return FilterResult(
            action_taken=action,
            rule_id=self.rule.rule_id,
            original_message=original_message,
            filtered_message=filtered_message if filtered_message != original_message else None,
            metadata={
                "features": features,
                "learned_pattern": len(self.pattern_history) > self.adaptation_threshold
            },
            processing_time=processing_time
        )
    
    def _extract_features(self, record: logging.LogRecord, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for learning."""
        return {
            "level": record.levelno,
            "logger_name": record.name,
            "message_length": len(record.getMessage()),
            "has_exception": record.exc_info is not None,
            "hour_of_day": datetime.now().hour,
            "user_id": context.get("user_id"),
            "session_id": context.get("session_id"),
            "component": context.get("component"),
            "processing_time": context.get("processing_time", 0),
            "system_load": context.get("system_load", 0)
        }
    
    def _rule_based_decision(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """Fallback rule-based decision logic."""
        # Apply basic rules
        if record.levelno >= logging.ERROR:
            return False  # Always allow errors
        
        if record.name.startswith("src.observability"):
            return True  # Filter observability logs by default
        
        return False
    
    def _determine_adaptive_action(self, record: logging.LogRecord, 
                                  context: Dict[str, Any], features: Dict[str, Any]) -> FilterAction:
        """Determine action using adaptive logic."""
        # Use learned preferences if available
        if self.learner:
            try:
                prediction = self.learner.predict_filter_action(features)
                if prediction and "action" in prediction:
                    return FilterAction(prediction["action"])
            except Exception:
                pass
        
        # Fallback to heuristics
        if features.get("processing_time", 0) > 2.0:
            return FilterAction.ANNOTATE  # Annotate slow operations
        elif features.get("system_load", 0) > 0.8:
            return FilterAction.THROTTLE  # Throttle under high load
        else:
            return FilterAction.ALLOW
    
    def _adaptive_transform(self, record: logging.LogRecord, features: Dict[str, Any]) -> str:
        """Transform message based on learned patterns."""
        message = record.getMessage()
        
        # Add adaptive annotations based on features
        if features.get("processing_time", 0) > 1.0:
            message += f" [slow:{features['processing_time']:.2f}s]"
        
        if features.get("system_load", 0) > 0.5:
            message += f" [load:{features['system_load']:.1f}]"
        
        return message
    
    def provide_feedback(self, pattern_id: str, feedback: str, user_id: Optional[str] = None) -> None:
        """Provide feedback for learning."""
        feedback_data = {
            "timestamp": time.time(),
            "pattern_id": pattern_id,
            "feedback": feedback,
            "user_id": user_id
        }
        self.feedback_history.append(feedback_data)
        
        # Trigger learning update if enough feedback
        if len(self.feedback_history) >= 10:
            self._update_learning()
    
    def _update_learning(self) -> None:
        """Update the learning model with collected patterns and feedback."""
        if not self.learner:
            return
        
        try:
            # Prepare training data
            training_data = {
                "patterns": list(self.pattern_history),
                "feedback": list(self.feedback_history)
            }
            
            # Update learner
            self.learner.update_filter_model(training_data)
            
            # Clear processed feedback
            self.feedback_history.clear()
            
        except Exception as e:
            # Log learning error but continue operation
            pass


class EnhancedLogFilterManager:
    """
    Advanced Log Filter Manager for the AI Assistant.
    
    This manager coordinates multiple log filters to provide:
    - Sensitive data protection and masking
    - Intelligent rate limiting and throttling
    - Context-aware filtering based on user/session
    - Performance-based filtering under system load
    - Adaptive filtering with machine learning
    - Real-time filter rule management
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self, container: Container):
        """
        Initialize the enhanced log filter manager.
        
        Args:
            container: Dependency injection container
        """
        self.container = container
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        
        # Optional dependencies
        try:
            self.security_sanitizer = container.get(SecuritySanitizer)
            self.context_manager = container.get(ContextManager)
            self.continual_learner = container.get(ContinualLearner)
            self.metrics = container.get(MetricsCollector)
        except Exception:
            self.security_sanitizer = None
            self.context_manager = None
            self.continual_learner = None
            self.metrics = None
        
        # Filter management
        self.filters: Dict[str, LogFilter] = {}
        self.filter_rules: Dict[str, FilterRule] = {}
        self.filter_chain: List[LogFilter] = []
        
        # Performance tracking
        self.filter_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.processing_times: deque = deque(maxlen=1000)
        
        # Configuration
        self.load_configuration()
        self.setup_default_filters()
        
        # Metrics
        if self.metrics:
            self.setup_metrics()
    
    def load_configuration(self) -> None:
        """Load filter configuration from settings."""
        filter_config = self.config.get("logging.filters", {})
        
        # Load predefined filter rules
        for rule_data in filter_config.get("rules", []):
            rule = FilterRule(**rule_data)
            self.filter_rules[rule.rule_id] = rule
    
    def setup_default_filters(self) -> None:
        """Setup default filters for common use cases."""
        # Sensitive data filter
        sensitive_rule = FilterRule(
            rule_id="default_sensitive_data",
            name="Sensitive Data Masking",
            action=FilterAction.MASK,
            priority=FilterPriority.CRITICAL,
            description="Mask sensitive data in all log messages"
        )
        self.add_filter_rule(sensitive_rule)
        
        # Rate limiting filter
        rate_limit_rule = FilterRule(
            rule_id="default_rate_limit",
            name="Rate Limiting",
            action=FilterAction.THROTTLE,
            priority=FilterPriority.HIGH,
            rate_limit=100,  # 100 messages per minute
            time_window=60.0,
            description="Rate limit excessive logging"
        )
        self.add_filter_rule(rate_limit_rule)
        
        # Performance filter
        performance_rule = FilterRule(
            rule_id="default_performance",
            name="Performance Filtering",
            action=FilterAction.THROTTLE,
            priority=FilterPriority.HIGH,
            description="Filter logs based on system performance"
        )
        self.add_filter_rule(performance_rule)
        
        # Contextual filter for assistant components
        contextual_rule = FilterRule(
            rule_id="assistant_contextual",
            name="Assistant Context Filter",
            action=FilterAction.ANNOTATE,
            priority=FilterPriority.NORMAL,
            requires_context=True,
            context_keys={"session_id", "user_id"},
            description="Add context to assistant component logs"
        )
        self.add_filter_rule(contextual_rule)
        
        # Adaptive filter
        adaptive_rule = FilterRule(
            rule_id="adaptive_learning",
            name="Adaptive Learning Filter",
            action=FilterAction.TRANSFORM,
            priority=FilterPriority.LOW,
            description="Adaptive filter that learns from patterns"
        )
        self.add_filter_rule(adaptive_rule)
    
    def setup_metrics(self) -> None:
        """Setup metrics for filter performance monitoring."""
        self.metrics.register_counter("log_filter_applications_total")
        self.metrics.register_counter("log_filter_sensitive_data_masked")
        self.metrics.register_counter("log_filter_rate_limited")
        self.metrics.register_histogram("log_filter_processing_time_seconds")
        self.metrics.register_gauge("log_filter_active_filters")
    
    def add_filter_rule(self, rule: FilterRule) -> None:
        """Add a new filter rule and create corresponding filter."""
        self.filter_rules[rule.rule_id] = rule
        
        # Create appropriate filter instance
        filter_instance = self._create_filter_instance(rule)
        if filter_instance:
            self.filters[rule.rule_id] = filter_instance
            self._rebuild_filter_chain()
    
    def _create_filter_instance(self, rule: FilterRule) -> Optional[LogFilter]:
        """Create the appropriate filter instance for a rule."""
        try:
            if rule.rule_id.startswith("sensitive") or "sensitive" in rule.name.lower():
                return SensitiveDataFilter(rule, self.security_sanitizer)
            elif rule.rule_id.startswith("rate") or "rate" in rule.name.lower():
                return RateLimitFilter(rule)
            elif rule.rule_id.startswith("performance") or "performance" in rule.name.lower():
                return PerformanceFilter(rule, self.metrics)
            elif rule.rule_id.startswith("contextual") or rule.requires_context:
                return ContextualFilter(rule, self.context_manager)
            elif rule.rule_id.startswith("adaptive") or "adaptive" in rule.name.lower():
                return AdaptiveFilter(rule, self.continual_learner)
            else:
                # Generic contextual filter for others
                return ContextualFilter(rule, self.context_manager)
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(e, {"rule_id": rule.rule_id})
            return None
    
    def _rebuild_filter_chain(self) -> None:
        """Rebuild the filter chain based on priorities."""
        # Sort filters by priority
        sorted_filters = sorted(
            self.filters.values(),
            key=lambda f: (f.rule.priority.value, f.rule.rule_id)
        )
        
        # Only include enabled filters
        self.filter_chain = [f for f in sorted_filters if f.is_enabled()]
        
        # Update metrics
        if self.metrics:
            self.metrics.set("log_filter_active_filters", len(self.filter_chain))
    
    def remove_filter_rule(self, rule_id: str) -> bool:
        """Remove a filter rule and its instance."""
        if rule_id in self.filter_rules:
            del self.filter_rules[rule_id]
        
        if rule_id in self.filters:
            del self.filters[rule_id]
            self._rebuild_filter_chain()
            return True
        
        return False
    
    def enable_filter(self, rule_id: str) -> bool:
        """Enable a specific filter."""
        if rule_id in self.filters:
            self.filters[rule_id].enabled = True
            self._rebuild_filter_chain()
            return True
        return False
    
    def disable_filter(self, rule_id: str) -> bool:
        """Disable a specific filter."""
        if rule_id in self.filters:
            self.filters[rule_id].enabled = False
            self._rebuild_filter_chain()
            return True
        return False
    
    def create_log_filter(self, context: Optional[Dict[str, Any]] = None) -> Callable[[logging.LogRecord], bool]:
        """
        Create a log filter function for use with Python logging.
        
        Args:
            context: Optional context to use for filtering
            
        Returns:
            Filter function compatible with Python logging
        """
        def filter_function(record: logging.LogRecord) -> bool:
            return self.filter_log_record(record, context or {})
        
        return filter_function
    
    @handle_exceptions
    def filter_log_record(self, record: logging.LogRecord, context: Dict[str, Any]) -> bool:
        """
        Filter a log record through the filter chain.
        
        Args:
            record: Log record to filter
            context: Context information for filtering
            
        Returns:
            True if record should be logged, False otherwise
        """
        start_time = time.time()
        
        try:
            # Apply each filter in the chain
            for log_filter in self.filter_chain:
                if not log_filter.should_filter(record, context):
                    continue
                
                # Apply the filter
                result = log_filter.apply_filter(record, context)
                
                # Emit events for significant actions
                if result.action_taken == FilterAction.MASK:
                    asyncio.create_task(self.event_bus.emit(LogDataMasked(
                        filter_id=result.rule_id,
                        sensitivity_level=result.sensitivity_detected.value if result.sensitivity_detected else "unknown",
                        patterns_detected=result.metadata.get("detected_types", [])
                    )))
                elif result.action_taken == FilterAction.THROTTLE:
                    asyncio.create_task(self.event_bus.emit(LogRateLimited(
                        filter_id=result.rule_id,
                        rate_limit_key=result.metadata.get("rate_limit_key", ""),
                        current_rate=result.metadata.get("current_rate", 0)
                    )))
                
                # Handle filter actions
                if result.action_taken == FilterAction.DENY:
                    return False
                elif result.action_taken == FilterAction.THROTTLE:
                    # For throttling, we still deny but emit event
                    return False
                
                # For other actions (MASK, TRANSFORM, ANNOTATE), continue processing
                # The filter has already modified the record
            
            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update metrics
            if self.metrics:
                self.metrics.increment("log_filter_applications_total")
                self.metrics.record("log_filter_processing_time_seconds", processing_time)
            
            return True  # Allow the record
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(e, {"record": str(record)})
            return True  # Allow on error to prevent log loss
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about filter performance."""
        stats = {
            "total_filters": len(self.filters),
            "active_filters": len(self.filter_chain),
            "processing_times": {
                "average": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                "max": max(self.processing_times) if self.processing_times else 0,
                "min": min(self.processing_times) if self.processing_times else 0
            },
            "filters": {}
        }
        
        # Individual filter statistics
        for filter_id, log_filter in self.filters.items():
            stats["filters"][filter_id] = log_filter.get_metrics()
        
        return stats
    
    def update_filter_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing filter rule."""
        if rule_id not in self.filter_rules:
            return False
        
        rule = self.filter_rules[rule_id]
        
        # Update rule attributes
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        # Recreate filter instance with updated rule
        new_filter = self._create_filter_instance(rule)
        if new_filter:
            self.filters[rule_id] = new_filter
            self._rebuild_filter_chain()
            return True
        
        return False
    
    def list_filter_rules(self) -> List[Dict[str, Any]]:
        """List all filter rules with their current status."""
        rules = []
        for rule in self.filter_rules.values():
            rule_info = {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "action": rule.action.value,
                "priority": rule.priority.value,
                "enabled": rule.enabled,
                "description": rule.description,
                "created_at": rule.created_at.isoformat()
            }
            
            # Add filter-specific information
            if rule.rule_id in self.filters:
                filter_metrics = self.filters[rule.rule_id].get_metrics()
                rule_info["metrics"] = filter_metrics
            
            rules.append(rule_info)
        
        return rules
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export the current filter configuration."""
        return {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "pattern": rule.pattern if isinstance(rule.pattern, str) else None,
                    "action": rule.action.value,
                    "priority": rule.priority.value,
                    "enabled": rule.enabled,
                    "mask_replacement": rule.mask_replacement,
                    "preserve_length": rule.preserve_length,
                    "rate_limit": rule.rate_limit,
                    "time_window": rule.time_window,
                    "requires_context": rule.requires_context,
                    "context_keys": list(rule.context_keys),
                    "description": rule.description,
                    "category": rule.category,
                    "tags": list(rule.tags)
                }
                for rule in self.filter_rules.values()
            ],
            "statistics": self.get_filter_statistics()
        }
    
    def import_configuration(self, config: Dict[str, Any]) -> int:
        """Import filter configuration and return number of rules imported."""
        imported_count = 0
        
        for rule_data in config.get("rules", []):
            try:
                # Convert string enums back to enum instances
                if "action" in rule_data:
                    rule_data["action"] = FilterAction(rule_data["action"])
                if "priority" in rule_data:
                    rule_data["priority"] = FilterPriority(rule_data["priority"])
                
                # Convert lists back to sets
                if "context_keys" in rule_data:
                    rule_data["context_keys"] = set(rule_data["context_keys"])
                if "tags" in rule_data:
                    rule_data["tags"] = set(rule_data["tags"])
                
                rule = FilterRule(**rule_data)
                self.add_filter_rule(rule)
                imported_count += 1
                
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(e, {"rule_data": rule_data})
        
        return imported_count
    
    def cleanup(self) -> None:
        """Cleanup resources used by filters."""
        for log_filter in self.filters.values():
            if hasattr(log_filter, 'cleanup'):
                try:
                    log_filter.cleanup()
                except Exception as e:
                    if self.error_handler:
                        self.error_handler.handle_error(e, {"filter": log_filter.rule.rule_id})
        
        self.filters.clear()
        self.filter_chain.clear()
        self.filter_rules.clear()


# Factory function for easy integration
def create_log_filter_manager(container: Container) -> EnhancedLogFilterManager:
    """
    Factory function to create a log filter manager.
    
    Args:
        container: Dependency injection container
        
    Returns:
        Configured log filter manager
    """
    return EnhancedLogFilterManager(container)


# Utility functions for common filter patterns
def create_sensitive_data_filter(patterns: Optional[Dict[str, str]] = None) -> FilterRule:
    """Create a sensitive data filter rule with custom patterns."""
    return FilterRule(
        rule_id=f"sensitive_data_{int(time.time())}",
        name="Custom Sensitive Data Filter",
        action=FilterAction.MASK,
        priority=FilterPriority.CRITICAL,
        pattern="|".join(patterns.values()) if patterns else None,
        description="Custom sensitive data masking filter"
    )


def create_rate_limit_filter(rate: int, window: float = 60.0, 
                           user_specific: bool = False) -> FilterRule:
    """Create a rate limiting filter rule."""
    return FilterRule(
        rule_id=f"rate_limit_{int(time.time())}",
        name=f"Rate Limit {rate}/{window}s",
        action=FilterAction.THROTTLE,
        priority=FilterPriority.HIGH,
        rate_limit=rate,
        time_window=window,
        user_specific=user_specific,
        description=f"Rate limit to {rate} messages per {window} seconds"
    )


def create_context_filter(context_keys: List[str], 
                         action: FilterAction = FilterAction.ANNOTATE) -> FilterRule:
    """Create a context-aware filter rule."""
    return FilterRule(
        rule_id=f"context_{int(time.time())}",
        name="Context-Aware Filter",
        action=action,
        priority=FilterPriority.NORMAL,
        requires_context=True,
        context_keys=set(context_keys),
        description=f"Context-aware filter using keys: {', '.join(context_keys)}"
    )
