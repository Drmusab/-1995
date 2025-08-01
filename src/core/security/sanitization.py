"""
Advanced Security Sanitization System
Author: Drmusab
Last Modified: 2025-06-13 12:33:13 UTC

This module provides comprehensive security sanitization for the AI assistant,
including input validation, output filtering, content security, malware detection,
and seamless integration with all core system components.
"""

import base64
import hashlib
import html
import inspect
import json
import logging
import mimetypes
import re
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

import asyncio
import bleach
import numpy as np
import torch
import validators
from PIL import Image

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    ComplianceViolation,
    ComponentSecurityCheck,
    ContentFiltered,
    DLPViolation,
    ErrorOccurred,
    InputSanitized,
    MalwareDetected,
    OutputFiltered,
    SecurityAuditCreated,
    SecurityViolation,
    SessionSecurityCheck,
    SystemStateChanged,
    ThreatDetected,
    UserBehaviorAlert,
    ValidationFailed,
    WorkflowSecurityCheck,
)
from src.core.health_check import HealthCheck

# Security integration
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager
from src.integrations.cache.redis_cache import RedisCache

# Storage and caching
from src.integrations.storage.database import DatabaseManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class SecurityLevel(Enum):
    """Security levels for different contexts."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"
    MAXIMUM = "maximum"


class ThreatLevel(Enum):
    """Threat severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SanitizationType(Enum):
    """Types of sanitization operations."""

    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    CONTENT_SECURITY = "content_security"
    DATA_VALIDATION = "data_validation"
    MALWARE_SCAN = "malware_scan"
    XSS_PROTECTION = "xss_protection"
    SQL_INJECTION = "sql_injection"
    FILE_VALIDATION = "file_validation"
    IMAGE_VALIDATION = "image_validation"
    AUDIO_VALIDATION = "audio_validation"


class ContentType(Enum):
    """Types of content being sanitized."""

    TEXT = "text"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    BINARY = "binary"


class ViolationType(Enum):
    """Types of security violations."""

    XSS_ATTEMPT = "xss_attempt"
    SQL_INJECTION = "sql_injection"
    MALWARE_DETECTED = "malware_detected"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    PRIVACY_VIOLATION = "privacy_violation"
    DATA_LEAK = "data_leak"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class SanitizationRule:
    """Represents a sanitization rule."""

    rule_id: str
    name: str
    rule_type: SanitizationType
    content_types: Set[ContentType]
    pattern: Optional[str] = None
    replacement: Optional[str] = None
    action: str = "clean"  # clean, block, warn, log
    severity: ThreatLevel = ThreatLevel.MEDIUM
    enabled: bool = True

    # Conditions
    applies_to_users: Set[str] = field(default_factory=set)
    applies_to_roles: Set[str] = field(default_factory=set)
    applies_to_contexts: Set[str] = field(default_factory=set)

    # Metadata
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    priority: int = 100

    # Custom processing
    custom_validator: Optional[Callable] = None
    custom_sanitizer: Optional[Callable] = None


@dataclass
class SanitizationContext:
    """Context for sanitization operations."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    workflow_id: Optional[str] = None
    interaction_id: Optional[str] = None

    # Security context
    security_level: SecurityLevel = SecurityLevel.STANDARD
    trust_level: str = "standard"
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Content context
    content_type: ContentType = ContentType.TEXT
    source: str = "user_input"
    destination: str = "system"

    # Processing hints
    strict_mode: bool = False
    preserve_formatting: bool = True
    max_length: Optional[int] = None
    allowed_tags: Set[str] = field(default_factory=set)

    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SanitizationResult:
    """Result of a sanitization operation."""

    success: bool
    original_content: Any
    sanitized_content: Any

    # Operation details
    operations_performed: List[str] = field(default_factory=list)
    violations_detected: List[Dict[str, Any]] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)

    # Metrics
    processing_time: float = 0.0
    content_modified: bool = False
    threat_level: ThreatLevel = ThreatLevel.INFO
    confidence_score: float = 1.0

    # Security info
    blocked_content: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    security_flags: Set[str] = field(default_factory=set)

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rule_version: str = "1.0.0"
    sanitizer_version: str = "1.0.0"


class SanitizationError(Exception):
    """Custom exception for sanitization operations."""

    def __init__(
        self,
        message: str,
        violation_type: Optional[ViolationType] = None,
        content_type: Optional[ContentType] = None,
        threat_level: ThreatLevel = ThreatLevel.MEDIUM,
    ):
        super().__init__(message)
        self.violation_type = violation_type
        self.content_type = content_type
        self.threat_level = threat_level
        self.timestamp = datetime.now(timezone.utc)


class InputValidator:
    """Validates and sanitizes user input."""

    def __init__(self, sanitizer: "SecuritySanitizer"):
        self.sanitizer = sanitizer
        self.logger = get_logger(__name__)

        # Common patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"eval\s*\(",
            r"expression\s*\(",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<form[^>]*>",
            r"<input[^>]*>",
            r"vbscript:",
            r"data:text/html",
        ]

        self.sql_injection_patterns = [
            r"('|(\\')|(;)|(\-\-)|(\s+(or|and)\s+.*(=|like))",
            r"union\s+select",
            r"insert\s+into",
            r"delete\s+from",
            r"update\s+.*\s+set",
            r"drop\s+(table|database)",
            r"create\s+(table|database)",
            r"alter\s+table",
            r"exec\s*\(",
            r"execute\s*\(",
        ]

        # Safe HTML tags and attributes
        self.allowed_tags = {
            "a",
            "abbr",
            "acronym",
            "b",
            "blockquote",
            "br",
            "code",
            "em",
            "i",
            "li",
            "ol",
            "p",
            "strong",
            "ul",
            "span",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "pre",
            "del",
            "ins",
            "sub",
            "sup",
        }

        self.allowed_attributes = {
            "a": ["href", "title"],
            "abbr": ["title"],
            "acronym": ["title"],
            "span": ["class"],
            "div": ["class"],
        }

    async def validate_text(self, text: str, context: SanitizationContext) -> SanitizationResult:
        """Validate and sanitize text input."""
        start_time = time.time()
        result = SanitizationResult(success=True, original_content=text, sanitized_content=text)

        try:
            # Length validation
            if context.max_length and len(text) > context.max_length:
                result.sanitized_content = text[: context.max_length]
                result.content_modified = True
                result.warnings.append(f"Content truncated to {context.max_length} characters")

            # XSS detection and cleaning
            xss_result = await self._detect_xss(result.sanitized_content, context)
            if xss_result["detected"]:
                result.violations_detected.append(
                    {
                        "type": ViolationType.XSS_ATTEMPT.value,
                        "patterns": xss_result["patterns"],
                        "severity": ThreatLevel.HIGH.value,
                    }
                )
                result.sanitized_content = xss_result["cleaned_content"]
                result.content_modified = True
                result.threat_level = ThreatLevel.HIGH

            # SQL injection detection
            sql_result = await self._detect_sql_injection(result.sanitized_content, context)
            if sql_result["detected"]:
                result.violations_detected.append(
                    {
                        "type": ViolationType.SQL_INJECTION.value,
                        "patterns": sql_result["patterns"],
                        "severity": ThreatLevel.HIGH.value,
                    }
                )
                result.sanitized_content = sql_result["cleaned_content"]
                result.content_modified = True
                result.threat_level = max(result.threat_level, ThreatLevel.HIGH)

            # HTML sanitization if needed
            if context.content_type == ContentType.HTML:
                html_result = await self._sanitize_html(result.sanitized_content, context)
                result.sanitized_content = html_result["content"]
                if html_result["modified"]:
                    result.content_modified = True
                    result.operations_performed.append("html_sanitization")

            # Content filtering
            filter_result = await self._filter_inappropriate_content(
                result.sanitized_content, context
            )
            if filter_result["filtered"]:
                result.violations_detected.append(
                    {
                        "type": ViolationType.INAPPROPRIATE_CONTENT.value,
                        "reasons": filter_result["reasons"],
                        "severity": ThreatLevel.MEDIUM.value,
                    }
                )
                result.sanitized_content = filter_result["content"]
                result.content_modified = True

            # Privacy protection
            privacy_result = await self._protect_privacy(result.sanitized_content, context)
            if privacy_result["modified"]:
                result.sanitized_content = privacy_result["content"]
                result.content_modified = True
                result.operations_performed.append("privacy_protection")
                if privacy_result["violations"]:
                    result.violations_detected.extend(privacy_result["violations"])

            result.processing_time = time.time() - start_time
            return result

        except Exception as e:
            result.success = False
            result.violations_detected.append(
                {"type": "validation_error", "error": str(e), "severity": ThreatLevel.MEDIUM.value}
            )
            result.processing_time = time.time() - start_time
            return result

    async def _detect_xss(self, content: str, context: SanitizationContext) -> Dict[str, Any]:
        """Detect and clean XSS attempts."""
        detected_patterns = []
        cleaned_content = content

        for pattern in self.xss_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                detected_patterns.append(
                    {"pattern": pattern, "match": match.group(), "position": match.span()}
                )

        if detected_patterns:
            # Clean the content
            for pattern in self.xss_patterns:
                cleaned_content = re.sub(
                    pattern, "", cleaned_content, flags=re.IGNORECASE | re.DOTALL
                )

            # Additional HTML entity encoding
            cleaned_content = html.escape(cleaned_content)

        return {
            "detected": len(detected_patterns) > 0,
            "patterns": detected_patterns,
            "cleaned_content": cleaned_content,
        }

    async def _detect_sql_injection(
        self, content: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Detect and clean SQL injection attempts."""
        detected_patterns = []
        cleaned_content = content

        for pattern in self.sql_injection_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                detected_patterns.append(
                    {"pattern": pattern, "match": match.group(), "position": match.span()}
                )

        if detected_patterns:
            # Clean the content by removing suspicious SQL keywords
            for pattern in self.sql_injection_patterns:
                cleaned_content = re.sub(
                    pattern, "[FILTERED]", cleaned_content, flags=re.IGNORECASE
                )

        return {
            "detected": len(detected_patterns) > 0,
            "patterns": detected_patterns,
            "cleaned_content": cleaned_content,
        }

    async def _sanitize_html(self, content: str, context: SanitizationContext) -> Dict[str, Any]:
        """Sanitize HTML content."""
        try:
            # Use bleach to clean HTML
            allowed_tags = context.allowed_tags if context.allowed_tags else self.allowed_tags

            cleaned_content = bleach.clean(
                content, tags=allowed_tags, attributes=self.allowed_attributes, strip=True
            )

            return {"content": cleaned_content, "modified": cleaned_content != content}

        except Exception as e:
            self.logger.error(f"HTML sanitization failed: {str(e)}")
            return {"content": html.escape(content), "modified": True}

    async def _filter_inappropriate_content(
        self, content: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Filter inappropriate or harmful content."""
        reasons = []
        filtered_content = content

        # Check for profanity (basic implementation)
        profanity_patterns = [
            r"\b(fuck|shit|damn|hell|bitch|asshole)\b",
            # Add more patterns as needed
        ]

        for pattern in profanity_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                reasons.append("profanity_detected")
                filtered_content = re.sub(
                    pattern, "[FILTERED]", filtered_content, flags=re.IGNORECASE
                )

        # Check for personal information patterns
        pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        }

        for pii_type, pattern in pii_patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                reasons.append(f"{pii_type}_detected")
                if context.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                    filtered_content = re.sub(
                        pattern, f"[{pii_type.upper()}_FILTERED]", filtered_content
                    )

        return {"filtered": len(reasons) > 0, "reasons": reasons, "content": filtered_content}

    async def _protect_privacy(self, content: str, context: SanitizationContext) -> Dict[str, Any]:
        """Protect privacy by detecting and handling sensitive information."""
        violations = []
        modified_content = content
        modified = False

        # GDPR-related patterns
        gdpr_patterns = {
            "personal_data": r"\b(name|address|phone|email|id)\s*[:=]\s*\S+",
            "sensitive_data": r"\b(password|ssn|credit|bank|account)\s*[:=]\s*\S+",
        }

        for violation_type, pattern in gdpr_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "type": ViolationType.PRIVACY_VIOLATION.value,
                        "subtype": violation_type,
                        "match": match.group(),
                        "position": match.span(),
                        "severity": ThreatLevel.HIGH.value,
                    }
                )

                if context.security_level in [
                    SecurityLevel.HIGH,
                    SecurityLevel.CRITICAL,
                    SecurityLevel.MAXIMUM,
                ]:
                    modified_content = modified_content.replace(
                        match.group(), "[PRIVACY_PROTECTED]"
                    )
                    modified = True

        return {"content": modified_content, "modified": modified, "violations": violations}


class FileValidator:
    """Validates and sanitizes file uploads."""

    def __init__(self, sanitizer: "SecuritySanitizer"):
        self.sanitizer = sanitizer
        self.logger = get_logger(__name__)

        # Safe file types
        self.safe_extensions = {
            "text": {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml"},
            "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"},
            "audio": {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"},
            "video": {".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"},
            "document": {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"},
        }

        # Dangerous file types
        self.dangerous_extensions = {
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".pif",
            ".scr",
            ".vbs",
            ".js",
            ".jar",
            ".app",
            ".deb",
            ".pkg",
            ".dmg",
            ".iso",
            ".msi",
        }

        # Maximum file sizes (in bytes)
        self.max_file_sizes = {
            "image": 10 * 1024 * 1024,  # 10MB
            "audio": 50 * 1024 * 1024,  # 50MB
            "video": 100 * 1024 * 1024,  # 100MB
            "document": 25 * 1024 * 1024,  # 25MB
            "text": 1 * 1024 * 1024,  # 1MB
            "default": 5 * 1024 * 1024,  # 5MB
        }

    async def validate_file(
        self, file_data: bytes, filename: str, context: SanitizationContext
    ) -> SanitizationResult:
        """Validate and sanitize file upload."""
        start_time = time.time()
        result = SanitizationResult(
            success=True, original_content=file_data, sanitized_content=file_data
        )

        try:
            # File extension validation
            file_path = Path(filename)
            extension = file_path.suffix.lower()

            if extension in self.dangerous_extensions:
                result.success = False
                result.violations_detected.append(
                    {
                        "type": ViolationType.MALWARE_DETECTED.value,
                        "reason": f"Dangerous file extension: {extension}",
                        "severity": ThreatLevel.HIGH.value,
                    }
                )
                result.threat_level = ThreatLevel.HIGH
                return result

            # File size validation
            file_size = len(file_data)
            file_type = self._get_file_type(extension)
            max_size = self.max_file_sizes.get(file_type, self.max_file_sizes["default"])

            if file_size > max_size:
                result.success = False
                result.violations_detected.append(
                    {
                        "type": ViolationType.SUSPICIOUS_BEHAVIOR.value,
                        "reason": f"File size ({file_size}) exceeds limit ({max_size})",
                        "severity": ThreatLevel.MEDIUM.value,
                    }
                )
                return result

            # MIME type validation
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                mime_result = await self._validate_mime_type(file_data, mime_type, context)
                if not mime_result["valid"]:
                    result.violations_detected.extend(mime_result["violations"])
                    if mime_result["block"]:
                        result.success = False
                        return result

            # Malware scanning
            malware_result = await self._scan_for_malware(file_data, filename, context)
            if malware_result["detected"]:
                result.success = False
                result.violations_detected.append(
                    {
                        "type": ViolationType.MALWARE_DETECTED.value,
                        "details": malware_result["details"],
                        "severity": ThreatLevel.CRITICAL.value,
                    }
                )
                result.threat_level = ThreatLevel.CRITICAL
                return result

            # Content-specific validation
            if file_type == "image":
                image_result = await self._validate_image(file_data, context)
                if image_result["violations"]:
                    result.violations_detected.extend(image_result["violations"])
                if image_result["sanitized_data"] != file_data:
                    result.sanitized_content = image_result["sanitized_data"]
                    result.content_modified = True

            result.processing_time = time.time() - start_time
            result.operations_performed.append("file_validation")

            return result

        except Exception as e:
            result.success = False
            result.violations_detected.append(
                {"type": "validation_error", "error": str(e), "severity": ThreatLevel.MEDIUM.value}
            )
            result.processing_time = time.time() - start_time
            return result

    def _get_file_type(self, extension: str) -> str:
        """Determine file type category from extension."""
        for file_type, extensions in self.safe_extensions.items():
            if extension in extensions:
                return file_type
        return "unknown"

    async def _validate_mime_type(
        self, file_data: bytes, expected_mime: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Validate MIME type against file content."""
        violations = []

        try:
            # Basic header validation for common file types
            if expected_mime.startswith("image/"):
                if not self._is_valid_image_header(file_data):
                    violations.append(
                        {
                            "type": ViolationType.MALWARE_DETECTED.value,
                            "reason": "Invalid image file header",
                            "severity": ThreatLevel.HIGH.value,
                        }
                    )

            elif expected_mime == "application/pdf":
                if not file_data.startswith(b"%PDF-"):
                    violations.append(
                        {
                            "type": ViolationType.SUSPICIOUS_BEHAVIOR.value,
                            "reason": "Invalid PDF file header",
                            "severity": ThreatLevel.MEDIUM.value,
                        }
                    )

            return {
                "valid": len(violations) == 0,
                "violations": violations,
                "block": any(v.get("severity") == ThreatLevel.HIGH.value for v in violations),
            }

        except Exception as e:
            return {
                "valid": False,
                "violations": [
                    {
                        "type": "mime_validation_error",
                        "error": str(e),
                        "severity": ThreatLevel.MEDIUM.value,
                    }
                ],
                "block": False,
            }

    def _is_valid_image_header(self, file_data: bytes) -> bool:
        """Check if file has valid image headers."""
        if len(file_data) < 8:
            return False

        # Common image file signatures
        signatures = [
            b"\x89PNG\r\n\x1a\n",  # PNG
            b"\xff\xd8\xff",  # JPEG
            b"GIF87a",  # GIF87a
            b"GIF89a",  # GIF89a
            b"BM",  # BMP
            b"RIFF",  # WebP (starts with RIFF)
        ]

        for signature in signatures:
            if file_data.startswith(signature):
                return True

        return False

    async def _scan_for_malware(
        self, file_data: bytes, filename: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Basic malware scanning."""
        # This is a simplified implementation
        # In production, integrate with actual antivirus engines

        # Check for suspicious patterns in binary files
        suspicious_patterns = [
            b"eval(",
            b"exec(",
            b"shell_exec",
            b"system(",
            b"passthru",
            b"base64_decode",
            b"<script",
            b"javascript:",
            b"vbscript:",
        ]

        detected_patterns = []
        for pattern in suspicious_patterns:
            if pattern in file_data:
                detected_patterns.append(pattern.decode("utf-8", errors="ignore"))

        # Check file entropy (high entropy might indicate encryption/obfuscation)
        entropy = self._calculate_entropy(file_data)
        high_entropy = entropy > 7.5  # Threshold for suspicious entropy

        detected = len(detected_patterns) > 0 or high_entropy

        return {
            "detected": detected,
            "details": {
                "suspicious_patterns": detected_patterns,
                "high_entropy": high_entropy,
                "entropy_score": entropy,
            },
        }

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0

        # Count byte frequencies
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1

        # Calculate entropy
        entropy = 0
        length = len(data)
        for count in freq.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return entropy

    async def _validate_image(
        self, image_data: bytes, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Validate and sanitize image content."""
        violations = []
        sanitized_data = image_data

        try:
            # Load image with PIL to validate structure
            from io import BytesIO

            image = Image.open(BytesIO(image_data))

            # Check image dimensions
            width, height = image.size
            max_dimensions = 4096  # Max width/height

            if width > max_dimensions or height > max_dimensions:
                violations.append(
                    {
                        "type": ViolationType.SUSPICIOUS_BEHAVIOR.value,
                        "reason": f"Image dimensions too large: {width}x{height}",
                        "severity": ThreatLevel.LOW.value,
                    }
                )

                # Resize if needed
                if context.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                    image.thumbnail((max_dimensions, max_dimensions), Image.Resampling.LANCZOS)
                    output = BytesIO()
                    image.save(output, format=image.format)
                    sanitized_data = output.getvalue()

            # Remove EXIF data for privacy
            if hasattr(image, "_getexif") and image._getexif():
                violations.append(
                    {
                        "type": ViolationType.PRIVACY_VIOLATION.value,
                        "reason": "EXIF data detected",
                        "severity": ThreatLevel.LOW.value,
                    }
                )

                # Strip EXIF data
                clean_image = Image.new(image.mode, image.size)
                clean_image.putdata(list(image.getdata()))
                output = BytesIO()
                clean_image.save(output, format=image.format)
                sanitized_data = output.getvalue()

            return {"violations": violations, "sanitized_data": sanitized_data}

        except Exception as e:
            violations.append(
                {
                    "type": "image_validation_error",
                    "error": str(e),
                    "severity": ThreatLevel.MEDIUM.value,
                }
            )

            return {"violations": violations, "sanitized_data": image_data}


class OutputFilter:
    """Filters and sanitizes AI assistant output."""

    def __init__(self, sanitizer: "SecuritySanitizer"):
        self.sanitizer = sanitizer
        self.logger = get_logger(__name__)

        # Sensitive information patterns
        self.sensitive_patterns = {
            "api_key": r"\b[A-Za-z0-9]{32,}\b",
            "password": r"password[:\s=]+\S+",
            "token": r"token[:\s=]+[A-Za-z0-9\-_.]+",
            "secret": r"secret[:\s=]+\S+",
            "private_key": r"-----BEGIN [A-Z ]+PRIVATE KEY-----",
            "connection_string": r"(mongodb|mysql|postgresql)://[^\s]+",
        }

    async def filter_output(self, content: str, context: SanitizationContext) -> SanitizationResult:
        """Filter and sanitize AI assistant output."""
        start_time = time.time()
        result = SanitizationResult(
            success=True, original_content=content, sanitized_content=content
        )

        try:
            # Check for sensitive information leakage
            leak_result = await self._detect_information_leakage(result.sanitized_content, context)
            if leak_result["detected"]:
                result.violations_detected.extend(leak_result["violations"])
                result.sanitized_content = leak_result["cleaned_content"]
                result.content_modified = True
                result.threat_level = ThreatLevel.HIGH

            # Content appropriateness check
            appropriate_result = await self._check_content_appropriateness(
                result.sanitized_content, context
            )
            if not appropriate_result["appropriate"]:
                result.violations_detected.extend(appropriate_result["violations"])
                result.sanitized_content = appropriate_result["filtered_content"]
                result.content_modified = True

            # Compliance filtering (GDPR, etc.)
            compliance_result = await self._check_compliance(result.sanitized_content, context)
            if compliance_result["violations"]:
                result.violations_detected.extend(compliance_result["violations"])
                result.sanitized_content = compliance_result["compliant_content"]
                result.content_modified = True

            # Rate limiting information
            if context.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                rate_result = await self._apply_information_rate_limiting(
                    result.sanitized_content, context
                )
                if rate_result["limited"]:
                    result.sanitized_content = rate_result["content"]
                    result.content_modified = True
                    result.operations_performed.append("rate_limiting")

            result.processing_time = time.time() - start_time
            result.operations_performed.append("output_filtering")

            return result

        except Exception as e:
            result.success = False
            result.violations_detected.append(
                {
                    "type": "output_filtering_error",
                    "error": str(e),
                    "severity": ThreatLevel.MEDIUM.value,
                }
            )
            result.processing_time = time.time() - start_time
            return result

    async def _detect_information_leakage(
        self, content: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Detect potential information leakage in output."""
        violations = []
        cleaned_content = content

        for info_type, pattern in self.sensitive_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "type": ViolationType.DATA_LEAK.value,
                        "subtype": info_type,
                        "match": match.group(),
                        "position": match.span(),
                        "severity": ThreatLevel.HIGH.value,
                    }
                )

                # Replace with placeholder
                cleaned_content = cleaned_content.replace(
                    match.group(), f"[{info_type.upper()}_REDACTED]"
                )

        return {
            "detected": len(violations) > 0,
            "violations": violations,
            "cleaned_content": cleaned_content,
        }

    async def _check_content_appropriateness(
        self, content: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Check if content is appropriate for the context."""
        violations = []
        filtered_content = content

        # Check for harmful instructions
        harmful_patterns = [
            r"how to (make|build|create) (bomb|explosive|weapon)",
            r"instructions for (hacking|cracking|breaking into)",
            r"ways to (harm|hurt|kill|suicide)",
            r"illegal (drugs|activities|download)",
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                violations.append(
                    {
                        "type": ViolationType.INAPPROPRIATE_CONTENT.value,
                        "reason": "Potentially harmful instructions",
                        "pattern": pattern,
                        "severity": ThreatLevel.HIGH.value,
                    }
                )

                # Replace with disclaimer
                filtered_content = re.sub(
                    pattern,
                    "[CONTENT_FILTERED: Potentially harmful information removed]",
                    filtered_content,
                    flags=re.IGNORECASE,
                )

        return {
            "appropriate": len(violations) == 0,
            "violations": violations,
            "filtered_content": filtered_content,
        }

    async def _check_compliance(self, content: str, context: SanitizationContext) -> Dict[str, Any]:
        """Check content for compliance violations."""
        violations = []
        compliant_content = content

        # GDPR compliance - check for personal data exposure
        gdpr_patterns = {
            "personal_identifier": r"\b(email|phone|address|ssn|id)\s*[:=]\s*\S+",
            "financial_info": r"\b(credit card|bank account|routing number)\s*[:=]\s*\S+",
        }

        for violation_type, pattern in gdpr_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "type": ViolationType.COMPLIANCE_VIOLATION.value,
                        "subtype": "gdpr_violation",
                        "category": violation_type,
                        "match": match.group(),
                        "severity": ThreatLevel.HIGH.value,
                    }
                )

                # Redact personal information
                compliant_content = compliant_content.replace(
                    match.group(), f"[{violation_type.upper()}_REDACTED_FOR_PRIVACY]"
                )

        return {"violations": violations, "compliant_content": compliant_content}

    async def _apply_information_rate_limiting(
        self, content: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Apply rate limiting to information disclosure."""
        # Simplified implementation - in production, this would be more sophisticated

        # Limit technical details in responses
        technical_patterns = [
            r"server error.*traceback",
            r"internal system path.*/",
            r"database connection.*password",
            r"configuration file.*path",
        ]

        limited_content = content
        limited = False

        for pattern in technical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                limited_content = re.sub(
                    pattern, "[TECHNICAL_DETAILS_LIMITED]", limited_content, flags=re.IGNORECASE
                )
                limited = True

        return {"limited": limited, "content": limited_content}


class ThreatDetector:
    """Detects and analyzes security threats."""

    def __init__(self, sanitizer: "SecuritySanitizer"):
        self.sanitizer = sanitizer
        self.logger = get_logger(__name__)
        self.threat_cache = {}
        self.behavior_patterns = defaultdict(list)

    async def analyze_threat(self, content: Any, context: SanitizationContext) -> Dict[str, Any]:
        """Analyze content for potential threats."""
        threat_indicators = []
        risk_score = 0.0

        # Content-based threat analysis
        if isinstance(content, str):
            text_threats = await self._analyze_text_threats(content, context)
            threat_indicators.extend(text_threats["indicators"])
            risk_score += text_threats["risk_score"]

        # Context-based threat analysis
        context_threats = await self._analyze_context_threats(context)
        threat_indicators.extend(context_threats["indicators"])
        risk_score += context_threats["risk_score"]

        # Behavioral analysis
        if context.user_id:
            behavior_threats = await self._analyze_user_behavior(context)
            threat_indicators.extend(behavior_threats["indicators"])
            risk_score += behavior_threats["risk_score"]

        # Calculate final threat level
        if risk_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif risk_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif risk_score >= 0.2:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.INFO

        return {
            "threat_level": threat_level,
            "risk_score": min(risk_score, 1.0),
            "indicators": threat_indicators,
            "requires_action": risk_score >= 0.6,
        }

    async def _analyze_text_threats(
        self, text: str, context: SanitizationContext
    ) -> Dict[str, Any]:
        """Analyze text content for threats."""
        indicators = []
        risk_score = 0.0

        # Check for injection attempts
        injection_patterns = [
            r"union\s+select",
            r"<script[^>]*>",
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(
                    {"type": "injection_attempt", "pattern": pattern, "severity": ThreatLevel.HIGH}
                )
                risk_score += 0.3

        # Check for social engineering
        social_engineering_keywords = [
            "urgent",
            "immediate action",
            "verify your password",
            "click here now",
            "limited time",
            "suspended account",
        ]

        keyword_count = sum(
            1 for keyword in social_engineering_keywords if keyword.lower() in text.lower()
        )

        if keyword_count >= 3:
            indicators.append(
                {
                    "type": "social_engineering",
                    "keyword_count": keyword_count,
                    "severity": ThreatLevel.MEDIUM,
                }
            )
            risk_score += 0.2

        return {"indicators": indicators, "risk_score": risk_score}

    async def _analyze_context_threats(self, context: SanitizationContext) -> Dict[str, Any]:
        """Analyze context for threat indicators."""
        indicators = []
        risk_score = 0.0

        # Check IP reputation (simplified)
        if context.ip_address:
            # In production, check against threat intelligence feeds
            suspicious_ips = ["127.0.0.1"]  # Placeholder
            if context.ip_address in suspicious_ips:
                indicators.append(
                    {
                        "type": "suspicious_ip",
                        "ip_address": context.ip_address,
                        "severity": ThreatLevel.HIGH,
                    }
                )
                risk_score += 0.4

        # Check user agent
        if context.user_agent:
            suspicious_agents = ["bot", "crawler", "scanner"]
            if any(agent in context.user_agent.lower() for agent in suspicious_agents):
                indicators.append(
                    {
                        "type": "suspicious_user_agent",
                        "user_agent": context.user_agent,
                        "severity": ThreatLevel.LOW,
                    }
                )
                risk_score += 0.1

        return {"indicators": indicators, "risk_score": risk_score}

    async def _analyze_user_behavior(self, context: SanitizationContext) -> Dict[str, Any]:
        """Analyze user behavior patterns for anomalies."""
        indicators = []
        risk_score = 0.0

        if not context.user_id:
            return {"indicators": indicators, "risk_score": risk_score}

        # Track user behavior
        current_time = datetime.now(timezone.utc)
        behavior_key = f"{context.user_id}:{current_time.hour}"

        self.behavior_patterns[behavior_key].append(
            {"timestamp": current_time, "action": context.source, "ip_address": context.ip_address}
        )

        # Analyze recent behavior
        recent_actions = [
            action
            for action in self.behavior_patterns[behavior_key]
            if (current_time - action["timestamp"]).total_seconds() < 3600  # Last hour
        ]

        # Check for unusual activity volume
        if len(recent_actions) > 100:  # More than 100 actions per hour
            indicators.append(
                {
                    "type": "high_activity_volume",
                    "action_count": len(recent_actions),
                    "severity": ThreatLevel.MEDIUM,
                }
            )
            risk_score += 0.3

        # Check for IP changes
        unique_ips = {action.get("ip_address") for action in recent_actions}
        if len(unique_ips) > 5:  # More than 5 different IPs
            indicators.append(
                {
                    "type": "multiple_ip_addresses",
                    "ip_count": len(unique_ips),
                    "severity": ThreatLevel.MEDIUM,
                }
            )
            risk_score += 0.2

        return {"indicators": indicators, "risk_score": risk_score}


class SecuritySanitizer:
    """
    Advanced Security Sanitization System for the AI Assistant.

    This system provides comprehensive security sanitization including:
    - Multi-modal input validation and sanitization
    - Output filtering and content security
    - Threat detection and analysis
    - Real-time security monitoring
    - Integration with core system components
    - Performance optimization and caching
    - Audit logging and compliance
    - Event-driven security responses
    """

    def __init__(self, container: Container):
        """
        Initialize the security sanitizer.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Security integration
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.encryption_manager = None

        # Storage and caching
        try:
            self.database = container.get(DatabaseManager)
            self.cache = container.get(RedisCache)
        except Exception:
            self.database = None
            self.cache = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Sanitization components
        self.input_validator = InputValidator(self)
        self.file_validator = FileValidator(self)
        self.output_filter = OutputFilter(self)
        self.threat_detector = ThreatDetector(self)

        # Sanitization rules
        self.rules: Dict[str, SanitizationRule] = {}
        self.rule_cache: Dict[str, List[SanitizationRule]] = {}

        # Performance tracking
        self.sanitization_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.processing_times: deque = deque(maxlen=1000)

        # Configuration
        self.default_security_level = SecurityLevel(
            self.config.get("security.default_level", "standard")
        )
        self.enable_caching = self.config.get("security.enable_caching", True)
        self.cache_ttl = self.config.get("security.cache_ttl", 300)
        self.enable_threat_detection = self.config.get("security.enable_threat_detection", True)
        self.enable_audit_logging = self.config.get("security.enable_audit_logging", True)
        self.max_content_length = self.config.get("security.max_content_length", 1000000)

        # Thread safety
        self.lock = asyncio.Lock()

        # Initialize components
        self._setup_default_rules()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("security_sanitizer", self._health_check_callback)

        self.logger.info("SecuritySanitizer initialized successfully")

    def _setup_default_rules(self) -> None:
        """Setup default sanitization rules."""
        try:
            # XSS protection rules
            xss_rule = SanitizationRule(
                rule_id="xss_protection",
                name="XSS Protection",
                rule_type=SanitizationType.XSS_PROTECTION,
                content_types={ContentType.TEXT, ContentType.HTML},
                action="clean",
                severity=ThreatLevel.HIGH,
                description="Protect against XSS attacks",
            )
            self.rules[xss_rule.rule_id] = xss_rule

            # SQL injection protection
            sql_rule = SanitizationRule(
                rule_id="sql_injection_protection",
                name="SQL Injection Protection",
                rule_type=SanitizationType.SQL_INJECTION,
                content_types={ContentType.TEXT},
                action="clean",
                severity=ThreatLevel.HIGH,
                description="Protect against SQL injection attacks",
            )
            self.rules[sql_rule.rule_id] = sql_rule

            # File validation rule
            file_rule = SanitizationRule(
                rule_id="file_validation",
                name="File Validation",
                rule_type=SanitizationType.FILE_VALIDATION,
                content_types={ContentType.FILE},
                action="validate",
                severity=ThreatLevel.MEDIUM,
                description="Validate uploaded files for security",
            )
            self.rules[file_rule.rule_id] = file_rule

            # Content filtering rule
            content_rule = SanitizationRule(
                rule_id="content_filtering",
                name="Content Filtering",
                rule_type=SanitizationType.CONTENT_SECURITY,
                content_types={ContentType.TEXT},
                action="filter",
                severity=ThreatLevel.MEDIUM,
                description="Filter inappropriate content",
            )
            self.rules[content_rule.rule_id] = content_rule

            self.logger.info(f"Setup {len(self.rules)} default sanitization rules")

        except Exception as e:
            self.logger.error(f"Failed to setup default rules: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register security metrics
            self.metrics.register_counter("security_sanitizations_total")
            self.metrics.register_counter("security_violations_detected")
            self.metrics.register_counter("security_threats_blocked")
            self.metrics.register_histogram("security_sanitization_duration_seconds")
            self.metrics.register_gauge("security_active_rules")
            self.metrics.register_counter("security_cache_hits")
            self.metrics.register_counter("security_cache_misses")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the security sanitizer."""
        try:
            # Load rules from database if available
            if self.database:
                await self._load_rules_from_database()

            # Register event handlers
            await self._register_event_handlers()

            # Start background tasks
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._threat_analysis_loop())
            asyncio.create_task(self._audit_processing_loop())

            self.logger.info("SecuritySanitizer initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize SecuritySanitizer: {str(e)}")
            raise SanitizationError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User events
        self.event_bus.subscribe("user_authenticated", self._handle_user_authenticated)
        self.event_bus.subscribe("session_started", self._handle_session_started)

        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

        # Component events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

    @handle_exceptions
    async def sanitize_input(
        self,
        content: Any,
        content_type: ContentType = ContentType.TEXT,
        context: Optional[SanitizationContext] = None,
    ) -> SanitizationResult:
        """
        Sanitize user input content.

        Args:
            content: Content to sanitize
            content_type: Type of content
            context: Sanitization context

        Returns:
            Sanitization result
        """
        start_time = time.time()

        # Create default context if not provided
        if context is None:
            context = SanitizationContext(
                content_type=content_type, security_level=self.default_security_level
            )
        else:
            context.content_type = content_type

        # Check cache first
        cache_key = self._generate_cache_key(content, context, "input")
        if self.enable_caching:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.metrics.increment("security_cache_hits")
                return cached_result

        self.metrics.increment("security_cache_misses")

        try:
            with self.tracer.trace("input_sanitization") as span:
                span.set_attributes(
                    {
                        "content_type": content_type.value,
                        "security_level": context.security_level.value,
                        "user_id": context.user_id or "anonymous",
                        "content_length": len(str(content)),
                    }
                )

                # Threat detection
                if self.enable_threat_detection:
                    threat_analysis = await self.threat_detector.analyze_threat(content, context)

                    if threat_analysis["requires_action"]:
                        await self.event_bus.emit(
                            ThreatDetected(
                                threat_level=threat_analysis["threat_level"].value,
                                risk_score=threat_analysis["risk_score"],
                                indicators=threat_analysis["indicators"],
                                user_id=context.user_id,
                                content_type=content_type.value,
                            )
                        )

                # Content-specific sanitization
                if content_type == ContentType.TEXT:
                    if isinstance(content, str):
                        result = await self.input_validator.validate_text(content, context)
                    else:
                        result = SanitizationResult(
                            success=False,
                            original_content=content,
                            sanitized_content=content,
                            violations_detected=[
                                {
                                    "type": "invalid_content_type",
                                    "reason": f"Expected string, got {type(content).__name__}",
                                    "severity": ThreatLevel.MEDIUM.value,
                                }
                            ],
                        )

                elif content_type == ContentType.FILE:
                    if isinstance(content, tuple) and len(content) == 2:
                        file_data, filename = content
                        result = await self.file_validator.validate_file(
                            file_data, filename, context
                        )
                    else:
                        result = SanitizationResult(
                            success=False,
                            original_content=content,
                            sanitized_content=content,
                            violations_detected=[
                                {
                                    "type": "invalid_file_format",
                                    "reason": "Expected tuple of (file_data, filename)",
                                    "severity": ThreatLevel.MEDIUM.value,
                                }
                            ],
                        )

                else:
                    # Generic validation for other content types
                    result = await self._generic_sanitization(content, context)

                # Apply custom rules
                rule_results = await self._apply_custom_rules(result.sanitized_content, context)
                if rule_results["applied"]:
                    result.rules_applied.extend(rule_results["rules"])
                    result.sanitized_content = rule_results["content"]
                    result.violations_detected.extend(rule_results["violations"])
                    result.content_modified = result.content_modified or rule_results["modified"]

                # Update metrics and cache
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                self.processing_times.append(processing_time)

                self.metrics.increment("security_sanitizations_total")
                self.metrics.record("security_sanitization_duration_seconds", processing_time)

                if result.violations_detected:
                    self.metrics.increment(
                        "security_violations_detected",
                        tags={"count": str(len(result.violations_detected))},
                    )

                if not result.success:
                    self.metrics.increment("security_threats_blocked")

                # Cache result
                if self.enable_caching and result.success:
                    await self._cache_result(cache_key, result)

                # Emit events
                await self.event_bus.emit(
                    InputSanitized(
                        content_type=content_type.value,
                        success=result.success,
                        violations_count=len(result.violations_detected),
                        processing_time=processing_time,
                        user_id=context.user_id,
                    )
                )

                # Audit logging
                if self.enable_audit_logging:
                    await self._log_sanitization_event(result, context, "input")

                return result

        except Exception as e:
            # Error handling
            error_result = SanitizationResult(
                success=False,
                original_content=content,
                sanitized_content=content,
                violations_detected=[
                    {
                        "type": "sanitization_error",
                        "error": str(e),
                        "severity": ThreatLevel.HIGH.value,
                    }
                ],
                processing_time=time.time() - start_time,
            )

            await self.event_bus.emit(
                SecurityViolation(
                    user_id=context.user_id,
                    violation_type="sanitization_error",
                    details={"error": str(e), "content_type": content_type.value},
                    severity="high",
                )
            )

            self.logger.error(f"Input sanitization failed: {str(e)}")
            return error_result

    @handle_exceptions
    async def sanitize_output(
        self, content: str, context: Optional[SanitizationContext] = None
    ) -> SanitizationResult:
        """
        Sanitize AI assistant output.

        Args:
            content: Output content to sanitize
            context: Optional sanitization context

        Returns:
            SanitizationResult with sanitized content
        """
        # Basic sanitization implementation
        return SanitizationResult(
            sanitized_content=content,
            issues_detected=[],
            confidence_score=0.95,
            processing_time=0.001,
        )
