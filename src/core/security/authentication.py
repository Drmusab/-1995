"""
Advanced Authentication Management System
Author: Drmusab
Last Modified: 2025-06-13 11:38:53 UTC

This module provides comprehensive authentication for the AI assistant,
including multi-factor authentication, session management, token handling,
social authentication, and seamless integration with all core components.
"""

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from urllib.parse import parse_qs, urlencode

import aiohttp
import asyncio
import bcrypt
import jwt
import pyotp
import qrcode

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EnhancedEventBus
from src.core.events.event_types import (
    AuthenticationConfigChanged,
    AuthenticationTokenGenerated,
    AuthenticationTokenRevoked,
    ComponentHealthChanged,
    ErrorOccurred,
    MultiFactorAuthCompleted,
    MultiFactorAuthRequired,
    PasswordResetCompleted,
    PasswordResetRequested,
    SecurityViolationDetected,
    SystemStateChanged,
    UserAccountLocked,
    UserAccountUnlocked,
    UserAuthenticated,
    UserLoggedOut,
    UserLoginFailed,
    UserRegistered,
    UserSessionEnded,
    UserSessionStarted,
)
from src.core.health_check import HealthCheck
from src.core.security.encryption import EncryptionManager

# Security interfaces
from src.core.security.interfaces import ISessionProvider, IUserProvider
from src.core.security.sanitization import SecuritySanitizer
from src.integrations.cache.redis_cache import RedisCache

# External APIs for social authentication
from src.integrations.external_apis.oauth_providers import OAuthProviderManager

# Storage and caching
from src.integrations.storage.database import DatabaseManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager


class AuthenticationMethod(Enum):
    """Supported authentication methods."""

    PASSWORD = "password"
    MULTI_FACTOR = "multi_factor"
    SOCIAL_OAUTH = "social_oauth"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    SINGLE_SIGN_ON = "single_sign_on"
    TEMPORARY_TOKEN = "temporary_token"


class UserRole(Enum):
    """User roles for authorization."""

    GUEST = "guest"
    USER = "user"
    PREMIUM_USER = "premium_user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    SYSTEM = "system"
    API_CLIENT = "api_client"


class AccountStatus(Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_VERIFICATION = "pending_verification"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    BANNED = "banned"
    DELETED = "deleted"


class TokenType(Enum):
    """Authentication token types."""

    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    API_KEY = "api_key"
    SESSION_TOKEN = "session_token"
    RESET_TOKEN = "reset_token"
    VERIFICATION_TOKEN = "verification_token"
    MFA_TOKEN = "mfa_token"
    TEMPORARY_TOKEN = "temporary_token"


class MFAMethod(Enum):
    """Multi-factor authentication methods."""

    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    BACKUP_CODES = "backup_codes"
    HARDWARE_TOKEN = "hardware_token"
    BIOMETRIC = "biometric"
    PUSH_NOTIFICATION = "push_notification"


@dataclass
class UserProfile:
    """Comprehensive user profile information."""

    user_id: str
    username: str
    email: str

    # Basic information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None

    # Account details
    role: UserRole = UserRole.USER
    status: AccountStatus = AccountStatus.PENDING_VERIFICATION

    # Security settings
    password_hash: Optional[str] = None
    password_salt: Optional[str] = None
    mfa_enabled: bool = False
    mfa_methods: Set[MFAMethod] = field(default_factory=set)
    mfa_backup_codes: List[str] = field(default_factory=list)
    totp_secret: Optional[str] = None

    # Account management
    email_verified: bool = False
    phone_verified: bool = False
    phone_number: Optional[str] = None

    # Security tracking
    last_login: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None

    # Social authentication
    social_accounts: Dict[str, str] = field(default_factory=dict)  # provider -> external_id

    # API access
    api_keys: List[str] = field(default_factory=list)
    api_rate_limit: int = 1000  # requests per hour

    # Preferences and settings
    timezone: str = "UTC"
    language: str = "en"
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: Optional[datetime] = None

    # Privacy and compliance
    privacy_settings: Dict[str, bool] = field(default_factory=dict)
    consent_given: bool = False
    data_retention_days: Optional[int] = None


@dataclass
class AuthenticationToken:
    """Authentication token information."""

    token_id: str
    user_id: str
    token_type: TokenType
    token_value: str

    # Token lifecycle
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None

    # Token metadata
    scope: Set[str] = field(default_factory=set)
    client_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Security
    is_revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_reason: Optional[str] = None

    # Usage tracking
    usage_count: int = 0
    rate_limit: Optional[int] = None


@dataclass
class AuthenticationRequest:
    """Authentication request data."""

    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None

    # Method-specific data
    mfa_code: Optional[str] = None
    mfa_method: Optional[MFAMethod] = None
    social_provider: Optional[str] = None
    social_token: Optional[str] = None
    api_key: Optional[str] = None
    certificate: Optional[str] = None

    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_id: Optional[str] = None
    client_id: Optional[str] = None

    # Additional data
    remember_me: bool = False
    requested_scope: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthenticationResult:
    """Authentication result."""

    success: bool
    user_id: Optional[str] = None

    # Authentication data
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None

    # User information
    user_profile: Optional[UserProfile] = None
    permissions: Set[str] = field(default_factory=set)

    # MFA requirements
    mfa_required: bool = False
    available_mfa_methods: Set[MFAMethod] = field(default_factory=set)
    mfa_token: Optional[str] = None

    # Error information
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Security information
    login_count: int = 0
    last_login: Optional[datetime] = None
    security_warnings: List[str] = field(default_factory=list)

    # Session information
    session_id: Optional[str] = None


class AuthenticationError(Exception):
    """Custom exception for authentication operations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.user_id = user_id
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)


class PasswordValidator:
    """Advanced password validation and strength checking."""

    def __init__(self, config: Dict[str, Any]):
        self.min_length = config.get("min_length", 8)
        self.max_length = config.get("max_length", 128)
        self.require_uppercase = config.get("require_uppercase", True)
        self.require_lowercase = config.get("require_lowercase", True)
        self.require_digits = config.get("require_digits", True)
        self.require_special = config.get("require_special", True)
        self.forbidden_patterns = config.get("forbidden_patterns", [])
        self.min_entropy = config.get("min_entropy", 3.0)

        # Common passwords list (would be loaded from a file)
        self.common_passwords = set()

        self.logger = get_logger(__name__)

    def validate_password(
        self, password: str, user_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Validate password strength and requirements.

        Args:
            password: Password to validate
            user_info: User information for context-based validation

        Returns:
            Validation result with score and feedback
        """
        result = {
            "valid": True,
            "score": 0,
            "max_score": 100,
            "feedback": [],
            "requirements_met": {},
            "entropy": 0.0,
        }

        # Length check
        if len(password) < self.min_length:
            result["valid"] = False
            result["feedback"].append(
                f"Password must be at least {self.min_length} characters long"
            )
        elif len(password) > self.max_length:
            result["valid"] = False
            result["feedback"].append(
                f"Password must be no more than {self.max_length} characters long"
            )
        else:
            result["score"] += 15

        result["requirements_met"]["length"] = self.min_length <= len(password) <= self.max_length

        # Character requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        if self.require_uppercase and not has_upper:
            result["valid"] = False
            result["feedback"].append("Password must contain at least one uppercase letter")
        elif has_upper:
            result["score"] += 15

        if self.require_lowercase and not has_lower:
            result["valid"] = False
            result["feedback"].append("Password must contain at least one lowercase letter")
        elif has_lower:
            result["score"] += 15

        if self.require_digits and not has_digit:
            result["valid"] = False
            result["feedback"].append("Password must contain at least one digit")
        elif has_digit:
            result["score"] += 15

        if self.require_special and not has_special:
            result["valid"] = False
            result["feedback"].append("Password must contain at least one special character")
        elif has_special:
            result["score"] += 15

        result["requirements_met"].update(
            {
                "uppercase": has_upper,
                "lowercase": has_lower,
                "digits": has_digit,
                "special": has_special,
            }
        )

        # Entropy calculation
        entropy = self._calculate_entropy(password)
        result["entropy"] = entropy

        if entropy < self.min_entropy:
            result["valid"] = False
            result["feedback"].append("Password is too predictable")
        else:
            result["score"] += min(25, int(entropy * 5))

        # Common password check
        if password.lower() in self.common_passwords:
            result["valid"] = False
            result["feedback"].append("Password is too common")
        else:
            result["score"] += 10

        # Personal information check
        if user_info and self._contains_personal_info(password, user_info):
            result["valid"] = False
            result["feedback"].append("Password should not contain personal information")
        else:
            result["score"] += 5

        return result

    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy."""
        if not password:
            return 0.0

        # Character set size
        charset_size = 0
        if any(c.islower() for c in password):
            charset_size += 26
        if any(c.isupper() for c in password):
            charset_size += 26
        if any(c.isdigit() for c in password):
            charset_size += 10
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            charset_size += 32

        if charset_size == 0:
            return 0.0

        # Entropy = log2(charset_size^length)
        import math

        entropy = len(password) * math.log2(charset_size)

        # Adjust for patterns and repetition
        unique_chars = len(set(password))
        repetition_factor = unique_chars / len(password)

        return entropy * repetition_factor

    def _contains_personal_info(self, password: str, user_info: Dict[str, str]) -> bool:
        """Check if password contains personal information."""
        password_lower = password.lower()

        for field, value in user_info.items():
            if value and len(value) > 2 and value.lower() in password_lower:
                return True

        return False


class MultiFactorAuthenticator:
    """Multi-factor authentication handler."""

    def __init__(self, config: Dict[str, Any], encryption_manager: EncryptionManager):
        self.config = config
        self.encryption_manager = encryption_manager
        self.logger = get_logger(__name__)

        # TOTP configuration
        self.totp_issuer = config.get("totp_issuer", "AI Assistant")
        self.totp_period = config.get("totp_period", 30)
        self.totp_digits = config.get("totp_digits", 6)

        # Backup codes configuration
        self.backup_codes_count = config.get("backup_codes_count", 10)
        self.backup_code_length = config.get("backup_code_length", 8)

    async def setup_totp(self, user_id: str, username: str) -> Dict[str, Any]:
        """
        Setup TOTP for a user.

        Args:
            user_id: User ID
            username: Username for TOTP label

        Returns:
            TOTP setup information including QR code
        """
        try:
            # Generate secret
            secret = pyotp.random_base32()

            # Create TOTP instance
            totp = pyotp.TOTP(secret, issuer=self.totp_issuer)

            # Generate provisioning URI
            provisioning_uri = totp.provisioning_uri(name=username, issuer_name=self.totp_issuer)

            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)

            # Convert QR code to base64 image
            from io import BytesIO

            img = qr.make_image(fill_color="black", back_color="white")
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            qr_code_base64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                "secret": secret,
                "qr_code": f"data:image/png;base64,{qr_code_base64}",
                "provisioning_uri": provisioning_uri,
                "manual_entry_key": secret,
            }

        except Exception as e:
            self.logger.error(f"Failed to setup TOTP for user {user_id}: {str(e)}")
            raise AuthenticationError(f"TOTP setup failed: {str(e)}", "TOTP_SETUP_FAILED")

    async def verify_totp(self, secret: str, code: str) -> bool:
        """
        Verify TOTP code.

        Args:
            secret: TOTP secret
            code: Code to verify

        Returns:
            True if code is valid
        """
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)  # Allow 1 window tolerance

        except Exception as e:
            self.logger.error(f"TOTP verification failed: {str(e)}")
            return False

    def generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA."""
        codes = []
        for _ in range(self.backup_codes_count):
            code = "".join(secrets.choice("0123456789") for _ in range(self.backup_code_length))
            codes.append(code)
        return codes

    def hash_backup_codes(self, codes: List[str]) -> List[str]:
        """Hash backup codes for secure storage."""
        return [hashlib.sha256(code.encode()).hexdigest() for code in codes]

    def verify_backup_code(self, code: str, hashed_codes: List[str]) -> bool:
        """Verify backup code against hashed codes."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        return code_hash in hashed_codes


class AuthenticationManager:
    """
    Advanced Authentication Management System for the AI Assistant.

    This manager provides comprehensive authentication including:
    - Multiple authentication methods (password, MFA, social, API keys)
    - User account management and profiles
    - Token management and validation
    - Password security and validation
    - Multi-factor authentication support
    - Social authentication integration
    - Session management integration
    - Security monitoring and threat detection
    - Rate limiting and abuse prevention
    - Compliance and audit logging
    """

    def __init__(self, container: Container):
        """
        Initialize the authentication manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EnhancedEventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Security services
        self.encryption_manager = container.get(EncryptionManager)
        self.security_sanitizer = container.get(SecuritySanitizer)

        # Storage and caching
        self.database = container.get(DatabaseManager)
        self.redis_cache = container.get(RedisCache)

        # Session management
        self.session_provider = container.get(ISessionProvider)

        # External integrations
        try:
            self.oauth_providers = container.get(OAuthProviderManager)
        except Exception:
            self.oauth_providers = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Authentication configuration
        self.auth_config = self._load_auth_config()

        # Initialize components
        self.password_validator = PasswordValidator(self.auth_config.get("password_policy", {}))
        self.mfa_authenticator = MultiFactorAuthenticator(
            self.auth_config.get("mfa", {}), self.encryption_manager
        )

        # State management
        self.active_tokens: Dict[str, AuthenticationToken] = {}
        self.user_cache: Dict[str, UserProfile] = {}
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Security settings
        self.max_login_attempts = self.auth_config.get("max_login_attempts", 5)
        self.lockout_duration = self.auth_config.get("lockout_duration", 900)  # 15 minutes
        self.token_cleanup_interval = self.auth_config.get("token_cleanup_interval", 3600)  # 1 hour
        self.enable_rate_limiting = self.auth_config.get("enable_rate_limiting", True)
        self.enable_security_monitoring = self.auth_config.get("enable_security_monitoring", True)

        # JWT configuration
        self.jwt_secret = self.auth_config.get("jwt_secret") or secrets.token_urlsafe(32)
        self.jwt_algorithm = self.auth_config.get("jwt_algorithm", "HS256")
        self.access_token_lifetime = timedelta(hours=self.auth_config.get("access_token_hours", 1))
        self.refresh_token_lifetime = timedelta(days=self.auth_config.get("refresh_token_days", 30))

        # Setup monitoring and background tasks
        self._setup_monitoring()
        self._start_background_tasks()

        # Register health check
        self.health_check.register_component("authentication_manager", self._health_check_callback)

        self.logger.info("AuthenticationManager initialized successfully")

    def _load_auth_config(self) -> Dict[str, Any]:
        """Load authentication configuration."""
        return self.config.get_section(
            "authentication",
            {
                "password_policy": {
                    "min_length": 8,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_digits": True,
                    "require_special": True,
                    "min_entropy": 3.0,
                },
                "mfa": {"totp_issuer": "AI Assistant", "backup_codes_count": 10},
                "tokens": {"access_token_hours": 1, "refresh_token_days": 30},
                "security": {
                    "max_login_attempts": 5,
                    "lockout_duration": 900,
                    "enable_rate_limiting": True,
                },
            },
        )

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register authentication metrics
            self.metrics.register_counter("auth_login_attempts_total")
            self.metrics.register_counter("auth_login_successful")
            self.metrics.register_counter("auth_login_failed")
            self.metrics.register_counter("auth_registrations_total")
            self.metrics.register_counter("auth_tokens_generated")
            self.metrics.register_counter("auth_tokens_revoked")
            self.metrics.register_gauge("auth_active_users")
            self.metrics.register_gauge("auth_active_tokens")
            self.metrics.register_counter("auth_mfa_attempts")
            self.metrics.register_counter("auth_security_violations")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        asyncio.create_task(self._token_cleanup_loop())
        asyncio.create_task(self._security_monitoring_loop())
        asyncio.create_task(self._rate_limit_cleanup_loop())

    async def initialize(self) -> None:
        """Initialize the authentication manager."""
        try:
            # Initialize database tables
            await self._initialize_database()

            # Load active tokens into cache
            await self._load_active_tokens()

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("AuthenticationManager initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize AuthenticationManager: {str(e)}")
            raise AuthenticationError(f"Initialization failed: {str(e)}")

    async def _initialize_database(self) -> None:
        """Initialize database tables for authentication."""
        try:
            # Users table
            await self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT,
                    password_salt TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    display_name TEXT,
                    avatar_url TEXT,
                    role TEXT DEFAULT 'user',
                    status TEXT DEFAULT 'pending_verification',
                    email_verified BOOLEAN DEFAULT FALSE,
                    phone_verified BOOLEAN DEFAULT FALSE,
                    phone_number TEXT,
                    mfa_enabled BOOLEAN DEFAULT FALSE,
                    mfa_methods TEXT,
                    totp_secret TEXT,
                    backup_codes TEXT,
                    last_login TIMESTAMP,
                    last_password_change TIMESTAMP,
                    failed_login_attempts INTEGER DEFAULT 0,
                    account_locked_until TIMESTAMP,
                    social_accounts TEXT,
                    api_keys TEXT,
                    api_rate_limit INTEGER DEFAULT 1000,
                    timezone TEXT DEFAULT 'UTC',
                    language TEXT DEFAULT 'en',
                    preferences TEXT,
                    privacy_settings TEXT,
                    consent_given BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP
                )
            """
            )

            # Authentication tokens table
            await self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_type TEXT NOT NULL,
                    token_value TEXT NOT NULL,
                    scope TEXT,
                    client_id TEXT,
                    device_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    last_used TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    is_revoked BOOLEAN DEFAULT FALSE,
                    revoked_at TIMESTAMP,
                    revoked_reason TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            # Authentication logs table
            await self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_logs (
                    log_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    username TEXT,
                    event_type TEXT NOT NULL,
                    method TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN,
                    error_message TEXT,
                    metadata TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes
            await self.database.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)"
            )
            await self.database.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)"
            )
            await self.database.execute(
                "CREATE INDEX IF NOT EXISTS idx_tokens_user_id ON auth_tokens (user_id)"
            )
            await self.database.execute(
                "CREATE INDEX IF NOT EXISTS idx_tokens_value ON auth_tokens (token_value)"
            )
            await self.database.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON auth_logs (user_id)"
            )
            await self.database.execute(
                "CREATE INDEX IF NOT EXISTS idx_auth_logs_timestamp ON auth_logs (timestamp)"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    async def _load_active_tokens(self) -> None:
        """Load active tokens into cache."""
        try:
            tokens = await self.database.fetch_all(
                """
                SELECT * FROM auth_tokens 
                WHERE is_revoked = FALSE 
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
            )

            for token_data in tokens:
                token = self._deserialize_token(token_data)
                self.active_tokens[token.token_value] = token

            self.logger.info(f"Loaded {len(self.active_tokens)} active tokens")

        except Exception as e:
            self.logger.warning(f"Failed to load active tokens: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    @handle_exceptions
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        Register a new user account.

        Args:
            username: Unique username
            email: User email address
            password: User password
            additional_info: Additional user information

        Returns:
            Created user profile
        """
        # Input validation
        if not username or not email or not password:
            raise AuthenticationError("Username, email, and password are required", "INVALID_INPUT")

        # Sanitize inputs
        username = self.security_sanitizer.sanitize_username(username)
        email = self.security_sanitizer.sanitize_email(email)

        # Validate email format
        if not self._is_valid_email(email):
            raise AuthenticationError("Invalid email format", "INVALID_EMAIL")

        # Validate password
        password_validation = self.password_validator.validate_password(
            password, {"username": username, "email": email}
        )

        if not password_validation["valid"]:
            raise AuthenticationError(
                f"Password validation failed: {'; '.join(password_validation['feedback'])}",
                "WEAK_PASSWORD",
                details=password_validation,
            )

        try:
            with self.tracer.trace("user_registration") as span:
                span.set_attributes({"username": username, "email": email})

                # Check if user already exists
                existing_user = await self._get_user_by_email(email)
                if existing_user:
                    raise AuthenticationError("Email already registered", "EMAIL_EXISTS")

                existing_user = await self._get_user_by_username(username)
                if existing_user:
                    raise AuthenticationError("Username already taken", "USERNAME_EXISTS")

                # Hash password
                password_salt = secrets.token_hex(16)
                password_hash = self._hash_password(password, password_salt)

                # Create user profile
                user_id = str(uuid.uuid4())

                user_profile = UserProfile(
                    user_id=user_id,
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    password_salt=password_salt,
                    status=AccountStatus.PENDING_VERIFICATION,
                )

                # Add additional information
                if additional_info:
                    for key, value in additional_info.items():
                        if hasattr(user_profile, key):
                            setattr(user_profile, key, value)

                # Store user in database
                await self._store_user_profile(user_profile)

                # Generate verification token
                verification_token = await self.generate_verification_token(user_id)

                # Send verification email (would integrate with notification service)
                await self._send_verification_email(email, verification_token)

                # Cache user profile
                self.user_cache[user_id] = user_profile

                # Log registration
                await self._log_auth_event(
                    user_id=user_id, username=username, event_type="user_registered", success=True
                )

                # Emit registration event
                await self.event_bus.emit(
                    UserRegistered(user_id=user_id, username=username, email=email)
                )

                # Update metrics
                self.metrics.increment("auth_registrations_total")

                self.logger.info(f"User registered successfully: {username} ({email})")
                return user_profile

        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"User registration failed: {str(e)}")
            raise AuthenticationError(f"Registration failed: {str(e)}", "REGISTRATION_FAILED")

    @handle_exceptions
    async def authenticate_user(
        self,
        request: AuthenticationRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuthenticationResult:
        """
        Authenticate a user with various methods.

        Args:
            request: Authentication request data
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Authentication result
        """
        start_time = time.time()

        # Rate limiting check
        if self.enable_rate_limiting and ip_address:
            if not self._check_rate_limit(ip_address):
                raise AuthenticationError("Too many login attempts", "RATE_LIMITED")

        # Determine authentication method
        auth_method = self._determine_auth_method(request)

        try:
            with self.tracer.trace("user_authentication") as span:
                span.set_attributes(
                    {
                        "auth_method": auth_method.value,
                        "username": request.username or "",
                        "email": request.email or "",
                        "ip_address": ip_address or "",
                    }
                )

                # Update metrics
                self.metrics.increment("auth_login_attempts_total")

                # Perform authentication based on method
                if auth_method == AuthenticationMethod.PASSWORD:
                    result = await self._authenticate_password(request, ip_address, user_agent)
                elif auth_method == AuthenticationMethod.MULTI_FACTOR:
                    result = await self._authenticate_mfa(request, ip_address, user_agent)
                elif auth_method == AuthenticationMethod.SOCIAL_OAUTH:
                    result = await self._authenticate_social(request, ip_address, user_agent)
                elif auth_method == AuthenticationMethod.API_KEY:
                    result = await self._authenticate_api_key(request, ip_address, user_agent)
                else:
                    raise AuthenticationError(
                        f"Unsupported authentication method: {auth_method}", "UNSUPPORTED_METHOD"
                    )

                # Handle successful authentication
                if result.success:
                    # Update user login information
                    if result.user_id:
                        await self._update_user_login(result.user_id, ip_address)

                    # Generate tokens if not MFA pending
                    if not result.mfa_required:
                        result.access_token, result.refresh_token = (
                            await self._generate_user_tokens(
                                result.user_id,
                                request.client_id,
                                request.device_id,
                                ip_address,
                                user_agent,
                            )
                        )

                    # Create session if needed
                    if result.user_id and not result.mfa_required:
                        try:
                            session_id = await self.session_provider.create_session(
                                result.user_id,
                                session_config={
                                    "device_info": {"user_agent": user_agent},
                                    "network_info": {"ip_address": ip_address},
                                },
                            )
                            result.session_id = session_id
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to create session for user {result.user_id}: {e}"
                            )
                            # Continue without session - not critical for authentication

                    # Log successful authentication
                    await self._log_auth_event(
                        user_id=result.user_id,
                        username=request.username or request.email,
                        event_type="login_successful",
                        method=auth_method.value,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=True,
                    )

                    # Emit authentication event
                    await self.event_bus.emit(
                        UserAuthenticated(
                            user_id=result.user_id,
                            username=result.user_profile.username if result.user_profile else "",
                            method=auth_method.value,
                            ip_address=ip_address,
                            session_id=result.session_id,
                        )
                    )

                    self.metrics.increment("auth_login_successful")

                else:
                    # Handle failed authentication
                    await self._handle_authentication_failure(
                        request, auth_method, result, ip_address, user_agent
                    )

                authentication_time = time.time() - start_time
                self.logger.info(
                    f"Authentication completed for {request.username or request.email} "
                    f"in {authentication_time:.2f}s: {'SUCCESS' if result.success else 'FAILED'}"
                )

                return result

        except AuthenticationError:
            raise
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}", "AUTHENTICATION_ERROR")

    def _determine_auth_method(self, request: AuthenticationRequest) -> AuthenticationMethod:
        """Determine authentication method from request."""
        if request.mfa_code:
            return AuthenticationMethod.MULTI_FACTOR
        elif request.social_provider and request.social_token:
            return AuthenticationMethod.SOCIAL_OAUTH
        elif request.api_key:
            return AuthenticationMethod.API_KEY
        elif request.password:
            return AuthenticationMethod.PASSWORD
        else:
            raise AuthenticationError("No valid authentication method provided", "INVALID_METHOD")

    async def _authenticate_password(
        self, request: AuthenticationRequest, ip_address: Optional[str], user_agent: Optional[str]
    ) -> AuthenticationResult:
        """Authenticate using username/password."""
        # Get user
        user = None
        if request.email:
            user = await self._get_user_by_email(request.email)
        elif request.username:
            user = await self._get_user_by_username(request.username)

        if not user:
            return AuthenticationResult(
                success=False, error_code="USER_NOT_FOUND", error_message="Invalid credentials"
            )

        # Check account status
        if user.status == AccountStatus.LOCKED:
            if user.account_locked_until and datetime.now(timezone.utc) < user.account_locked_until:
                return AuthenticationResult(
                    success=False,
                    error_code="ACCOUNT_LOCKED",
                    error_message="Account is temporarily locked",
                )
            else:
                # Unlock account
                await self._unlock_user_account(user.user_id)
                user.status = AccountStatus.ACTIVE

        if user.status not in [AccountStatus.ACTIVE, AccountStatus.PENDING_VERIFICATION]:
            return AuthenticationResult(
                success=False, error_code="ACCOUNT_DISABLED", error_message="Account is disabled"
            )

        # Verify password
        if not self._verify_password(request.password, user.password_hash, user.password_salt):
            # Increment failed attempts
            await self._increment_failed_attempts(user.user_id)

            return AuthenticationResult(
                success=False, error_code="INVALID_CREDENTIALS", error_message="Invalid credentials"
            )

        # Reset failed attempts on successful password verification
        await self._reset_failed_attempts(user.user_id)

        # Check if MFA is required
        if user.mfa_enabled and user.mfa_methods:
            mfa_token = await self._generate_mfa_token(user.user_id)

            return AuthenticationResult(
                success=False,  # Not fully authenticated yet
                user_id=user.user_id,
                user_profile=user,
                mfa_required=True,
                available_mfa_methods=user.mfa_methods,
                mfa_token=mfa_token,
            )

        # Successful authentication
        return AuthenticationResult(
            success=True,
            user_id=user.user_id,
            user_profile=user,
            permissions=self._get_user_permissions(user),
            login_count=getattr(user, "login_count", 0) + 1,
            last_login=user.last_login,
        )

    async def _authenticate_mfa(
        self, request: AuthenticationRequest, ip_address: Optional[str], user_agent: Optional[str]
    ) -> AuthenticationResult:
        """Authenticate using multi-factor authentication."""
        if not request.mfa_token or not request.mfa_code:
            return AuthenticationResult(
                success=False,
                error_code="INVALID_MFA_REQUEST",
                error_message="MFA token and code are required",
            )

        # Validate MFA token
        mfa_data = await self._validate_mfa_token(request.mfa_token)
        if not mfa_data:
            return AuthenticationResult(
                success=False,
                error_code="INVALID_MFA_TOKEN",
                error_message="Invalid or expired MFA token",
            )

        user = await self._get_user_by_id(mfa_data["user_id"])
        if not user:
            return AuthenticationResult(
                success=False, error_code="USER_NOT_FOUND", error_message="User not found"
            )

        # Verify MFA code based on method
        mfa_method = request.mfa_method or MFAMethod.TOTP

        if mfa_method == MFAMethod.TOTP:
            if not user.totp_secret:
                return AuthenticationResult(
                    success=False,
                    error_code="TOTP_NOT_SETUP",
                    error_message="TOTP is not set up for this user",
                )

            if not await self.mfa_authenticator.verify_totp(user.totp_secret, request.mfa_code):
                self.metrics.increment("auth_mfa_attempts")
                return AuthenticationResult(
                    success=False, error_code="INVALID_MFA_CODE", error_message="Invalid MFA code"
                )

        elif mfa_method == MFAMethod.BACKUP_CODES:
            if not user.mfa_backup_codes:
                return AuthenticationResult(
                    success=False,
                    error_code="BACKUP_CODES_NOT_SETUP",
                    error_message="Backup codes are not set up",
                )

            if not self.mfa_authenticator.verify_backup_code(
                request.mfa_code, user.mfa_backup_codes
            ):
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_BACKUP_CODE",
                    error_message="Invalid backup code",
                )

            # Remove used backup code
            await self._remove_used_backup_code(user.user_id, request.mfa_code)

        # Revoke MFA token
        await self._revoke_mfa_token(request.mfa_token)

        # Emit MFA completion event
        await self.event_bus.emit(
            MultiFactorAuthCompleted(user_id=user.user_id, method=mfa_method.value)
        )

        # Successful MFA authentication
        return AuthenticationResult(
            success=True,
            user_id=user.user_id,
            user_profile=user,
            permissions=self._get_user_permissions(user),
            login_count=getattr(user, "login_count", 0) + 1,
            last_login=user.last_login,
        )

    async def _authenticate_social(
        self, request: AuthenticationRequest, ip_address: Optional[str], user_agent: Optional[str]
    ) -> AuthenticationResult:
        """Authenticate using social OAuth."""
        if not self.oauth_providers:
            return AuthenticationResult(
                success=False,
                error_code="SOCIAL_AUTH_DISABLED",
                error_message="Social authentication is not available",
            )

        try:
            # Validate social token
            user_info = await self.oauth_providers.validate_token(
                request.social_provider, request.social_token
            )

            if not user_info:
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_SOCIAL_TOKEN",
                    error_message="Invalid social authentication token",
                )

            # Find or create user
            user = await self._find_or_create_social_user(request.social_provider, user_info)

            return AuthenticationResult(
                success=True,
                user_id=user.user_id,
                user_profile=user,
                permissions=self._get_user_permissions(user),
                login_count=getattr(user, "login_count", 0) + 1,
                last_login=user.last_login,
            )

        except Exception as e:
            self.logger.error(f"Social authentication failed: {str(e)}")
            return AuthenticationResult(
                success=False,
                error_code="SOCIAL_AUTH_ERROR",
                error_message="Social authentication failed",
            )

    async def _authenticate_api_key(
        self, request: AuthenticationRequest, ip_address: Optional[str], user_agent: Optional[str]
    ) -> AuthenticationResult:
        """Authenticate using API key."""
        api_key_hash = hashlib.sha256(request.api_key.encode()).hexdigest()

        # Find user by API key
        user = await self._get_user_by_api_key(api_key_hash)
        if not user:
            return AuthenticationResult(
                success=False, error_code="INVALID_API_KEY", error_message="Invalid API key"
            )

        # Check rate limiting for API key
        if not self._check_api_rate_limit(user.user_id):
            return AuthenticationResult(
                success=False,
                error_code="API_RATE_LIMIT_EXCEEDED",
                error_message="API rate limit exceeded",
            )

        return AuthenticationResult(
            success=True,
            user_id=user.user_id,
            user_profile=user,
            permissions=self._get_user_permissions(user),
            login_count=getattr(user, "login_count", 0) + 1,
        )

    async def _handle_authentication_failure(
        self,
        request: AuthenticationRequest,
        auth_method: AuthenticationMethod,
        result: AuthenticationResult,
        ip_address: Optional[str],
        user_agent: Optional[str],
    ) -> None:
        """Handle authentication failure."""
        # Log failed authentication
        await self._log_auth_event(
            username=request.username or request.email,
            event_type="login_failed",
            method=auth_method.value,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            error_message=result.error_message,
        )

        # Emit failure event
        await self.event_bus.emit(
            UserLoginFailed(
                username=request.username or request.email or "",
                method=auth_method.value,
                error_code=result.error_code,
                ip_address=ip_address,
            )
        )

        # Update metrics
        self.metrics.increment("auth_login_failed")

        # Track rate limiting
        if ip_address:
            self._record_failed_attempt(ip_address)

    @handle_exceptions
    async def validate_token(self, token: str) -> Optional[AuthenticationToken]:
        """
        Validate an authentication token.

        Args:
            token: Token to validate

        Returns:
            Token information if valid, None otherwise
        """
        try:
            # Check cache first
            if token in self.active_tokens:
                token_obj = self.active_tokens[token]

                # Check expiration
                if token_obj.expires_at and datetime.now(timezone.utc) > token_obj.expires_at:
                    await self._revoke_token(token, "expired")
                    return None

                # Update last used
                token_obj.last_used = datetime.now(timezone.utc)
                token_obj.usage_count += 1

                return token_obj

            # Check database
            token_data = await self.database.fetch_one(
                "SELECT * FROM auth_tokens WHERE token_value = ? AND is_revoked = FALSE", (token,)
            )

            if not token_data:
                return None

            token_obj = self._deserialize_token(token_data)

            # Check expiration
            if token_obj.expires_at and datetime.now(timezone.utc) > token_obj.expires_at:
                await self._revoke_token(token, "expired")
                return None

            # Add to cache
            self.active_tokens[token] = token_obj

            # Update last used
            token_obj.last_used = datetime.now(timezone.utc)
            token_obj.usage_count += 1

            return token_obj

        except Exception as e:
            self.logger.error(f"Token validation failed: {str(e)}")
            return None

    @handle_exceptions
    async def revoke_token(self, token: str, reason: str = "user_requested") -> bool:
        """
        Revoke an authentication token.

        Args:
            token: Token to revoke
            reason: Reason for revocation

        Returns:
            True if token was revoked
        """
        try:
            # Update database
            result = await self.database.execute(
                """
                UPDATE auth_tokens 
                SET is_revoked = TRUE, revoked_at = CURRENT_TIMESTAMP, revoked_reason = ?
                WHERE token_value = ? AND is_revoked = FALSE
                """,
                (reason, token),
            )

            # Remove from cache
            if token in self.active_tokens:
                token_obj = self.active_tokens.pop(token)

                # Emit revocation event
                await self.event_bus.emit(
                    AuthenticationTokenRevoked(
                        token_id=token_obj.token_id,
                        user_id=token_obj.user_id,
                        token_type=token_obj.token_type.value,
                        reason=reason,
                    )
                )

            self.metrics.increment("auth_tokens_revoked")

            return result.rowcount > 0 if result else False

        except Exception as e:
            self.logger.error(f"Token revocation failed: {str(e)}")
            return False

    @handle_exceptions
    async def logout_user(self, user_id: str, token: Optional[str] = None) -> bool:
        """
        Logout a user and invalidate their tokens.

        Args:
            user_id: User ID to logout
            token: Specific token to revoke (optional)

        Returns:
            True if logout was successful
        """
        try:
            if token:
                # Revoke specific token
                await self.revoke_token(token, "logout")
            else:
                # Revoke all user tokens
                await self.database.execute(
                    """
                    UPDATE auth_tokens 
                    SET is_revoked = TRUE, revoked_at = CURRENT_TIMESTAMP, revoked_reason = 'logout'
                    WHERE user_id = ? AND is_revoked = FALSE
                    """,
                    (user_id,),
                )

                # Remove from cache
                tokens_to_remove = [
                    t for t in self.active_tokens.keys() if self.active_tokens[t].user_id == user_id
                ]

                for token_key in tokens_to_remove:
                    self.active_tokens.pop(token_key, None)

            # End user sessions
            try:
                user_sessions = await self.session_provider.list_user_sessions(user_id)
                for session_id in user_sessions:
                    await self.session_provider.invalidate_session(session_id)
            except Exception as e:
                self.logger.warning(f"Failed to invalidate sessions for user {user_id}: {e}")
                # Continue - session cleanup is not critical for logout

            # Log logout
            await self._log_auth_event(user_id=user_id, event_type="logout", success=True)

            # Emit logout event
            await self.event_bus.emit(UserLoggedOut(user_id=user_id))

            return True

        except Exception as e:
            self.logger.error(f"User logout failed: {str(e)}")
            return False

    @handle_exceptions
    async def setup_mfa(self, user_id: str, method: MFAMethod) -> Dict[str, Any]:
        """
        Setup multi-factor authentication for a user.

        Args:
            user_id: User ID
            method: MFA method to setup

        Returns:
            Setup information
        """
        user = await self._get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found", "USER_NOT_FOUND")

        try:
            if method == MFAMethod.TOTP:
                # Setup TOTP
                totp_setup = await self.mfa_authenticator.setup_totp(user_id, user.username)

                # Store encrypted secret
                encrypted_secret = await self.encryption_manager.encrypt(totp_setup["secret"])
                await self.database.execute(
                    "UPDATE users SET totp_secret = ? WHERE user_id = ?",
                    (encrypted_secret, user_id),
                )

                return {
                    "method": method.value,
                    "qr_code": totp_setup["qr_code"],
                    "manual_entry_key": totp_setup["manual_entry_key"],
                    "backup_codes": [],  # Will be generated after verification
                }

            elif method == MFAMethod.BACKUP_CODES:
                # Generate backup codes
                backup_codes = self.mfa_authenticator.generate_backup_codes()
                hashed_codes = self.mfa_authenticator.hash_backup_codes(backup_codes)

                # Store hashed codes
                await self.database.execute(
                    "UPDATE users SET backup_codes = ? WHERE user_id = ?",
                    (json.dumps(hashed_codes), user_id),
                )

                return {"method": method.value, "backup_codes": backup_codes}

            else:
                raise AuthenticationError(
                    f"MFA method {method} not supported", "UNSUPPORTED_MFA_METHOD"
                )

        except Exception as e:
            self.logger.error(f"MFA setup failed for user {user_id}: {str(e)}")
            raise AuthenticationError(f"MFA setup failed: {str(e)}", "MFA_SETUP_FAILED")

    @handle_exceptions
    async def enable_mfa(self, user_id: str, verification_code: str) -> bool:
        """
        Enable MFA for a user after verification.

        Args:
            user_id: User ID
            verification_code: Verification code to confirm setup

        Returns:
            True if MFA was enabled
        """
        user = await self._get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found", "USER_NOT_FOUND")

        try:
            # Verify TOTP code
            if user.totp_secret:
                decrypted_secret = await self.encryption_manager.decrypt(user.totp_secret)
                if not await self.mfa_authenticator.verify_totp(
                    decrypted_secret, verification_code
                ):
                    return False

            # Enable MFA
            mfa_methods = list(user.mfa_methods) if user.mfa_methods else []
            if MFAMethod.TOTP not in mfa_methods:
                mfa_methods.append(MFAMethod.TOTP)

            await self.database.execute(
                "UPDATE users SET mfa_enabled = TRUE, mfa_methods = ? WHERE user_id = ?",
                (json.dumps([m.value for m in mfa_methods]), user_id),
            )

            # Generate backup codes
            backup_codes = self.mfa_authenticator.generate_backup_codes()
            hashed_codes = self.mfa_authenticator.hash_backup_codes(backup_codes)

            await self.database.execute(
                "UPDATE users SET backup_codes = ? WHERE user_id = ?",
                (json.dumps(hashed_codes), user_id),
            )

            # Log MFA enablement
            await self._log_auth_event(user_id=user_id, event_type="mfa_enabled", success=True)

            return True

        except Exception as e:
            self.logger.error(f"MFA enablement failed for user {user_id}: {str(e)}")
            return False

    async def generate_api_key(self, user_id: str, name: Optional[str] = None) -> str:
        """Generate a new API key for a user."""
        user = await self._get_user_by_id(user_id)
        if not user:
            raise AuthenticationError("User not found", "USER_NOT_FOUND")

        # Generate API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Store API key
        current_keys = user.api_keys or []
        current_keys.append(
            {
                "hash": api_key_hash,
                "name": name or f"API Key {len(current_keys) + 1}",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        await self.database.execute(
            "UPDATE users SET api_keys = ? WHERE user_id = ?", (json.dumps(current_keys), user_id)
        )

        return api_key

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information for authenticated user."""
        user = await self._get_user_by_id(user_id)
        if not user:
            return None

        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "display_name": user.display_name,
            "role": user.role.value,
            "status": user.status.value,
            "email_verified": user.email_verified,
            "mfa_enabled": user.mfa_enabled,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "created_at": user.created_at.isoformat(),
        }

    async def is_authenticated(self, user_id: str) -> bool:
        """Check if user is authenticated (has active tokens)."""
        active_tokens = [
            token
            for token in self.active_tokens.values()
            if token.user_id == user_id and not token.is_revoked
        ]
        return len(active_tokens) > 0

    # Helper methods
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format."""
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(email_pattern, email) is not None

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt."""
        return hashlib.pbkdf2_hex(password.encode(), salt.encode(), 100000)

    def _verify_password(self, password: str, hash_value: str, salt: str) -> bool:
        """Verify password against hash."""
        return self._hash_password(password, salt) == hash_value

    async def _health_check_callback(self) -> Dict[str, Any]:
        """Health check callback for the authentication manager."""
        return {
            "status": "healthy",
            "active_tokens": len(self.active_tokens),
            "cache_size": len(self.user_cache),
            "component": "authentication_manager",
        }
