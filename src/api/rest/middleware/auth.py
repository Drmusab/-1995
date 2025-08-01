"""
Unified Authentication and Authorization Service for AI Assistant
Author: Drmusab
Last Modified: 2025-05-26 16:30:00 UTC

This module provides a comprehensive authentication and authorization service
that integrates with all core system components including session management,
component management, workflow orchestration, and plugin management.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

import asyncio
import bcrypt
import jwt
import pyotp
import qrcode

from src.assistant.component_manager import ComponentManager, ComponentPriority
from src.assistant.interaction_handler import InteractionHandler, InteractionPriority
from src.assistant.plugin_manager import PluginManager

# Assistant components
from src.assistant.session_manager import SessionConfiguration, SessionManager, SessionType
from src.assistant.workflow_orchestrator import WorkflowOrchestrator, WorkflowPriority

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    AuthenticationAttempt,
    AuthorizationCheck,
    ComponentHealthChanged,
    ErrorOccurred,
    PasswordChanged,
    PermissionGranted,
    PermissionRevoked,
    RoleAssigned,
    RoleRevoked,
    SecurityAudit,
    SecurityViolation,
    SessionEnded,
    SessionStarted,
    SystemStateChanged,
    TokenGenerated,
    TokenRevoked,
    TwoFactorDisabled,
    TwoFactorEnabled,
    UserAuthenticated,
    UserAuthorized,
    UserLoggedOut,
    UserProfileUpdated,
    UserRegistered,
    UserUnauthorized,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import (
    AuthenticationError,
    AuthenticationManager,
    AuthenticationResult,
    User,
)
from src.core.security.authorization import (
    AuthorizationContext,
    AuthorizationManager,
    AuthorizationResult,
    Permission,
    Role,
)
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.backup_manager import BackupManager
from src.integrations.storage.database import DatabaseManager

# Learning and adaptation
from src.learning.continual_learning import ContinualLearner
from src.learning.feedback_processor import FeedbackProcessor
from src.learning.preference_learning import PreferenceLearner

# Memory and storage
from src.memory.core_memory.memory_manager import MemoryManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Type definitions
T = TypeVar("T")


class AuthenticationState(Enum):
    """Authentication states for users."""

    UNAUTHENTICATED = "unauthenticated"
    AUTHENTICATED = "authenticated"
    MULTI_FACTOR_PENDING = "multi_factor_pending"
    EXPIRED = "expired"
    LOCKED = "locked"
    SUSPENDED = "suspended"
    DISABLED = "disabled"


class AuthorizationLevel(Enum):
    """Authorization levels for different system access."""

    GUEST = "guest"  # Anonymous access
    USER = "user"  # Basic user access
    POWER_USER = "power_user"  # Advanced user features
    ADMIN = "admin"  # Administrative access
    SYSTEM = "system"  # System-level access
    SUPER_ADMIN = "super_admin"  # Full system control


class SessionSecurityLevel(Enum):
    """Security levels for sessions."""

    LOW = "low"  # Basic security
    MEDIUM = "medium"  # Standard security
    HIGH = "high"  # Enhanced security
    CRITICAL = "critical"  # Maximum security


class TokenType(Enum):
    """Types of authentication tokens."""

    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    SESSION_TOKEN = "session_token"
    API_TOKEN = "api_token"
    TEMPORARY_TOKEN = "temporary_token"
    PLUGIN_TOKEN = "plugin_token"


class AuthenticationMethod(Enum):
    """Authentication methods supported."""

    PASSWORD = "password"
    MULTI_FACTOR = "multi_factor"
    BIOMETRIC = "biometric"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SSO = "sso"
    CERTIFICATE = "certificate"


@dataclass
class AuthUser:
    """Extended user information with assistant-specific data."""

    user_id: str
    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None

    # Authentication data
    password_hash: Optional[str] = None
    salt: Optional[str] = None
    totp_secret: Optional[str] = None
    backup_codes: List[str] = field(default_factory=list)

    # Authorization
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    authorization_level: AuthorizationLevel = AuthorizationLevel.USER

    # Session preferences
    max_concurrent_sessions: int = 5
    session_timeout: float = 3600.0  # 1 hour
    preferred_security_level: SessionSecurityLevel = SessionSecurityLevel.MEDIUM

    # Assistant preferences
    preferred_interaction_mode: str = "conversational"
    preferred_language: str = "en"
    accessibility_settings: Dict[str, Any] = field(default_factory=dict)
    privacy_settings: Dict[str, Any] = field(default_factory=dict)

    # Security settings
    require_mfa: bool = False
    allowed_ips: List[str] = field(default_factory=list)
    api_access_enabled: bool = True
    plugin_permissions: Dict[str, Set[str]] = field(default_factory=dict)

    # Audit information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    login_attempts: int = 0
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None

    # State
    is_active: bool = True
    is_verified: bool = False
    is_locked: bool = False
    lock_reason: Optional[str] = None
    lock_until: Optional[datetime] = None


@dataclass
class AuthSession:
    """Authentication session with enhanced security features."""

    session_id: str
    user_id: str
    token: str
    token_type: TokenType = TokenType.SESSION_TOKEN

    # Session metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Security context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    security_level: SessionSecurityLevel = SessionSecurityLevel.MEDIUM

    # Authentication context
    authentication_method: AuthenticationMethod = AuthenticationMethod.PASSWORD
    mfa_verified: bool = False
    certificate_verified: bool = False

    # Permissions and capabilities
    granted_permissions: Set[str] = field(default_factory=set)
    accessible_components: Set[str] = field(default_factory=set)
    allowed_workflows: Set[str] = field(default_factory=set)
    plugin_access: Dict[str, Set[str]] = field(default_factory=dict)

    # Session state
    is_active: bool = True
    is_revoked: bool = False
    revocation_reason: Optional[str] = None


@dataclass
class AuthorizationRequest:
    """Authorization request for system resources."""

    request_id: str
    user_id: str
    session_id: str
    resource_type: str
    resource_id: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AuthError(Exception):
    """Custom exception for authentication and authorization errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = datetime.now(timezone.utc)


class AuthStore(ABC):
    """Abstract interface for authentication data storage."""

    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[AuthUser]:
        """Get user by ID."""
        pass

    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[AuthUser]:
        """Get user by username."""
        pass

    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Get user by email."""
        pass

    @abstractmethod
    async def create_user(self, user: AuthUser) -> None:
        """Create a new user."""
        pass

    @abstractmethod
    async def update_user(self, user: AuthUser) -> None:
        """Update user information."""
        pass

    @abstractmethod
    async def delete_user(self, user_id: str) -> None:
        """Delete a user."""
        pass

    @abstractmethod
    async def create_session(self, session: AuthSession) -> None:
        """Create a new session."""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Get session by ID."""
        pass

    @abstractmethod
    async def update_session(self, session: AuthSession) -> None:
        """Update session information."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        pass

    @abstractmethod
    async def get_user_sessions(self, user_id: str) -> List[AuthSession]:
        """Get all sessions for a user."""
        pass


class DatabaseAuthStore(AuthStore):
    """Database-backed authentication store."""

    def __init__(self, database: DatabaseManager):
        self.database = database
        self.logger = get_logger(__name__)

    async def get_user(self, user_id: str) -> Optional[AuthUser]:
        """Get user by ID from database."""
        try:
            result = await self.database.fetch_one(
                "SELECT * FROM auth_users WHERE user_id = ?", (user_id,)
            )
            return self._row_to_user(result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get user {user_id}: {str(e)}")
            return None

    async def get_user_by_username(self, username: str) -> Optional[AuthUser]:
        """Get user by username from database."""
        try:
            result = await self.database.fetch_one(
                "SELECT * FROM auth_users WHERE username = ?", (username,)
            )
            return self._row_to_user(result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get user by username {username}: {str(e)}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Get user by email from database."""
        try:
            result = await self.database.fetch_one(
                "SELECT * FROM auth_users WHERE email = ?", (email,)
            )
            return self._row_to_user(result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get user by email {email}: {str(e)}")
            return None

    async def create_user(self, user: AuthUser) -> None:
        """Create a new user in database."""
        try:
            await self.database.execute(
                """
                INSERT INTO auth_users (
                    user_id, username, email, display_name, password_hash, salt,
                    totp_secret, backup_codes, roles, permissions, authorization_level,
                    max_concurrent_sessions, session_timeout, preferred_security_level,
                    preferred_interaction_mode, preferred_language, accessibility_settings,
                    privacy_settings, require_mfa, allowed_ips, api_access_enabled,
                    plugin_permissions, created_at, is_active, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user.user_id,
                    user.username,
                    user.email,
                    user.display_name,
                    user.password_hash,
                    user.salt,
                    user.totp_secret,
                    json.dumps(user.backup_codes),
                    json.dumps(list(user.roles)),
                    json.dumps(list(user.permissions)),
                    user.authorization_level.value,
                    user.max_concurrent_sessions,
                    user.session_timeout,
                    user.preferred_security_level.value,
                    user.preferred_interaction_mode,
                    user.preferred_language,
                    json.dumps(user.accessibility_settings),
                    json.dumps(user.privacy_settings),
                    user.require_mfa,
                    json.dumps(user.allowed_ips),
                    user.api_access_enabled,
                    json.dumps({k: list(v) for k, v in user.plugin_permissions.items()}),
                    user.created_at,
                    user.is_active,
                    user.is_verified,
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to create user {user.user_id}: {str(e)}")
            raise AuthError(f"Failed to create user: {str(e)}")

    async def update_user(self, user: AuthUser) -> None:
        """Update user in database."""
        try:
            await self.database.execute(
                """
                UPDATE auth_users SET
                    username = ?, email = ?, display_name = ?, password_hash = ?, salt = ?,
                    totp_secret = ?, backup_codes = ?, roles = ?, permissions = ?,
                    authorization_level = ?, max_concurrent_sessions = ?, session_timeout = ?,
                    preferred_security_level = ?, preferred_interaction_mode = ?,
                    preferred_language = ?, accessibility_settings = ?, privacy_settings = ?,
                    require_mfa = ?, allowed_ips = ?, api_access_enabled = ?,
                    plugin_permissions = ?, last_login = ?, last_password_change = ?,
                    login_attempts = ?, failed_login_attempts = ?, last_failed_login = ?,
                    is_active = ?, is_verified = ?, is_locked = ?, lock_reason = ?,
                    lock_until = ?
                WHERE user_id = ?
                """,
                (
                    user.username,
                    user.email,
                    user.display_name,
                    user.password_hash,
                    user.salt,
                    user.totp_secret,
                    json.dumps(user.backup_codes),
                    json.dumps(list(user.roles)),
                    json.dumps(list(user.permissions)),
                    user.authorization_level.value,
                    user.max_concurrent_sessions,
                    user.session_timeout,
                    user.preferred_security_level.value,
                    user.preferred_interaction_mode,
                    user.preferred_language,
                    json.dumps(user.accessibility_settings),
                    json.dumps(user.privacy_settings),
                    user.require_mfa,
                    json.dumps(user.allowed_ips),
                    user.api_access_enabled,
                    json.dumps({k: list(v) for k, v in user.plugin_permissions.items()}),
                    user.last_login,
                    user.last_password_change,
                    user.login_attempts,
                    user.failed_login_attempts,
                    user.last_failed_login,
                    user.is_active,
                    user.is_verified,
                    user.is_locked,
                    user.lock_reason,
                    user.lock_until,
                    user.user_id,
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to update user {user.user_id}: {str(e)}")
            raise AuthError(f"Failed to update user: {str(e)}")

    async def delete_user(self, user_id: str) -> None:
        """Delete user from database."""
        try:
            await self.database.execute("DELETE FROM auth_users WHERE user_id = ?", (user_id,))
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {str(e)}")
            raise AuthError(f"Failed to delete user: {str(e)}")

    async def create_session(self, session: AuthSession) -> None:
        """Create session in database."""
        try:
            await self.database.execute(
                """
                INSERT INTO auth_sessions (
                    session_id, user_id, token, token_type, created_at, expires_at,
                    last_activity, ip_address, user_agent, device_fingerprint,
                    security_level, authentication_method, mfa_verified,
                    certificate_verified, granted_permissions, accessible_components,
                    allowed_workflows, plugin_access, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.user_id,
                    session.token,
                    session.token_type.value,
                    session.created_at,
                    session.expires_at,
                    session.last_activity,
                    session.ip_address,
                    session.user_agent,
                    session.device_fingerprint,
                    session.security_level.value,
                    session.authentication_method.value,
                    session.mfa_verified,
                    session.certificate_verified,
                    json.dumps(list(session.granted_permissions)),
                    json.dumps(list(session.accessible_components)),
                    json.dumps(list(session.allowed_workflows)),
                    json.dumps({k: list(v) for k, v in session.plugin_access.items()}),
                    session.is_active,
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to create session {session.session_id}: {str(e)}")
            raise AuthError(f"Failed to create session: {str(e)}")

    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Get session from database."""
        try:
            result = await self.database.fetch_one(
                "SELECT * FROM auth_sessions WHERE session_id = ?", (session_id,)
            )
            return self._row_to_session(result) if result else None
        except Exception as e:
            self.logger.error(f"Failed to get session {session_id}: {str(e)}")
            return None

    async def update_session(self, session: AuthSession) -> None:
        """Update session in database."""
        try:
            await self.database.execute(
                """
                UPDATE auth_sessions SET
                    last_activity = ?, granted_permissions = ?, accessible_components = ?,
                    allowed_workflows = ?, plugin_access = ?, is_active = ?,
                    is_revoked = ?, revocation_reason = ?
                WHERE session_id = ?
                """,
                (
                    session.last_activity,
                    json.dumps(list(session.granted_permissions)),
                    json.dumps(list(session.accessible_components)),
                    json.dumps(list(session.allowed_workflows)),
                    json.dumps({k: list(v) for k, v in session.plugin_access.items()}),
                    session.is_active,
                    session.is_revoked,
                    session.revocation_reason,
                    session.session_id,
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to update session {session.session_id}: {str(e)}")
            raise AuthError(f"Failed to update session: {str(e)}")

    async def delete_session(self, session_id: str) -> None:
        """Delete session from database."""
        try:
            await self.database.execute(
                "DELETE FROM auth_sessions WHERE session_id = ?", (session_id,)
            )
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {str(e)}")
            raise AuthError(f"Failed to delete session: {str(e)}")

    async def get_user_sessions(self, user_id: str) -> List[AuthSession]:
        """Get all sessions for a user."""
        try:
            results = await self.database.fetch_all(
                "SELECT * FROM auth_sessions WHERE user_id = ? AND is_active = 1", (user_id,)
            )
            return [self._row_to_session(row) for row in results]
        except Exception as e:
            self.logger.error(f"Failed to get sessions for user {user_id}: {str(e)}")
            return []

    def _row_to_user(self, row) -> AuthUser:
        """Convert database row to AuthUser."""
        return AuthUser(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            display_name=row["display_name"],
            password_hash=row["password_hash"],
            salt=row["salt"],
            totp_secret=row["totp_secret"],
            backup_codes=json.loads(row["backup_codes"] or "[]"),
            roles=set(json.loads(row["roles"] or "[]")),
            permissions=set(json.loads(row["permissions"] or "[]")),
            authorization_level=AuthorizationLevel(row["authorization_level"]),
            max_concurrent_sessions=row["max_concurrent_sessions"],
            session_timeout=row["session_timeout"],
            preferred_security_level=SessionSecurityLevel(row["preferred_security_level"]),
            preferred_interaction_mode=row["preferred_interaction_mode"],
            preferred_language=row["preferred_language"],
            accessibility_settings=json.loads(row["accessibility_settings"] or "{}"),
            privacy_settings=json.loads(row["privacy_settings"] or "{}"),
            require_mfa=bool(row["require_mfa"]),
            allowed_ips=json.loads(row["allowed_ips"] or "[]"),
            api_access_enabled=bool(row["api_access_enabled"]),
            plugin_permissions={
                k: set(v) for k, v in json.loads(row["plugin_permissions"] or "{}").items()
            },
            created_at=datetime.fromisoformat(row["created_at"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
            last_password_change=(
                datetime.fromisoformat(row["last_password_change"])
                if row["last_password_change"]
                else None
            ),
            login_attempts=row["login_attempts"],
            failed_login_attempts=row["failed_login_attempts"],
            last_failed_login=(
                datetime.fromisoformat(row["last_failed_login"])
                if row["last_failed_login"]
                else None
            ),
            is_active=bool(row["is_active"]),
            is_verified=bool(row["is_verified"]),
            is_locked=bool(row["is_locked"]),
            lock_reason=row["lock_reason"],
            lock_until=datetime.fromisoformat(row["lock_until"]) if row["lock_until"] else None,
        )

    def _row_to_session(self, row) -> AuthSession:
        """Convert database row to AuthSession."""
        return AuthSession(
            session_id=row["session_id"],
            user_id=row["user_id"],
            token=row["token"],
            token_type=TokenType(row["token_type"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
            device_fingerprint=row["device_fingerprint"],
            security_level=SessionSecurityLevel(row["security_level"]),
            authentication_method=AuthenticationMethod(row["authentication_method"]),
            mfa_verified=bool(row["mfa_verified"]),
            certificate_verified=bool(row["certificate_verified"]),
            granted_permissions=set(json.loads(row["granted_permissions"] or "[]")),
            accessible_components=set(json.loads(row["accessible_components"] or "[]")),
            allowed_workflows=set(json.loads(row["allowed_workflows"] or "[]")),
            plugin_access={k: set(v) for k, v in json.loads(row["plugin_access"] or "{}").items()},
            is_active=bool(row["is_active"]),
            is_revoked=bool(row["is_revoked"]),
            revocation_reason=row["revocation_reason"],
        )


class EnhancedAuthService:
    """
    Unified Authentication and Authorization Service for the AI Assistant.

    This service provides comprehensive authentication and authorization
    capabilities that integrate seamlessly with all core system components:
    - User authentication with multiple methods (password, MFA, biometric, etc.)
    - Role-based and attribute-based authorization
    - Session management with security levels
    - Integration with session manager, components, workflows, and plugins
    - Fine-grained permission control for AI assistant features
    - Security auditing and monitoring
    - User preference and personalization management
    - API token management for external integrations
    - Multi-tenant support for enterprise deployments
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced authentication service.

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

        # Core authentication and authorization
        self.auth_manager = container.get(AuthenticationManager)
        self.authz_manager = container.get(AuthorizationManager)

        # Assistant components
        self.session_manager = container.get(SessionManager)
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)
        self.interaction_handler = container.get(InteractionHandler)
        self.plugin_manager = container.get(PluginManager)

        # Memory and storage
        self.memory_manager = container.get(MemoryManager)
        self.database = container.get(DatabaseManager)
        self.redis_cache = container.get(RedisCache)

        # Learning systems
        self.continual_learner = container.get(ContinualLearner)
        self.preference_learner = container.get(PreferenceLearner)
        self.feedback_processor = container.get(FeedbackProcessor)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Setup storage
        self.auth_store = DatabaseAuthStore(self.database)

        # State management
        self.active_sessions: Dict[str, AuthSession] = {}
        self.user_cache: Dict[str, AuthUser] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}

        # Security configuration
        self.jwt_secret = self.config.get("auth.jwt_secret", secrets.token_urlsafe(32))
        self.token_expiry = self.config.get("auth.token_expiry", 3600)  # 1 hour
        self.max_failed_attempts = self.config.get("auth.max_failed_attempts", 5)
        self.account_lockout_duration = self.config.get("auth.lockout_duration", 300)  # 5 minutes
        self.require_mfa = self.config.get("auth.require_mfa", False)
        self.password_policy = self.config.get(
            "auth.password_policy",
            {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True,
            },
        )

        # Performance tracking
        self.auth_stats: Dict[str, int] = defaultdict(int)
        self.session_stats: Dict[str, int] = defaultdict(int)

        # Setup monitoring
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("auth_service", self._health_check_callback)

        self.logger.info("EnhancedAuthService initialized successfully")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register authentication metrics
            self.metrics.register_counter("auth_attempts_total")
            self.metrics.register_counter("auth_successes_total")
            self.metrics.register_counter("auth_failures_total")
            self.metrics.register_counter("auth_mfa_attempts_total")
            self.metrics.register_counter("auth_lockouts_total")
            self.metrics.register_gauge("active_sessions")
            self.metrics.register_gauge("active_users")
            self.metrics.register_histogram("auth_duration_seconds")
            self.metrics.register_counter("authorization_checks_total")
            self.metrics.register_counter("authorization_grants_total")
            self.metrics.register_counter("authorization_denials_total")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the authentication service."""
        try:
            # Initialize database tables
            await self._create_database_tables()

            # Load cached data
            await self._load_active_sessions()

            # Start background tasks
            asyncio.create_task(self._session_cleanup_loop())
            asyncio.create_task(self._security_monitoring_loop())
            asyncio.create_task(self._user_preference_sync_loop())

            # Register event handlers
            await self._register_event_handlers()

            self.logger.info("EnhancedAuthService initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize EnhancedAuthService: {str(e)}")
            raise AuthError(f"Initialization failed: {str(e)}")

    async def _create_database_tables(self) -> None:
        """Create necessary database tables."""
        try:
            # Users table
            await self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    display_name TEXT,
                    password_hash TEXT,
                    salt TEXT,
                    totp_secret TEXT,
                    backup_codes TEXT,
                    roles TEXT,
                    permissions TEXT,
                    authorization_level TEXT DEFAULT 'user',
                    max_concurrent_sessions INTEGER DEFAULT 5,
                    session_timeout REAL DEFAULT 3600.0,
                    preferred_security_level TEXT DEFAULT 'medium',
                    preferred_interaction_mode TEXT DEFAULT 'conversational',
                    preferred_language TEXT DEFAULT 'en',
                    accessibility_settings TEXT,
                    privacy_settings TEXT,
                    require_mfa BOOLEAN DEFAULT 0,
                    allowed_ips TEXT,
                    api_access_enabled BOOLEAN DEFAULT 1,
                    plugin_permissions TEXT,
                    created_at TEXT,
                    last_login TEXT,
                    last_password_change TEXT,
                    login_attempts INTEGER DEFAULT 0,
                    failed_login_attempts INTEGER DEFAULT 0,
                    last_failed_login TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    is_locked BOOLEAN DEFAULT 0,
                    lock_reason TEXT,
                    lock_until TEXT
                )
            """
            )

            # Sessions table
            await self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    token_type TEXT DEFAULT 'session_token',
                    created_at TEXT,
                    expires_at TEXT,
                    last_activity TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    device_fingerprint TEXT,
                    security_level TEXT DEFAULT 'medium',
                    authentication_method TEXT DEFAULT 'password',
                    mfa_verified BOOLEAN DEFAULT 0,
                    certificate_verified BOOLEAN DEFAULT 0,
                    granted_permissions TEXT,
                    accessible_components TEXT,
                    allowed_workflows TEXT,
                    plugin_access TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    is_revoked BOOLEAN DEFAULT 0,
                    revocation_reason TEXT,
                    FOREIGN KEY (user_id) REFERENCES auth_users (user_id)
                )
            """
            )

            # Audit log table
            await self.database.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT,
                    success BOOLEAN
                )
            """
            )

        except Exception as e:
            self.logger.error(f"Failed to create database tables: {str(e)}")
            raise

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Component events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

        # User preference events
        self.event_bus.subscribe("user_preference_updated", self._handle_preference_update)

    @handle_exceptions
    async def register_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        display_name: Optional[str] = None,
        authorization_level: AuthorizationLevel = AuthorizationLevel.USER,
        **kwargs,
    ) -> str:
        """
        Register a new user.

        Args:
            username: Unique username
            password: User password
            email: Optional email address
            display_name: Optional display name
            authorization_level: User authorization level
            **kwargs: Additional user properties

        Returns:
            User ID
        """
        # Validate password
        if not self._validate_password(password):
            raise AuthError("Password does not meet policy requirements")

        # Check if username already exists
        existing_user = await self.auth_store.get_user_by_username(username)
        if existing_user:
            raise AuthError(f"Username {username} already exists")

        # Check if email already exists
        if email:
            existing_email_user = await self.auth_store.get_user_by_email(email)
            if existing_email_user:
                raise AuthError(f"Email {email} already registered")

        # Create user
        user_id = str(uuid.uuid4())
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        user = AuthUser(
            user_id=user_id,
            username=username,
            email=email,
            display_name=display_name or username,
            password_hash=password_hash,
            salt=salt,
            authorization_level=authorization_level,
            **kwargs,
        )

        # Store user
        await self.auth_store.create_user(user)

        # Cache user
        self.user_cache[user_id] = user

        # Emit registration event
        await self.event_bus.emit(
            UserRegistered(
                user_id=user_id,
                username=username,
                email=email,
                authorization_level=authorization_level.value,
            )
        )

        # Update metrics
        self.metrics.increment("auth_registrations_total")
        self.auth_stats["registrations"] += 1

        self.logger.info(f"Registered new user: {username} ({user_id})")
        return user_id

    def _validate_password(self, password: str) -> bool:
        """Validate password against policy."""
        policy = self.password_policy

        if len(password) < policy.get("min_length", 8):
            return False

        if policy.get("require_uppercase", True) and not any(c.isupper() for c in password):
            return False

        if policy.get("require_lowercase", True) and not any(c.islower() for c in password):
            return False

        if policy.get("require_numbers", True) and not any(c.isdigit() for c in password):
            return False

        if policy.get("require_special", True) and not any(
            c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password
        ):
            return False

        return True

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt."""
        return bcrypt.hashpw((password + salt).encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    def _verify_password(self, password: str, salt: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw((password + salt).encode("utf-8"), password_hash.encode("utf-8"))
        except Exception:
            return False

    @handle_exceptions
    async def authenticate_user(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_fingerprint: Optional[str] = None,
    ) -> AuthSession:
        """
        Authenticate a user and create a session.

        Args:
            username: Username or email
            password: User password
            mfa_code: Optional MFA code
            ip_address: Client IP address
            user_agent: Client user agent
            device_fingerprint: Client device fingerprint

        Returns:
            Authentication session
        """
        start_time = time.time()

        try:
            # Get user
            user = await self._get_user_by_login(username)
            if not user:
                await self._log_auth_attempt(
                    None, username, ip_address, user_agent, False, "user_not_found"
                )
                raise AuthError("Invalid username or password")

            # Check if account is locked
            if user.is_locked and user.lock_until and datetime.now(timezone.utc) < user.lock_until:
                await self._log_auth_attempt(
                    user.user_id, username, ip_address, user_agent, False, "account_locked"
                )
                raise AuthError("Account is locked")

            # Unlock account if lock period expired
            if user.is_locked and user.lock_until and datetime.now(timezone.utc) >= user.lock_until:
                user.is_locked = False
                user.lock_reason = None
                user.lock_until = None
                user.failed_login_attempts = 0
                await self.auth_store.update_user(user)

            # Verify password
            if not self._verify_password(password, user.salt, user.password_hash):
                await self._handle_failed_login(user, ip_address, user_agent)
                raise AuthError("Invalid username or password")

            # Check MFA if required
            if user.require_mfa or self.require_mfa:
                if not mfa_code:
                    # Return partial authentication state
                    session = AuthSession(
                        session_id=str(uuid.uuid4()),
                        user_id=user.user_id,
                        token=self._generate_token(user.user_id, TokenType.TEMPORARY_TOKEN),
                        token_type=TokenType.TEMPORARY_TOKEN,
                        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                        ip_address=ip_address,
                        user_agent=user_agent,
                        device_fingerprint=device_fingerprint,
                        authentication_method=AuthenticationMethod.PASSWORD,
                        mfa_verified=False,
                    )
                    return session

                # Verify MFA code
                if not self._verify_mfa_code(user.totp_secret, mfa_code):
                    await self._handle_failed_login(user, ip_address, user_agent)
                    raise AuthError("Invalid MFA code")

            # Create authenticated session
            session = await self._create_authenticated_session(
                user, ip_address, user_agent, device_fingerprint, mfa_code is not None
            )

            # Update user login info
            user.last_login = datetime.now(timezone.utc)
            user.login_attempts += 1
            user.failed_login_attempts = 0
            await self.auth_store.update_user(user)

            # Update cache
            self.user_cache[user.user_id] = user
            self.active_sessions[session.session_id] = session

            # Create assistant session
            assistant_session_id = await self.session_manager.create_session(
                user_id=user.user_id,
                session_config=SessionConfiguration(
                    session_type=SessionType.AUTHENTICATED,
                    max_session_time=user.session_timeout,
                    encryption_enabled=user.preferred_security_level
                    in [SessionSecurityLevel.HIGH, SessionSecurityLevel.CRITICAL],
                ),
            )

            # Link sessions
            session.context = {"assistant_session_id": assistant_session_id}

            # Update user preferences in learning systems
            if self.preference_learner:
                await self.preference_learner.update_user_preferences(
                    user.user_id,
                    {
                        "interaction_mode": user.preferred_interaction_mode,
                        "language": user.preferred_language,
                        "accessibility": user.accessibility_settings,
                    },
                )

            # Emit authentication event
            await self.event_bus.emit(
                UserAuthenticated(
                    user_id=user.user_id,
                    session_id=session.session_id,
                    username=user.username,
                    authentication_method=session.authentication_method.value,
                    mfa_verified=session.mfa_verified,
                    ip_address=ip_address,
                )
            )

            # Update metrics
            auth_time = time.time() - start_time
            self.metrics.increment("auth_attempts_total")
            self.metrics.increment("auth_successes_total")
            self.metrics.record("auth_duration_seconds", auth_time)
            self.auth_stats["successful_logins"] += 1

            # Log successful authentication
            await self._log_auth_attempt(
                user.user_id, username, ip_address, user_agent, True, "success"
            )

            self.logger.info(f"User {username} authenticated successfully")
            return session

        except AuthError:
            self.metrics.increment("auth_failures_total")
            self.auth_stats["failed_logins"] += 1
            raise
        except Exception as e:
            self.metrics.increment("auth_failures_total")
            self.auth_stats["failed_logins"] += 1
            self.logger.error(f"Authentication error for {username}: {str(e)}")
            raise AuthError(f"Authentication failed: {str(e)}")

    async def _get_user_by_login(self, login: str) -> Optional[AuthUser]:
        """Get user by username or email."""
        # Try username first
        user = await self.auth_store.get_user_by_username(login)
        if user:
            return user

        # Try email if it looks like an email
        if "@" in login:
            user = await self.auth_store.get_user_by_email(login)

        return user

    def _verify_mfa_code(self, secret: str, code: str) -> bool:
        """Verify MFA TOTP code."""
        if not secret:
            return False

        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(code, valid_window=1)
        except Exception:
            return False

    async def _create_authenticated_session(
        self,
        user: AuthUser,
        ip_address: Optional[str],
        user_agent: Optional[str],
        device_fingerprint: Optional[str],
        mfa_verified: bool,
    ) -> AuthSession:
        """Create an authenticated session."""
        # Check concurrent session limit
        user_sessions = await self.auth_store.get_user_sessions(user.user_id)
        if len(user_sessions) >= user.max_concurrent_sessions:
            # Remove oldest session
            oldest_session = min(user_sessions, key=lambda s: s.last_activity)
            await self.revoke_session(oldest_session.session_id, "concurrent_limit_exceeded")

        # Create session
        session = AuthSession(
            session_id=str(uuid.uuid4()),
            user_id=user.user_id,
            token=self._generate_token(user.user_id, TokenType.SESSION_TOKEN),
            token_type=TokenType.SESSION_TOKEN,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=user.session_timeout),
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            security_level=user.preferred_security_level,
            authentication_method=(
                AuthenticationMethod.MULTI_FACTOR if mfa_verified else AuthenticationMethod.PASSWORD
            ),
            mfa_verified=mfa_verified,
            granted_permissions=user.permissions.copy(),
            accessible_components=self._get_user_accessible_components(user),
            allowed_workflows=self._get_user_allowed_workflows(user),
            plugin_access=user.plugin_permissions.copy(),
        )

        # Store session
        await self.auth_store.create_session(session)

        return session

    def _get_user_accessible_components(self, user: AuthUser) -> Set[str]:
        """Get components accessible to user based on authorization level."""
        accessible_components = set()

        if user.authorization_level in [
            AuthorizationLevel.USER,
            AuthorizationLevel.POWER_USER,
            AuthorizationLevel.ADMIN,
            AuthorizationLevel.SUPER_ADMIN,
        ]:
            accessible_components.update(
                [
                    "interaction_handler",
                    "session_manager",
                    "memory_manager",
                    "language_chain",
                    "intent_manager",
                    "sentiment_analyzer",
                ]
            )

        if user.authorization_level in [
            AuthorizationLevel.POWER_USER,
            AuthorizationLevel.ADMIN,
            AuthorizationLevel.SUPER_ADMIN,
        ]:
            accessible_components.update(
                [
                    "workflow_orchestrator",
                    "skill_factory",
                    "knowledge_graph",
                    "continual_learner",
                    "preference_learner",
                ]
            )

        if user.authorization_level in [AuthorizationLevel.ADMIN, AuthorizationLevel.SUPER_ADMIN]:
            accessible_components.update(["component_manager", "plugin_manager", "auth_service"])

        if user.authorization_level == AuthorizationLevel.SUPER_ADMIN:
            accessible_components.add("system_administration")

        return accessible_components

    def _get_user_allowed_workflows(self, user: AuthUser) -> Set[str]:
        """Get workflows allowed for user based on authorization level."""
        allowed_workflows = set()

        if user.authorization_level in [
            AuthorizationLevel.USER,
            AuthorizationLevel.POWER_USER,
            AuthorizationLevel.ADMIN,
            AuthorizationLevel.SUPER_ADMIN,
        ]:
            allowed_workflows.update(["builtin_qa", "builtin_multimodal"])

        if user.authorization_level in [
            AuthorizationLevel.POWER_USER,
            AuthorizationLevel.ADMIN,
            AuthorizationLevel.SUPER_ADMIN,
        ]:
            allowed_workflows.update(["advanced_reasoning", "custom_workflows"])

        if user.authorization_level in [AuthorizationLevel.ADMIN, AuthorizationLevel.SUPER_ADMIN]:
            allowed_workflows.update(["system_workflows", "maintenance_workflows"])

        return allowed_workflows

    async def _handle_failed_login(
        self, user: AuthUser, ip_address: Optional[str], user_agent: Optional[str]
    ) -> None:
        """Handle failed login attempt."""
        user.failed_login_attempts += 1
        user.last_failed_login = datetime.now(timezone.utc)

        # Lock account if too many failures
        if user.failed_login_attempts >= self.max_failed_attempts:
            user.is_locked = True
            user.lock_reason = "too_many_failed_attempts"
            user.lock_until = datetime.now(timezone.utc) + timedelta(
                seconds=self.account_lockout_duration
            )

            # Emit lockout event
            await self.event_bus.emit(
                SecurityViolation(
                    user_id=user.user_id,
                    violation_type="account_lockout",
                    details=f"Account locked after {user.failed_login_attempts} failed attempts",
                    ip_address=ip_address,
                )
            )

            self.metrics.increment("auth_lockouts_total")

        await self.auth_store.update_user(user)
        self.user_cache[user.user_id] = user

        # Log failed attempt
        await self._log_auth_attempt(
            user.user_id, user.username, ip_address, user_agent, False, "invalid_credentials"
        )

    def _generate_token(self, user_id: str, token_type: TokenType) -> str:
        """Generate JWT token for user."""
        payload = {
            "user_id": user_id,
            "token_type": token_type.value,
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(seconds=self.token_expiry),
        }

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    @handle_exceptions
    async def authorize_request(
        self,
        session_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Authorize a request for a specific resource and action.

        Args:
            session_id: Session ID making the request
            resource_type: Type of resource (component, workflow, plugin, etc.)
            resource_id: Specific resource identifier
            action: Action to be performed
            context: Additional context for authorization

        Returns:
            True if authorized, False otherwise
        """
        try:
            # Get session
            session = await self.get_session(session_id)
            if not session or not session.is_active:
                return False

            # Get user
            user = await self.get_user(session.user_id)
            if not user or not user.is_active:
                return False

            # Check resource-specific authorization
            is_authorized = await self._check_resource_authorization(
                user, session, resource_type, resource_id, action, context
            )

            # Create authorization request for auditing
            auth_request = AuthorizationRequest(
                request_id=str(uuid.uuid4()),
                user_id=user.user_id,
                session_id=session_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                context=context or {},
            )

            # Emit authorization event
            if is_authorized:
                await self.event_bus.emit(
                    UserAuthorized(
                        user_id=user.user_id,
                        session_id=session_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        action=action,
                    )
                )
                self.metrics.increment("authorization_grants_total")
            else:
                await self.event_bus.emit(
                    UserUnauthorized(
                        user_id=user.user_id,
                        session_id=session_id,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        action=action,
                        reason="insufficient_permissions",
                    )
                )
                self.metrics.increment("authorization_denials_total")

            self.metrics.increment("authorization_checks_total")

            # Log authorization check
            await self._log_authorization_check(auth_request, is_authorized)

            return is_authorized

        except Exception as e:
            self.logger.error(f"Authorization error for session {session_id}: {str(e)}")
            return False

    async def _check_resource_authorization(
        self,
        user: AuthUser,
        session: AuthSession,
        resource_type: str,
        resource_id: str,
        action: str,
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """Check authorization for specific resource type."""
        if resource_type == "component":
            return resource_id in session.accessible_components

        elif resource_type == "workflow":
            return resource_id in session.allowed_workflows

        elif resource_type == "plugin":
            return resource_id in session.plugin_access

        elif resource_type == "session":
            # Can only access own sessions unless admin
            if resource_id == session.session_id:
                return True
            return user.authorization_level in [
                AuthorizationLevel.ADMIN,
                AuthorizationLevel.SUPER_ADMIN,
            ]

        elif resource_type == "user_data":
            # Can only access own data unless admin
            if context and context.get("target_user_id") == user.user_id:
                return True
            return user.authorization_level in [
                AuthorizationLevel.ADMIN,
                AuthorizationLevel.SUPER_ADMIN,
            ]

        elif resource_type == "system":
            return user.authorization_level in [
                AuthorizationLevel.ADMIN,
                AuthorizationLevel.SUPER_ADMIN,
            ]

        else:
            # Default to permission-based check
            required_permission = f"{resource_type}:{action}"
            return required_permission in user.permissions

    @handle_exceptions
    async def get_session(self, session_id: str) -> Optional[AuthSession]:
        """
        Get authentication session.

        Args:
            session_id: Session identifier

        Returns:
            Authentication session or None
        """
        # Check cache first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]

            # Check if session is still valid
            if session.expires_at > datetime.now(timezone.utc) and session.is_active:
                # Update last activity
                session.last_activity = datetime.now(timezone.utc)
                await self.auth_store.update_session(session)
                return session
            else:
                # Session expired or inactive
                await self.revoke_session(session_id, "expired")
                return None

        # Load from storage
        session = await self.auth_store.get_session(session_id)
        if session and session.is_active and not session.is_revoked:
            # Check expiration
            if session.expires_at > datetime.now(timezone.utc):
                self.active_sessions[session_id] = session
                return session
            else:
                await self.revoke_session(session_id, "expired")

        return None

    @handle_exceptions
    async def get_user(self, user_id: str) -> Optional[AuthUser]:
        """
        Get user information.

        Args:
            user_id: User identifier

        Returns:
            User information or None
        """
        # Check cache first
        if user_id in self.user_cache:
            return self.user_cache[user_id]

        # Load from storage
        user = await self.auth_store.get_user(user_id)
        if user:
            self.user_cache[user_id] = user

        return user

    @handle_exceptions
    async def revoke_session(self, session_id: str, reason: str = "user_logout") -> None:
        """
        Revoke an authentication session.

        Args:
            session_id: Session to revoke
            reason: Reason for revocation
        """
        session = await self.get_session(session_id)
        if not session:
            return

        # Update session
        session.is_active = False
        session.is_revoked = True
        session.revocation_reason = reason

        # Store updated session
        await self.auth_store.update_session(session)

        # Remove from cache
        self.active_sessions.pop(session_id, None)

        # End assistant session if linked
        if hasattr(session, "context") and "assistant_session_id" in session.context:
            try:
                await self.session_manager.end_session(
                    session.context["assistant_session_id"], reason
                )
            except Exception as e:
                self.logger.warning(f"Failed to end assistant session: {str(e)}")

        # Emit logout event
        await self.event_bus.emit(
            UserLoggedOut(user_id=session.user_id, session_id=session_id, reason=reason)
        )

        # Update metrics
        self.session_stats["revoked_sessions"] += 1

        self.logger.info(f"Revoked session {session_id} for user {session.user_id}: {reason}")

    @handle_exceptions
    async def revoke_all_user_sessions(self, user_id: str, reason: str = "security_measure") -> int:
        """
        Revoke all sessions for a user.

        Args:
            user_id: User identifier
            reason: Reason for revocation

        Returns:
            Number of sessions revoked
        """
        user_sessions = await self.auth_store.get_user_sessions(user_id)
        revoked_count = 0

        for session in user_sessions:
            if session.is_active and not session.is_revoked:
                await self.revoke_session(session.session_id, reason)
                revoked_count += 1

        self.logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count

    @handle_exceptions
    async def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """
        Setup multi-factor authentication for a user.

        Args:
            user_id: User identifier

        Returns:
            MFA setup information including QR code
        """
        user = await self.get_user(user_id)
        if not user:
            raise AuthError(f"User {user_id} not found")

        # Generate TOTP secret
        secret = pyotp.random_base32()

        # Create provisioning URI
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(name=user.username, issuer_name="AI Assistant")

        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        qr_code_data = base64.b64encode(buffer.getvalue()).decode()

        # Generate backup codes
        backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]

        # Update user (don't save secret yet - wait for verification)
        setup_data = {
            "secret": secret,
            "backup_codes": backup_codes,
            "qr_code": qr_code_data,
            "provisioning_uri": provisioning_uri,
        }

        # Store temporarily in cache
        await self.redis_cache.set(f"mfa_setup_{user_id}", json.dumps(setup_data), ttl=300)
