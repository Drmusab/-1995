"""
Advanced Authorization Management System
Author: Drmusab
Last Modified: 2025-05-26 16:45:12 UTC

This module provides comprehensive authorization and access control for the AI assistant,
including role-based access control (RBAC), permission management, policy evaluation,
resource-level security, and seamless integration with all core system components.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, AsyncGenerator, TypeVar
import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import json
import hashlib
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import logging
import inspect
from concurrent.futures import ThreadPoolExecutor
import re

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    UserAuthenticated, UserAuthorized, UserAuthorizationFailed, PermissionGranted,
    PermissionRevoked, RoleAssigned, RoleRevoked, PolicyCreated, PolicyUpdated,
    PolicyDeleted, ResourceAccessGranted, ResourceAccessDenied, SecurityViolation,
    AuditLogCreated, SessionSecurityCheck, ComponentSecurityCheck,
    WorkflowSecurityCheck, PluginSecurityCheck, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.encryption import EncryptionManager

# Storage and caching
from src.integrations.storage.database import DatabaseManager
from src.integrations.cache.redis_cache import RedisCache

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger

# Type definitions
T = TypeVar('T')


class PermissionType(Enum):
    """Types of permissions in the system."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"
    ADMIN = "admin"
    MANAGE = "manage"
    VIEW = "view"
    CONFIGURE = "configure"
    MONITOR = "monitor"
    AUDIT = "audit"


class ResourceType(Enum):
    """Types of resources that can be protected."""
    SYSTEM = "system"
    COMPONENT = "component"
    WORKFLOW = "workflow"
    SESSION = "session"
    USER_DATA = "user_data"
    PLUGIN = "plugin"
    SKILL = "skill"
    MEMORY = "memory"
    API_ENDPOINT = "api_endpoint"
    CONFIGURATION = "configuration"
    LOGS = "logs"
    METRICS = "metrics"


class PolicyType(Enum):
    """Types of authorization policies."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"
    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    DEVICE_BASED = "device_based"
    CONTEXT_BASED = "context_based"


class AccessDecision(Enum):
    """Authorization decision outcomes."""
    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class Permission:
    """Represents a permission in the authorization system."""
    permission_id: str
    name: str
    permission_type: PermissionType
    resource_type: ResourceType
    resource_id: Optional[str] = None
    description: Optional[str] = None
    
    # Scope and constraints
    scope: str = "global"  # global, tenant, user, session
    conditions: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    # Hierarchy
    parent_permissions: Set[str] = field(default_factory=set)
    child_permissions: Set[str] = field(default_factory=set)
    
    # Tags and categorization
    tags: Set[str] = field(default_factory=set)
    category: Optional[str] = None


@dataclass
class Role:
    """Represents a role with associated permissions."""
    role_id: str
    name: str
    description: Optional[str] = None
    
    # Permissions
    permissions: Set[str] = field(default_factory=set)
    inherited_roles: Set[str] = field(default_factory=set)
    
    # Scope and constraints
    scope: str = "global"
    is_system_role: bool = False
    is_active: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Role hierarchy
    parent_roles: Set[str] = field(default_factory=set)
    child_roles: Set[str] = field(default_factory=set)
    
    # Configuration
    max_session_duration: Optional[timedelta] = None
    allowed_resources: Set[str] = field(default_factory=set)
    denied_resources: Set[str] = field(default_factory=set)


@dataclass
class Policy:
    """Represents an authorization policy."""
    policy_id: str
    name: str
    policy_type: PolicyType = PolicyType.ALLOW
    description: Optional[str] = None
    
    # Policy definition
    rules: List[Dict[str, Any]] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Scope
    applies_to_users: Set[str] = field(default_factory=set)
    applies_to_roles: Set[str] = field(default_factory=set)
    applies_to_resources: Set[str] = field(default_factory=set)
    
    # Timing
    effective_from: Optional[datetime] = None
    effective_until: Optional[datetime] = None
    
    # Evaluation
    priority: int = 100  # Higher numbers = higher priority
    is_active: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    version: int = 1


@dataclass
class Resource:
    """Represents a protected resource."""
    resource_id: str
    resource_type: ResourceType
    name: str
    description: Optional[str] = None
    
    # Resource properties
    owner: Optional[str] = None
    parent_resource: Optional[str] = None
    child_resources: Set[str] = field(default_factory=set)
    
    # Access control
    required_permissions: Set[str] = field(default_factory=set)
    allowed_roles: Set[str] = field(default_factory=set)
    denied_roles: Set[str] = field(default_factory=set)
    
    # Security attributes
    security_level: str = "standard"  # low, standard, high, critical
    encryption_required: bool = False
    audit_required: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Tags and properties
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorizationContext:
    """Context information for authorization decisions."""
    user_id: str
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # User context
    user_roles: Set[str] = field(default_factory=set)
    user_permissions: Set[str] = field(default_factory=set)
    user_attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Request context
    resource_id: str = ""
    resource_type: Optional[ResourceType] = None
    requested_permission: Optional[PermissionType] = None
    action: Optional[str] = None
    
    # Environment context
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    
    # Additional context
    tenant_id: Optional[str] = None
    component: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Security context
    authentication_method: Optional[str] = None
    trust_level: str = "standard"
    risk_score: float = 0.0
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""
    decision: AccessDecision
    user_id: str
    resource_id: str
    permission: str
    
    # Decision details
    reason: str = ""
    applicable_policies: List[str] = field(default_factory=list)
    evaluated_permissions: List[str] = field(default_factory=list)
    
    # Timing
    evaluation_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Context
    context_hash: Optional[str] = None
    request_id: Optional[str] = None
    
    # Audit information
    audit_required: bool = True
    risk_score: float = 0.0
    confidence_score: float = 1.0
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuthorizationError(Exception):
    """Custom exception for authorization operations."""
    
    def __init__(self, message: str, user_id: Optional[str] = None, 
                 resource_id: Optional[str] = None, permission: Optional[str] = None):
        super().__init__(message)
        self.user_id = user_id
        self.resource_id = resource_id
        self.permission = permission
        self.timestamp = datetime.now(timezone.utc)


class PermissionEvaluator(ABC):
    """Abstract base class for permission evaluators."""
    
    @abstractmethod
    async def evaluate(self, context: AuthorizationContext) -> AuthorizationResult:
        """Evaluate authorization for the given context."""
        pass
    
    @abstractmethod
    def can_evaluate(self, context: AuthorizationContext) -> bool:
        """Check if this evaluator can handle the context."""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get the priority of this evaluator (higher = evaluated first)."""
        pass


class RoleBasedEvaluator(PermissionEvaluator):
    """Role-based access control evaluator."""
    
    def __init__(self, authorization_manager: 'AuthorizationManager'):
        self.authorization_manager = authorization_manager
        self.logger = get_logger(__name__)
    
    def can_evaluate(self, context: AuthorizationContext) -> bool:
        """Always can evaluate role-based permissions."""
        return True
    
    def get_priority(self) -> int:
        """Medium priority for RBAC evaluation."""
        return 50
    
    async def evaluate(self, context: AuthorizationContext) -> AuthorizationResult:
        """Evaluate based on user roles and permissions."""
        start_time = time.time()
        
        # Check direct permissions
        if str(context.requested_permission.value) in context.user_permissions:
            return AuthorizationResult(
                decision=AccessDecision.ALLOW,
                user_id=context.user_id,
                resource_id=context.resource_id,
                permission=str(context.requested_permission.value),
                reason="Direct permission granted",
                evaluation_time=time.time() - start_time
            )
        
        # Check role-based permissions
        for role_id in context.user_roles:
            role = await self.authorization_manager.get_role(role_id)
            if role and str(context.requested_permission.value) in role.permissions:
                return AuthorizationResult(
                    decision=AccessDecision.ALLOW,
                    user_id=context.user_id,
                    resource_id=context.resource_id,
                    permission=str(context.requested_permission.value),
                    reason=f"Permission granted through role: {role.name}",
                    evaluation_time=time.time() - start_time
                )
        
        # No matching permissions found
        return AuthorizationResult(
            decision=AccessDecision.DENY,
            user_id=context.user_id,
            resource_id=context.resource_id,
            permission=str(context.requested_permission.value),
            reason="No matching permissions found",
            evaluation_time=time.time() - start_time
        )


class PolicyBasedEvaluator(PermissionEvaluator):
    """Policy-based authorization evaluator."""
    
    def __init__(self, authorization_manager: 'AuthorizationManager'):
        self.authorization_manager = authorization_manager
        self.logger = get_logger(__name__)
    
    def can_evaluate(self, context: AuthorizationContext) -> bool:
        """Can evaluate if there are applicable policies."""
        return True
    
    def get_priority(self) -> int:
        """High priority for policy evaluation."""
        return 75
    
    async def evaluate(self, context: AuthorizationContext) -> AuthorizationResult:
        """Evaluate based on authorization policies."""
        start_time = time.time()
        
        # Get applicable policies
        policies = await self.authorization_manager.get_applicable_policies(context)
        
        # Sort by priority (highest first)
        policies.sort(key=lambda p: p.priority, reverse=True)
        
        applicable_policy_ids = []
        
        for policy in policies:
            if await self._evaluate_policy(policy, context):
                applicable_policy_ids.append(policy.policy_id)
                
                if policy.policy_type == PolicyType.DENY:
                    return AuthorizationResult(
                        decision=AccessDecision.DENY,
                        user_id=context.user_id,
                        resource_id=context.resource_id,
                        permission=str(context.requested_permission.value),
                        reason=f"Denied by policy: {policy.name}",
                        applicable_policies=applicable_policy_ids,
                        evaluation_time=time.time() - start_time
                    )
                elif policy.policy_type == PolicyType.ALLOW:
                    return AuthorizationResult(
                        decision=AccessDecision.ALLOW,
                        user_id=context.user_id,
                        resource_id=context.resource_id,
                        permission=str(context.requested_permission.value),
                        reason=f"Allowed by policy: {policy.name}",
                        applicable_policies=applicable_policy_ids,
                        evaluation_time=time.time() - start_time
                    )
        
        # No applicable policies
        return AuthorizationResult(
            decision=AccessDecision.ABSTAIN,
            user_id=context.user_id,
            resource_id=context.resource_id,
            permission=str(context.requested_permission.value),
            reason="No applicable policies found",
            applicable_policies=applicable_policy_ids,
            evaluation_time=time.time() - start_time
        )
    
    async def _evaluate_policy(self, policy: Policy, context: AuthorizationContext) -> bool:
        """Evaluate if a policy applies to the context."""
        try:
            # Check if policy applies to user
            if policy.applies_to_users and context.user_id not in policy.applies_to_users:
                return False
            
            # Check if policy applies to user's roles
            if policy.applies_to_roles and not context.user_roles.intersection(policy.applies_to_roles):
                return False
            
            # Check if policy applies to resource
            if policy.applies_to_resources and context.resource_id not in policy.applies_to_resources:
                return False
            
            # Check time constraints
            current_time = datetime.now(timezone.utc)
            if policy.effective_from and current_time < policy.effective_from:
                return False
            if policy.effective_until and current_time > policy.effective_until:
                return False
            
            # Evaluate policy conditions
            for condition_key, condition_value in policy.conditions.items():
                if not await self._evaluate_condition(condition_key, condition_value, context):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating policy {policy.policy_id}: {str(e)}")
            return False
    
    async def _evaluate_condition(self, condition_key: str, condition_value: Any, 
                                context: AuthorizationContext) -> bool:
        """Evaluate a specific policy condition."""
        try:
            if condition_key == "time_of_day":
                current_hour = datetime.now(timezone.utc).hour
                return current_hour in condition_value
            
            elif condition_key == "ip_address":
                if context.ip_address:
                    return self._match_ip_pattern(context.ip_address, condition_value)
            
            elif condition_key == "user_attribute":
                attr_name, expected_value = condition_value.get("name"), condition_value.get("value")
                return context.user_attributes.get(attr_name) == expected_value
            
            elif condition_key == "device_type":
                return context.device_info.get("type") in condition_value
            
            elif condition_key == "trust_level":
                return context.trust_level in condition_value
            
            elif condition_key == "risk_score":
                max_risk = condition_value.get("max", 1.0)
                return context.risk_score <= max_risk
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error evaluating condition {condition_key}: {str(e)}")
            return False
    
    def _match_ip_pattern(self, ip_address: str, patterns: List[str]) -> bool:
        """Match IP address against patterns (supports CIDR notation)."""
        try:
            import ipaddress
            ip = ipaddress.ip_address(ip_address)
            
            for pattern in patterns:
                if "/" in pattern:  # CIDR notation
                    network = ipaddress.ip_network(pattern, strict=False)
                    if ip in network:
                        return True
                else:  # Exact match or wildcard
                    if pattern == ip_address or pattern == "*":
                        return True
            
            return False
            
        except Exception:
            return False


class ResourceBasedEvaluator(PermissionEvaluator):
    """Resource-specific authorization evaluator."""
    
    def __init__(self, authorization_manager: 'AuthorizationManager'):
        self.authorization_manager = authorization_manager
        self.logger = get_logger(__name__)
    
    def can_evaluate(self, context: AuthorizationContext) -> bool:
        """Can evaluate if resource information is available."""
        return bool(context.resource_id)
    
    def get_priority(self) -> int:
        """High priority for resource-specific evaluation."""
        return 80
    
    async def evaluate(self, context: AuthorizationContext) -> AuthorizationResult:
        """Evaluate based on resource-specific permissions."""
        start_time = time.time()
        
        # Get resource information
        resource = await self.authorization_manager.get_resource(context.resource_id)
        if not resource:
            return AuthorizationResult(
                decision=AccessDecision.DENY,
                user_id=context.user_id,
                resource_id=context.resource_id,
                permission=str(context.requested_permission.value),
                reason="Resource not found",
                evaluation_time=time.time() - start_time
            )
        
        # Check if user is resource owner
        if resource.owner == context.user_id:
            return AuthorizationResult(
                decision=AccessDecision.ALLOW,
                user_id=context.user_id,
                resource_id=context.resource_id,
                permission=str(context.requested_permission.value),
                reason="User is resource owner",
                evaluation_time=time.time() - start_time
            )
        
        # Check denied roles
        if context.user_roles.intersection(resource.denied_roles):
            return AuthorizationResult(
                decision=AccessDecision.DENY,
                user_id=context.user_id,
                resource_id=context.resource_id,
                permission=str(context.requested_permission.value),
                reason="User role is explicitly denied for this resource",
                evaluation_time=time.time() - start_time
            )
        
        # Check allowed roles
        if resource.allowed_roles and context.user_roles.intersection(resource.allowed_roles):
            return AuthorizationResult(
                decision=AccessDecision.ALLOW,
                user_id=context.user_id,
                resource_id=context.resource_id,
                permission=str(context.requested_permission.value),
                reason="User role is allowed for this resource",
                evaluation_time=time.time() - start_time
            )
        
        # Check required permissions
        if resource.required_permissions:
            permission_str = str(context.requested_permission.value)
            if permission_str in resource.required_permissions:
                # Need to check if user has this specific permission
                return AuthorizationResult(
                    decision=AccessDecision.ABSTAIN,
                    user_id=context.user_id,
                    resource_id=context.resource_id,
                    permission=permission_str,
                    reason="Resource requires specific permission check",
                    evaluation_time=time.time() - start_time
                )
        
        # No specific resource rules apply
        return AuthorizationResult(
            decision=AccessDecision.ABSTAIN,
            user_id=context.user_id,
            resource_id=context.resource_id,
            permission=str(context.requested_permission.value),
            reason="No resource-specific rules apply",
            evaluation_time=time.time() - start_time
        )


class AuthorizationManager:
    """
    Advanced Authorization Management System for the AI Assistant.
    
    This manager provides comprehensive access control including:
    - Role-based access control (RBAC)
    - Permission management and evaluation
    - Policy-based authorization
    - Resource-level security
    - Context-aware authorization decisions
    - Audit logging and compliance
    - Integration with all core system components
    - Performance optimization and caching
    - Real-time permission updates
    - Multi-tenant support
    """
    
    def __init__(self, container: Container):
        """
        Initialize the authorization manager.
        
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
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
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
        
        # Authorization data
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.policies: Dict[str, Policy] = {}
        self.resources: Dict[str, Resource] = {}
        
        # User associations
        self.user_roles: Dict[str, Set[str]] = defaultdict(set)
        self.user_permissions: Dict[str, Set[str]] = defaultdict(set)
        
        # Evaluation system
        self.evaluators: List[PermissionEvaluator] = []
        self.evaluation_cache: Dict[str, AuthorizationResult] = {}
        
        # Performance tracking
        self.authorization_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.evaluation_times: deque = deque(maxlen=1000)
        
        # Configuration
        self.cache_ttl = self.config.get("authorization.cache_ttl", 300)
        self.enable_audit_logging = self.config.get("authorization.audit_logging", True)
        self.enable_caching = self.config.get("authorization.enable_caching", True)
        self.enable_policy_evaluation = self.config.get("authorization.enable_policies", True)
        self.default_decision = AccessDecision.DENY
        
        # Thread safety
        self.lock = asyncio.Lock()
        
        # Initialize components
        self._setup_evaluators()
        self._setup_monitoring()
        self._setup_default_permissions()
        
        # Register health check
        self.health_check.register_component("authorization_manager", self._health_check_callback)
        
        self.logger.info("AuthorizationManager initialized successfully")

    def _setup_evaluators(self) -> None:
        """Setup permission evaluators."""
        try:
            # Add built-in evaluators
            self.evaluators = [
                ResourceBasedEvaluator(self),
                PolicyBasedEvaluator(self) if self.enable_policy_evaluation else None,
                RoleBasedEvaluator(self)
            ]
            
            # Remove None evaluators
            self.evaluators = [e for e in self.evaluators if e is not None]
            
            # Sort by priority (highest first)
            self.evaluators.sort(key=lambda e: e.get_priority(), reverse=True)
            
            self.logger.info(f"Initialized {len(self.evaluators)} permission evaluators")
            
        except Exception as e:
            self.logger.error(f"Failed to setup evaluators: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register authorization metrics
            self.metrics.register_counter("authorization_checks_total")
            self.metrics.register_counter("authorization_allowed_total")
            self.metrics.register_counter("authorization_denied_total")
            self.metrics.register_histogram("authorization_evaluation_duration_seconds")
            self.metrics.register_gauge("active_permissions")
            self.metrics.register_gauge("active_roles")
            self.metrics.register_gauge("active_policies")
            self.metrics.register_counter("authorization_cache_hits")
            self.metrics.register_counter("authorization_cache_misses")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    def _setup_default_permissions(self) -> None:
        """Setup default system permissions and roles."""
        try:
            # System permissions
            system_permissions = [
                Permission("system.admin", "System Administration", PermissionType.ADMIN, ResourceType.SYSTEM),
                Permission("system.monitor", "System Monitoring", PermissionType.MONITOR, ResourceType.SYSTEM),
                Permission("system.configure", "System Configuration", PermissionType.CONFIGURE, ResourceType.SYSTEM),
                
                # Component permissions
                Permission("component.read", "Read Components", PermissionType.READ, ResourceType.COMPONENT),
                Permission("component.manage", "Manage Components", PermissionType.MANAGE, ResourceType.COMPONENT),
                
                # Workflow permissions
                Permission("workflow.execute", "Execute Workflows", PermissionType.EXECUTE, ResourceType.WORKFLOW),
                Permission("workflow.create", "Create Workflows", PermissionType.CREATE, ResourceType.WORKFLOW),
                Permission("workflow.manage", "Manage Workflows", PermissionType.MANAGE, ResourceType.WORKFLOW),
                
                # Session permissions
                Permission("session.create", "Create Sessions", PermissionType.CREATE, ResourceType.SESSION),
                Permission("session.manage", "Manage Sessions", PermissionType.MANAGE, ResourceType.SESSION),
                
                # Plugin permissions
                Permission("plugin.install", "Install Plugins", PermissionType.CREATE, ResourceType.PLUGIN),
                Permission("plugin.manage", "Manage Plugins", PermissionType.MANAGE, ResourceType.PLUGIN),
                
                # API permissions
                Permission("api.read", "Read API", PermissionType.READ, ResourceType.API_ENDPOINT),
                Permission("api.write", "Write API", PermissionType.WRITE, ResourceType.API_ENDPOINT),
            ]
            
            for perm in system_permissions:
                self.permissions[perm.permission_id] = perm
            
            # System roles
            admin_role = Role(
                role_id="system.admin",
                name="System Administrator",
                description="Full system administration access",
                permissions={perm.permission_id for perm in system_permissions},
                is_system_role=True
            )
            
            user_role = Role(
                role_id="system.user",
                name="Standard User",
                description="Standard user access",
                permissions={
                    "session.create", "workflow.execute", "api.read",
                    "component.read"
                },
                is_system_role=True
            )
            
            self.roles[admin_role.role_id] = admin_role
            self.roles[user_role.role_id] = user_role
            
            self.logger.info(f"Setup {len(system_permissions)} default permissions and {len([admin_role, user_role])} default roles")
            
        except Exception as e:
            self.logger.error(f"Failed to setup default permissions: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the authorization manager."""
        try:
            # Load data from persistence if available
            if self.database:
                await self._load_from_database()
            
            # Register event handlers
            await self._register_event_handlers()
            
            # Start background tasks
            asyncio.create_task(self._cache_cleanup_loop())
            asyncio.create_task(self._audit_processing_loop())
            asyncio.create_task(self._permission_sync_loop())
            
            self.logger.info("AuthorizationManager initialization completed")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AuthorizationManager: {str(e)}")
            raise AuthorizationError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User authentication events
        self.event_bus.subscribe("user_authenticated", self._handle_user_authenticated)
        self.event_bus.subscribe("user_logged_out", self._handle_user_logout)
        
        # Session events
        self.event_bus.subscribe("session_started", self._handle_session_started)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)
        
        # Component events
        self.event_bus.subscribe("component_registered", self._handle_component_registered)
        
        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    @handle_exceptions
    async def check_permission(
        self,
        user_id: str,
        resource_id: str,
        permission: Union[str, PermissionType],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a user has permission to access a resource.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            permission: Permission to check
            context: Additional context for evaluation
            
        Returns:
            True if permission is granted, False otherwise
        """
        result = await self.evaluate_authorization(user_id, resource_id, permission, context)
        return result.decision == AccessDecision.ALLOW

    @handle_exceptions
    async def evaluate_authorization(
        self,
        user_id: str,
        resource_id: str,
        permission: Union[str, PermissionType],
        context: Optional[Dict[str, Any]] = None
    ) -> AuthorizationResult:
        """
        Evaluate authorization for a user and resource.
        
        Args:
            user_id: User identifier
            resource_id: Resource identifier
            permission: Permission to check
            context: Additional context for evaluation
            
        Returns:
            Authorization result with decision and details
        """
        start_time = time.time()
        
        # Convert permission to enum if needed
        if isinstance(permission, str):
            try:
                permission_enum = PermissionType(permission)
            except ValueError:
                # Custom permission string
                permission_enum = permission
        else:
            permission_enum = permission
        
        # Create authorization context
        auth_context = await self._build_authorization_context(
            user_id, resource_id, permission_enum, context
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(auth_context)
        if self.enable_caching:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.metrics.increment("authorization_cache_hits")
                return cached_result
        
        self.metrics.increment("authorization_cache_misses")
        
        try:
            with self.tracer.trace("authorization_evaluation") as span:
                span.set_attributes({
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "permission": str(permission_enum),
                    "resource_type": auth_context.resource_type.value if auth_context.resource_type else "unknown"
                })
                
                # Evaluate using all evaluators
                results = []
                for evaluator in self.evaluators:
                    if evaluator.can_evaluate(auth_context):
                        try:
                            result = await evaluator.evaluate(auth_context)
                            results.append(result)
                            
                            # Stop on first DENY decision (fail-safe)
                            if result.decision == AccessDecision.DENY:
                                break
                                
                        except Exception as e:
                            self.logger.error(f"Evaluator {type(evaluator).__name__} failed: {str(e)}")
                
                # Combine results
                final_result = await self._combine_evaluation_results(results, auth_context)
                
                # Update metrics
                evaluation_time = time.time() - start_time
                final_result.evaluation_time = evaluation_time
                self.evaluation_times.append(evaluation_time)
                
                self.metrics.increment("authorization_checks_total")
                if final_result.decision == AccessDecision.ALLOW:
                    self.metrics.increment("authorization_allowed_total")
                else:
                    self.metrics.increment("authorization_denied_total")
                
                self.metrics.record("authorization_evaluation_duration_seconds", evaluation_time)
                
                # Cache result
                if self.enable_caching:
                    await self._cache_result(cache_key, final_result)
                
                # Audit logging
                if self.enable_audit_logging:
                    await self._log_authorization_event(final_result, auth_context)
                
                # Emit events
                if final_result.decision == AccessDecision.ALLOW:
                    await self.event_bus.emit(UserAuthorized(
                        user_id=user_id,
                        resource_id=resource_id,
                        permission=str(permission_enum),
                        evaluation_time=evaluation_time
                    ))
                else:
                    await self.event_bus.emit(UserAuthorizationFailed(
                        user_id=user_id,
                        resource_id=resource_id,
                        permission=str(permission_enum),
                        reason=final_result.reason,
                        evaluation_time=evaluation_time
                    ))
                
                return final_result
                
        except Exception as e:
            # Default to deny on error
            error_result = AuthorizationResult(
                decision=AccessDecision.DENY,
                user_id=user_id,
                resource_id=resource_id,
                permission=str(permission_enum),
                reason=f"Authorization evaluation failed: {str(e)}",
                evaluation_time=time.time() - start_time
            )
            
            await self.event_bus.emit(SecurityViolation(
                user_id=user_id,
                violation_type="authorization_error",
                details={"error": str(e), "resource_id": resource_id},
                severity="high"
            ))
            
            self.logger.error(f"Authorization evaluation failed for user {user_id}: {str(e)}")
            return error_result

    async def _build_authorization_context(
        self,
        user_id: str,
        resource_id: str,
        permission: Union[str, PermissionType],
        context: Optional[Dict[str, Any]]
    ) -> AuthorizationContext:
        """Build comprehensive authorization context."""
        auth_context = AuthorizationContext(
            user_id=user_id,
            resource_id=resource_id,
            requested_permission=permission if isinstance(permission, PermissionType) else None,
            action=str(permission)
        )
        
        # Load user roles and permissions
        auth_context.user_roles = self.user_roles.get(user_id, set())
        auth_context.user_permissions = self.user_permissions.get(user_id, set())
        
        # Add inherited permissions from roles
        for role_id in auth_context.user_roles:
            role = self.roles.get(role_id)
            if role:
                auth_context.user_permissions.update(role.permissions)
        
        # Get resource information
        resource = self.resources.get(resource_id)
        if resource:
            auth_context.resource_type = resource.resource_type
        
        # Add additional context
        if context:
            auth_context.session_id = context.get("session_id")
            auth_context.ip_address = context.get("ip_address")
            auth_context.user_agent = context.get("user_agent")
            auth_context.device_info = context.get("device_info", {})
            auth_context.tenant_id = context.get("tenant_id")
            auth_context.component = context.get("component")
            auth_context.workflow_id = context.get("workflow_id")
            auth_context.trust_level = context.get("trust_level", "standard")
            auth_context.risk_score = context.get("risk_score", 0.0)
            auth_context.custom_attributes.update(context.get("custom_attributes", {}))
        
        # Get user attributes from authentication manager
        if self.auth_manager:
            try:
                user_info = await self.auth_manager.get_user_info(user_id)
                if user_info:
                    auth_context.user_attributes.update(user_info.get("attributes", {}))
                    auth_context.authentication_method = user_info.get("auth_method")
            except Exception as e:
                self.logger.warning(f"Failed to get user attributes: {str(e)}")
        
        return auth_context

    async def _combine_evaluation_results(
        self,
        results: List[AuthorizationResult],
        context: AuthorizationContext
    ) -> AuthorizationResult:
        """Combine multiple evaluation results into a final decision."""
        if not results:
            return AuthorizationResult(
                decision=self.default_decision,
                user_id=context.user_id,
                resource_id=context.resource_id,
                permission=str(context.requested_permission),
                reason="No evaluators returned results"
            )
        
        # Check for explicit DENY (takes precedence)
        for result in results:
            if result.decision == AccessDecision.DENY:
                return result
        
        # Check for explicit ALLOW
        for result in results:
            if result.decision == AccessDecision.ALLOW:
                return result
        
        # All results were ABSTAIN or NOT_APPLICABLE
        return AuthorizationResult(
            decision=self.default_decision,
            user_id=context.user_id,
            resource_id=context.resource_id,
            permission=str(context.requested_permission),
            reason="No definitive authorization decision from evaluators"
        )

    def _generate_cache_key(self, context: AuthorizationContext) -> str:
        """Generate cache key for authorization context."""
        key_parts = [
            context.user_id,
            context.resource_id,
            str(context.requested_permission),
            "|".join(sorted(context.user_roles)),
            "|".join(sorted(context.user_permissions)),
            context.trust_level,
            str(int(context.risk_score * 100))
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _get_cached_result(self, cache_key: str) -> Optional[AuthorizationResult]:
        """Get cached authorization result."""
        if not self.cache:
            return self.evaluation_cache.get(cache_key)
        
        try:
            cached_data = await self.cache.get(f"auth:{cache_key}")
            if cached_data:
                return AuthorizationResult(**json.loads(cached_data))
        except Exception as e:
            self.logger.warning(f"Failed to get cached result: {str(e)}")
        
        return None

    async def _cache_result(self, cache_key: str, result: AuthorizationResult) -> None:
        """Cache authorization result."""
        try:
            if self.cache:
                await self.cache.set(
                    f"auth:{cache_key}",
                    json.dumps(asdict(result), default=str),
                    ttl=self.cache_ttl
                )
            else:
                self.evaluation_cache[cache_key] = result
                
                # Limit in-memory cache size
                if len(self.evaluation_cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(self.evaluation_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.evaluation_cache[key]
        
        except Exception as e:
            self.logger.warning(f"Failed to cache result: {str(e)}")

    async def _log_authorization_event(self, result: AuthorizationResult, context: AuthorizationContext) -> None:
        """Log authorization event for audit purposes."""
        try:
            audit_data = {
                "event_type": "authorization_check",
                "user_id": context.user_id,
                "resource_id": context.resource_id,
                "permission": str(context.requested_permission),
                "decision": result.decision.value,
                "reason": result.reason,
                "evaluation_time": result.evaluation_time,
                "timestamp": result.timestamp.isoformat(),
                "context": {
                    "session_id": context.session_id,
                    "ip_address": context.ip_address,
                    "user_agent": context.user_agent,
                    "component": context.component,
                    "workflow_id": context.workflow_id,
                    "trust_level": context.trust_level,
                    "risk_score": context.risk_score
                }
            }
            
            await self.event_bus.emit(AuditLogCreated(
                event_type="authorization_check",
                user_id=context.user_id,
                data=audit_data,
                timestamp=result.timestamp
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to log authorization event: {str(e)}")

    @handle_exceptions
    async def grant_permission(
        self,
        user_id: str,
        permission_id: str,
        granted_by: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> None:
        """
        Grant a permission to a user.
        
        Args:
            user_id: User identifier
            permission_id: Permission identifier
            granted_by: Who granted the permission
            expires_at: When the permission expires
        """
        async with self.lock:
            if permission_id not in self.permissions:
                raise AuthorizationError(f"Permission {permission_id} not found")
            
            self.user_permissions[user_id].add(permission_id)
            
            # Store in database if available
            if self.database:
                await self._store_user_permission(user_id, permission_id, granted_by, expires_at)
            
            # Clear cache for this user
            await self._clear_user_cache(user_id)
            
            # Emit event
            await self.event_bus.emit(PermissionGranted(
                user_id=user_id,
                permission_id=permission_id,
                granted_by=granted_by,
                expires_at=expires_at
            ))
            
            self.logger.info(f"Granted permission {permission_id} to user {user_id}")

    @handle_exceptions
    async def revoke_permission(
        self,
        user_id: str,
        permission_id: str,
        revoked_by: Optional[str] = None
    ) -> None:
        """
        Revoke a permission from a user.
        
        Args:
            user_id: User identifier
            permission_id: Permission identifier
            revoked_by: Who revoked the permission
        """
        async with self.lock:
            self.user_permissions[user_id].discard(permission_id)
            
            # Remove from database if available
            if self.database:
                await self._remove_user_permission(user_id, permission_id)
            
            # Clear cache for this user
            await self._clear_user_cache(user_id)
            
            # Emit event
            await self.event_bus.emit(PermissionRevoked(
                user_id=user_id,
                permission_id=permission_id,
                revoked_by=revoked_by
            ))
            
            self.logger.info(f"Revoked permission {permission_id} from user {user_id}")

    @handle_exceptions
    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        assigned_by: Optional[str] = None
    ) -> None:
        """
        Assign a role to a user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            assigned_by: Who assigned the role
        """
        async with self.lock:
            if role_id not in self.roles:
                raise AuthorizationError(f"Role {role_id} not found")
            
            self.user_roles[user_id].add(role_id)
            
            # Store in database if available
            if self.database:
                await self._store_user_role(user_id, role_id, assigned_by)
            
            # Clear cache for this user
            await self._clear_user_cache(user_id)
            
            # Emit event
            await self.event_bus.emit(RoleAssigned(
                user_id=user_id,
                role_id=role_id,
                assigned_by=assigned_by
            ))
            
            self.logger.info(f"Assigned role {role_id} to user {user_id}")

    @handle_exceptions
    async def revoke_role(
        self,
        user_id: str,
        role_id: str,
        revoked_by: Optional[str] = None
    ) -> None:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User identifier
            role_id: Role identifier
            revoked_by: Who revoked the role
        """
        async with self.lock:
            self.user_roles[user_id].discard(role_id)
            
            # Remove from database if available
            if self.database:
                await self._remove_user_role(user_id, role_id)
            
            # Clear cache for this user
            await self._clear_user_cache(user_id)
            
            # Emit event
            await self.event_bus.emit(RoleRevoked(
                user_id=user_id,
                role_id=role_id,
                revoked_by=revoked_by
            ))
            
            self.logger.info(f"Revoked role {role_id} from user {user_id}")

    @handle_exceptions
    async def create_permission(self, permission: Permission) -> None:
        """
        Create a new permission.
        
        Args:
            permission: Permission to create
        """
        async with self.lock:
            if permission.permission_id in self.permissions:
                raise AuthorizationError(f"Permission {permission.permission_id} already exists")
            
            self.permissions[permission.permission_id] = permission
            
            # Store in database if available
            if self.database:
                await self._store_permission(permission)
            
            self.logger.info(f"Created permission: {permission.permission_id}")

    @handle_exceptions
    async def create_role(self, role: Role) -> None:
        """
        Create a new role.
        
        Args:
            role: Role to create
        """
        async with self.lock:
            if role.role_id in self.roles:
                raise AuthorizationError(f"Role {role.role_id} already exists")
            
            # Validate permissions exist
            for perm_id in role.permissions:
                if perm_id not in self.permissions:
                    raise AuthorizationError(f"Permission {perm_id} not found")
            
            self.roles[role.role_id] = role
            
            # Store in database if available
            if self.database:
                await self._store_role(role)
            
            self.logger.info(f"Created role: {role.role_id}")

    @handle_exceptions
    async def create_policy(self, policy: Policy) -> None:
        """
        Create a new authorization policy.
        
        Args:
            policy: Policy to create
        """
        async with self.lock:
            if policy.policy_id in self.policies:
                raise AuthorizationError(f"Policy {policy.policy_id} already exists")
            
            self.policies[policy.policy_id] = policy
            
            # Store in database if available
            if self.database:
                await self._store_policy(policy)
            
            # Clear all caches as policies affect all evaluations
            await self._clear_all_caches()
            
            # Emit event
            await self.event_bus.emit(PolicyCreated(
                policy_id=policy.policy_id,
                policy_name=policy.name,
                policy_type=policy.policy_type.value,
                created_by=policy.created_by
            ))
            
            self.logger.info(f"Created policy: {policy.policy_id}")

    @handle_exceptions
    async def register_resource(self, resource: Resource) -> None:
        """
        Register a resource for access control.
        
        Args:
            resource: Resource to register
        """
        async with self.lock:
            self.resources[resource.resource_id] = resource
            
            # Store in database if available
            if self.database:
                await self._store_resource(resource)
            
            self.logger.info(f"Registered resource: {resource.resource_id}")

    async def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user (direct + role-based)."""
        all_permissions = set(self.user_permissions.get(user_id, set()))
        
        # Add permissions from roles
        user_roles = self.user_roles.get(user_id, set())
        for role_id in user_roles:
            role = self.roles.get(role_id)
            if role:
                all_permissions.update(role.permissions)
        
        return all_permissions

    async def get_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles for a user."""
        return set(self.user_roles.get(user_id, set()))

    async def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self.roles.get(role_id)

    async def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID."""
        return self.permissions.get(permission_id)

    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get resource by ID."""
        return self.resources.get(resource_id)

    async def get_applicable_policies(self, context: AuthorizationContext) -> List[Policy]:
        """Get policies that apply to the given context."""
        applicable_policies = []
        
        for policy in self.policies.values():
            if not policy.is_active:
                continue
            
            # Check time constraints
            current_time = datetime.now(timezone.utc)
            if policy.effective_from and current_time < policy.effective_from:
                continue
            if policy.effective_until and current_time > policy.effective_until:
                continue
            
            # Check if policy applies to user/roles/resources
            applies = False
            
            if not policy.applies_to_users and not policy.applies_to_roles and not policy.applies_to_resources:
                applies = True  # Global policy
            else:
                if policy.applies_to_users and context.user_id in policy.applies_to_users:
                    applies = True
                elif policy.applies_to_roles and context.user_roles.intersection(policy.applies_to_roles):
                    applies = True
                elif policy.applies_to_resources and context.resource_id in policy.applies_to_resources:
                    applies = True
            
            if applies:
                applicable_policies.append(policy)
        
        return applicable_policies

    async def _clear_user_cache(self, user_id: str) -> None:
        """Clear cache entries for a specific user."""
        if self.cache:
            try:
                # Pattern match and delete user-specific cache entries
                pattern = f"auth:*{user_id}*"
                await self.cache.delete_pattern(pattern)
            except Exception as e:
                self.logger.warning(f"Failed to clear user cache: {str(e)}")
        else:
            # Clear in-memory cache entries containing this user
            keys_to_remove = [
                key for key in self.evaluation_cache.keys()
                if user_id in key
            ]
            for key in keys_to_remove:
                del self.evaluation_cache[key]

    async def _clear_all_caches(self) -> None:
        """Clear all authorization caches."""
        if self.cache:
            try:
                await self.cache.delete_pattern("auth:*")
            except Exception as e:
                self.logger.warning(f"Failed to clear all caches: {str(e)}")
        else:
            self.evaluation_cache.clear()

    # Database operations (placeholder implementations)
    async def _load_from_database(self) -> None:
        """Load authorization data from database."""
        try:
            # Load permissions, roles, policies, resources, and user associations
            # This would be implemented based on the actual database schema
            self.logger.info("Authorization data loaded from database")
        except Exception as e:
            self.logger.warning(f"Failed to load from database: {str(e)}")

    async def _store_permission(self, permission: Permission) -> None:
        """Store permission in database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to store permission: {str(e)}")

    async def _store_role(self, role: Role) -> None:
        """Store role in database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to store role: {str(e)}")

    async def _store_policy(self, policy: Policy) -> None:
        """Store policy in database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to store policy: {str(e)}")

    async def _store_resource(self, resource: Resource) -> None:
        """Store resource in database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to store resource: {str(e)}")

    async def _store_user_permission(self, user_id: str, permission_id: str, 
                                   granted_by: Optional[str], expires_at: Optional[datetime]) -> None:
        """Store user permission in database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to store user permission: {str(e)}")

    async def _store_user_role(self, user_id: str, role_id: str, assigned_by: Optional[str]) -> None:
        """Store user role in database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to store user role: {str(e)}")

    async def _remove_user_permission(self, user_id: str, permission_id: str) -> None:
        """Remove user permission from database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to remove user permission: {str(e)}")

    async def _remove_user_role(self, user_id: str, role_id: str) -> None:
        """Remove user role from database."""
        try:
            # Implementation would depend on database schema
            pass
        except Exception as e:
            self.logger.error(f"Failed to remove user role: {str(e)}")

    # Background tasks
    async def _cache_cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                # Clean up expired cache entries
                if not self.cache:  # In-memory cache cleanup
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, result in self.evaluation_cache.items():
                        if hasattr(result, 'timestamp'):
                            age = current_time - result.timestamp.timestamp()
                            if age > self.cache_ttl:
                                expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.evaluation_cache[key]
                    
                    if expired_keys:
                        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(300)

    async def _audit_processing_loop(self) -> None:
        """Background task for audit log processing."""
        while True:
            try:
                # Process audit log entries
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in audit log processing: {str(e)}")
                await asyncio.sleep(60)
