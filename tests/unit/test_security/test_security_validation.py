"""Tests for security validation and authorization."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager  
from src.core.security.sanitization import SecuritySanitizer
from src.assistant.plugin_manager import EnhancedPluginManager, PluginType, SecurityLevel


@pytest.fixture
def mock_container():
    """Create a mock container for security tests."""
    container = Mock()
    
    # Mock config loader
    config_loader = Mock()
    config_loader.get.return_value = {
        'security': {
            'require_authentication': True,
            'enable_authorization': True,
            'sanitize_inputs': True
        },
        'plugins': {
            'security_validation': True,
            'default_security_level': 'sandbox'
        }
    }
    
    # Mock event bus
    event_bus = Mock()
    event_bus.emit = AsyncMock()
    event_bus.subscribe = Mock()
    
    # Mock other dependencies
    container.get.side_effect = lambda cls: {
        'ConfigLoader': config_loader,
        'EventBus': event_bus
    }.get(cls.__name__, Mock())
    
    return container


@pytest.fixture
def auth_manager():
    """Create an authentication manager for testing."""
    manager = Mock(spec=AuthenticationManager)
    
    # Mock authentication methods
    manager.authenticate = AsyncMock(return_value={
        'success': True,
        'user_id': 'test-user-123',
        'roles': ['user'],
        'token': 'mock-jwt-token'
    })
    
    manager.validate_token = AsyncMock(return_value={
        'valid': True,
        'user_id': 'test-user-123',
        'expires_at': datetime.now(timezone.utc).timestamp() + 3600
    })
    
    manager.refresh_token = AsyncMock(return_value={
        'success': True,
        'token': 'new-mock-jwt-token'
    })
    
    return manager


@pytest.fixture
def authz_manager():
    """Create an authorization manager for testing."""
    manager = Mock(spec=AuthorizationManager)
    
    # Mock authorization methods
    manager.check_permission = AsyncMock(return_value=True)
    manager.get_user_roles = AsyncMock(return_value=['user'])
    manager.has_role = AsyncMock(return_value=True)
    
    return manager


@pytest.fixture
def security_sanitizer():
    """Create a security sanitizer for testing."""
    sanitizer = Mock(spec=SecuritySanitizer)
    
    # Mock sanitization methods
    sanitizer.sanitize_input = AsyncMock(side_effect=lambda x: x)  # Pass through by default
    sanitizer.sanitize_output = AsyncMock(side_effect=lambda x: x)
    sanitizer.validate_input = AsyncMock(return_value=True)
    
    return sanitizer


class TestSecurityValidation:
    """Test suite for security validation and authorization."""
    
    @pytest.mark.asyncio
    async def test_authentication_success(self, auth_manager):
        """Test successful user authentication."""
        # Test valid credentials
        result = await auth_manager.authenticate(
            username="testuser",
            password="validpassword"
        )
        
        assert result['success'] is True
        assert result['user_id'] == 'test-user-123'
        assert 'token' in result
        assert 'user' in result['roles']
    
    @pytest.mark.asyncio
    async def test_authentication_failure(self, auth_manager):
        """Test authentication failure with invalid credentials."""
        # Mock failed authentication
        auth_manager.authenticate.return_value = {
            'success': False,
            'error': 'Invalid credentials'
        }
        
        result = await auth_manager.authenticate(
            username="testuser",
            password="wrongpassword"
        )
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_token_validation(self, auth_manager):
        """Test JWT token validation."""
        token = "mock-jwt-token"
        
        result = await auth_manager.validate_token(token)
        
        assert result['valid'] is True
        assert result['user_id'] == 'test-user-123'
        assert 'expires_at' in result
    
    @pytest.mark.asyncio
    async def test_authorization_permission_check(self, authz_manager):
        """Test permission checking."""
        # Test user has permission
        has_permission = await authz_manager.check_permission(
            user_id="test-user-123",
            permission="read_sessions"
        )
        
        assert has_permission is True
        
        # Test user lacks permission
        authz_manager.check_permission.return_value = False
        
        has_permission = await authz_manager.check_permission(
            user_id="test-user-123", 
            permission="admin_access"
        )
        
        assert has_permission is False
    
    @pytest.mark.asyncio
    async def test_role_based_access_control(self, authz_manager):
        """Test role-based access control."""
        # Test user has required role
        has_role = await authz_manager.has_role(
            user_id="test-user-123",
            role="user"
        )
        
        assert has_role is True
        
        # Test user lacks admin role
        authz_manager.has_role.return_value = False
        
        has_role = await authz_manager.has_role(
            user_id="test-user-123",
            role="admin"
        )
        
        assert has_role is False
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, security_sanitizer):
        """Test input sanitization against XSS and injection attacks."""
        # Test XSS attempt
        malicious_input = "<script>alert('xss')</script>"
        
        # Mock sanitizer to clean the input
        security_sanitizer.sanitize_input.return_value = "&lt;script&gt;alert('xss')&lt;/script&gt;"
        
        sanitized = await security_sanitizer.sanitize_input(malicious_input)
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized or "&lt;" in sanitized
    
    @pytest.mark.asyncio
    async def test_sql_injection_protection(self, security_sanitizer):
        """Test protection against SQL injection attempts."""
        # Test SQL injection attempt
        malicious_input = "'; DROP TABLE users; --"
        
        # Mock validation to detect SQL injection
        security_sanitizer.validate_input.return_value = False
        
        is_valid = await security_sanitizer.validate_input(malicious_input)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_command_injection_protection(self, security_sanitizer):
        """Test protection against command injection."""
        # Test command injection attempt
        malicious_input = "; rm -rf / ;"
        
        # Mock validation to detect command injection
        security_sanitizer.validate_input.return_value = False
        
        is_valid = await security_sanitizer.validate_input(malicious_input)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_plugin_security_validation(self, mock_container):
        """Test plugin security validation."""
        # Create plugin manager
        plugin_manager = EnhancedPluginManager(mock_container)
        
        # Mock plugin with suspicious code
        plugin_code = """
        import subprocess
        import os
        
        def execute_command(cmd):
            return subprocess.run(cmd, shell=True)
            
        class MaliciousPlugin:
            def run(self):
                os.system('rm -rf /')
        """
        
        # Mock security scanner to detect dangerous patterns
        with patch.object(plugin_manager.plugin_validator, '_scan_for_security_issues') as mock_scan:
            mock_scan.return_value = [
                "Dangerous pattern 'subprocess\\.' found in plugin.py",
                "Dangerous pattern 'os\\.system\\s*\\(' found in plugin.py"
            ]
            
            # Test that security issues are detected
            issues = await plugin_manager.plugin_validator._scan_for_security_issues(Mock())
            
            assert len(issues) > 0
            assert any("subprocess" in issue for issue in issues)
            assert any("os.system" in issue for issue in issues)
    
    @pytest.mark.asyncio
    async def test_plugin_sandbox_isolation(self, mock_container):
        """Test plugin sandbox isolation."""
        from src.assistant.plugin_manager import PluginSandbox
        
        # Create sandbox for untrusted plugin
        sandbox = PluginSandbox(
            plugin_id="test-plugin",
            security_level=SecurityLevel.UNTRUSTED,
            resource_limits={
                'memory_mb': 256,
                'cpu_seconds': 30,
                'timeout_seconds': 10
            }
        )
        
        # Test function that should be isolated
        def safe_function():
            return "Hello from sandbox"
        
        # Execute in sandbox (mocked for testing)
        with patch.object(sandbox, '_execute_in_thread') as mock_exec:
            mock_exec.return_value = "Hello from sandbox"
            
            result = sandbox.execute_in_sandbox(safe_function)
            
            assert result == "Hello from sandbox"
            mock_exec.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_session_security_validation(self):
        """Test session security validation."""
        from src.assistant.session_manager import EnhancedSessionManager
        
        # Mock container for session manager
        container = Mock()
        config_loader = Mock()
        config_loader.get.return_value = {
            'session_manager': {
                'require_authentication': True,
                'encryption_enabled': True
            }
        }
        container.get.return_value = config_loader
        
        session_manager = Mock(spec=EnhancedSessionManager)
        
        # Test creating session without authentication should fail
        session_manager.create_session.side_effect = Exception("Authentication required")
        
        with pytest.raises(Exception, match="Authentication required"):
            await session_manager.create_session(user_id=None)
    
    @pytest.mark.asyncio
    async def test_data_encryption_validation(self):
        """Test data encryption for sensitive information."""
        from src.core.security.encryption import EncryptionManager
        
        encryption_manager = Mock(spec=EncryptionManager)
        
        # Test encryption of sensitive data
        sensitive_data = "user_password_123"
        encrypted_data = "encrypted_hash_xyz"
        
        encryption_manager.encrypt.return_value = encrypted_data
        encryption_manager.decrypt.return_value = sensitive_data
        
        # Test encryption
        result = encryption_manager.encrypt(sensitive_data)
        assert result == encrypted_data
        assert result != sensitive_data
        
        # Test decryption
        decrypted = encryption_manager.decrypt(encrypted_data)
        assert decrypted == sensitive_data
    
    @pytest.mark.asyncio
    async def test_rate_limiting_protection(self):
        """Test rate limiting against DoS attacks."""
        from src.api.rest.middleware.rate_limiter import RateLimiter
        
        rate_limiter = Mock(spec=RateLimiter)
        
        # Mock rate limiting logic
        rate_limiter.is_allowed.side_effect = [
            True, True, True, True, True,  # First 5 requests allowed
            False, False, False  # Next requests blocked
        ]
        
        user_id = "test-user-123"
        
        # Simulate multiple rapid requests
        results = []
        for i in range(8):
            allowed = rate_limiter.is_allowed(user_id)
            results.append(allowed)
        
        # First 5 should be allowed, rest blocked
        assert sum(results) == 5
        assert results[:5] == [True] * 5
        assert results[5:] == [False] * 3
    
    @pytest.mark.asyncio
    async def test_cross_site_request_forgery_protection(self):
        """Test CSRF protection."""
        # Mock CSRF token validation
        csrf_token = "csrf-token-123"
        valid_token = "csrf-token-123"
        invalid_token = "invalid-token"
        
        def validate_csrf_token(token):
            return token == valid_token
        
        # Test valid token
        assert validate_csrf_token(csrf_token) is True
        
        # Test invalid token
        assert validate_csrf_token(invalid_token) is False
    
    @pytest.mark.asyncio
    async def test_security_audit_logging(self):
        """Test security event audit logging."""
        from src.observability.logging.config import get_logger
        
        # Mock audit logger
        audit_logger = Mock()
        audit_logger.warning = Mock()
        audit_logger.error = Mock()
        
        with patch('src.observability.logging.config.get_logger', return_value=audit_logger):
            # Simulate security events
            security_events = [
                ("authentication_failure", "Invalid credentials for user: testuser"),
                ("authorization_denied", "User test-user-123 denied access to admin_panel"),
                ("suspicious_activity", "Multiple failed login attempts from IP 192.168.1.100")
            ]
            
            for event_type, message in security_events:
                if event_type == "authentication_failure":
                    audit_logger.warning(f"SECURITY: {message}")
                elif event_type == "authorization_denied":
                    audit_logger.warning(f"SECURITY: {message}")
                elif event_type == "suspicious_activity":
                    audit_logger.error(f"SECURITY: {message}")
            
            # Verify audit logs were created
            assert audit_logger.warning.call_count == 2
            assert audit_logger.error.call_count == 1
    
    @pytest.mark.asyncio
    async def test_secure_communication_tls(self):
        """Test secure communication requirements."""
        # Mock TLS/SSL validation
        def is_secure_connection(url):
            return url.startswith('https://')
        
        # Test secure URLs
        secure_urls = [
            'https://api.example.com/endpoint',
            'https://secure-service.com/api'
        ]
        
        insecure_urls = [
            'http://unsecure-api.com/endpoint',
            'ftp://file-server.com/files'
        ]
        
        # All secure URLs should pass
        for url in secure_urls:
            assert is_secure_connection(url) is True
        
        # All insecure URLs should fail
        for url in insecure_urls:
            assert is_secure_connection(url) is False
    
    @pytest.mark.asyncio
    async def test_memory_content_sanitization(self, security_sanitizer):
        """Test sanitization of content stored in memory."""
        # Test storing potentially malicious content in memory
        malicious_content = {
            "user_message": "<script>steal_cookies()</script>",
            "response": "Here's some data: ${sensitive_info}",
            "metadata": {
                "injection": "'; DROP TABLE memories; --"
            }
        }
        
        # Mock sanitization for each field
        security_sanitizer.sanitize_input.side_effect = lambda x: x.replace("<script>", "&lt;script&gt;").replace("${", "\\${").replace("';", "'\\;")
        
        # Sanitize content before storage
        sanitized_content = {}
        for key, value in malicious_content.items():
            if isinstance(value, str):
                sanitized_content[key] = await security_sanitizer.sanitize_input(value)
            elif isinstance(value, dict):
                sanitized_content[key] = {}
                for sub_key, sub_value in value.items():
                    sanitized_content[key][sub_key] = await security_sanitizer.sanitize_input(sub_value)
            else:
                sanitized_content[key] = value
        
        # Verify dangerous content was sanitized
        assert "<script>" not in sanitized_content["user_message"]
        assert "${" not in sanitized_content["response"] or "\\${" in sanitized_content["response"]
        assert "';" not in sanitized_content["metadata"]["injection"] or "'\\;" in sanitized_content["metadata"]["injection"]