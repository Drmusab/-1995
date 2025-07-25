"""Authentication tests for security validation."""

import pytest


@pytest.mark.security
def test_authentication_module_exists():
    """Test that authentication module exists."""
    try:
        from src.core.security import authentication
        assert authentication is not None
    except ImportError as e:
        pytest.fail(f"Authentication module not found: {e}")


@pytest.mark.security
def test_authorization_module_exists():
    """Test that authorization module exists."""
    try:
        from src.core.security import authorization
        assert authorization is not None
    except ImportError as e:
        pytest.fail(f"Authorization module not found: {e}")


@pytest.mark.security
def test_encryption_module_exists():
    """Test that encryption module exists."""
    try:
        from src.core.security import encryption
        assert encryption is not None
    except ImportError as e:
        pytest.fail(f"Encryption module not found: {e}")