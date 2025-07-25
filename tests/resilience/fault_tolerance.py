"""Fault tolerance tests for resilience validation."""

import pytest


@pytest.mark.resilience
def test_error_handling_available():
    """Test that error handling components are available."""
    try:
        from src.core import error_handling
        assert error_handling is not None
    except ImportError as e:
        pytest.fail(f"Error handling module not found: {e}")


@pytest.mark.resilience  
def test_health_check_available():
    """Test that health check functionality is available."""
    try:
        from src.core import health_check
        assert health_check is not None
    except ImportError as e:
        pytest.fail(f"Health check module not found: {e}")