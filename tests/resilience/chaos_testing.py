"""Chaos engineering tests for system resilience."""

import pytest


@pytest.mark.resilience
def test_system_resilience_components():
    """Test that system resilience components are in place."""
    try:
        from src.observability.monitoring import alerting
        from src.observability.monitoring import metrics
        assert alerting is not None
        assert metrics is not None
    except ImportError as e:
        pytest.fail(f"Resilience monitoring components not found: {e}")