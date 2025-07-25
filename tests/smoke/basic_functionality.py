"""Basic functionality smoke tests for the AI Assistant."""

import pytest


@pytest.mark.smoke
def test_basic_import():
    """Test that basic modules can be imported."""
    try:
        import src
        import src.main
        import src.cli
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import basic modules: {e}")


@pytest.mark.smoke
def test_core_components_available():
    """Test that core components are available."""
    try:
        from src.core import dependency_injection
        from src.core import error_handling
        from src.core.config import loader
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core components: {e}")


@pytest.mark.smoke
def test_assistant_components_available():
    """Test that assistant components are available."""
    try:
        from src.assistant import core_engine
        from src.assistant import component_manager
        from src.assistant import session_manager
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import assistant components: {e}")