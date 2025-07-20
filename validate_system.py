#!/usr/bin/env python3
"""
Basic system validation test without external dependencies.
Tests that all core components can be imported and basic syntax is correct.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_imports():
    """Test core module imports."""
    print("Testing core module imports...")
    
    # Test basic imports that don't require external dependencies
    try:
        from src.core.dependency_injection import Container, LifecycleScope
        print("✓ Dependency injection imports")
    except Exception as e:
        print(f"✗ Dependency injection: {e}")
        return False
    
    try:
        from src.core.events.event_types import SystemStarted, ErrorOccurred
        print("✓ Event types imports")
    except Exception as e:
        print(f"✗ Event types: {e}")
        return False
    
    try:
        from src.core.error_handling import ErrorHandler
        print("✓ Error handling imports")
    except Exception as e:
        print(f"✗ Error handling: {e}")
        return False
    
    return True

def test_assistant_imports():
    """Test assistant module imports."""
    print("Testing assistant module imports...")
    
    try:
        # Import modules without instantiating (to avoid dependency issues)
        import src.assistant.core_engine
        import src.assistant.component_manager  
        import src.assistant.session_manager
        print("✓ Assistant module imports")
    except Exception as e:
        print(f"✗ Assistant modules: {e}")
        return False
    
    return True

def test_api_setup_imports():
    """Test API setup function imports."""
    print("Testing API setup imports...")
    
    try:
        from src.api.rest import setup_rest_api
        print("✓ REST API setup import")
    except Exception as e:
        print(f"✗ REST API setup: {e}")
        return False
    
    try:
        from src.api.websocket import setup_websocket_api
        print("✓ WebSocket API setup import")
    except Exception as e:
        print(f"✗ WebSocket API setup: {e}")
        return False
    
    try:
        from src.api.graphql import setup_graphql_api
        print("✓ GraphQL API setup import")
    except Exception as e:
        print(f"✗ GraphQL API setup: {e}")
        return False
    
    return True

def test_syntax_compilation():
    """Test that all Python files compile without syntax errors."""
    print("Testing Python file compilation...")
    
    src_path = Path(__file__).parent / "src"
    python_files = list(src_path.rglob("*.py"))
    
    syntax_errors = []
    for py_file in python_files:
        try:
            compile(py_file.read_text(), str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{py_file}: {e}")
        except Exception as e:
            # Other compilation errors (imports, etc.) are expected without dependencies
            pass
    
    if syntax_errors:
        print(f"✗ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        return False
    else:
        print(f"✓ All {len(python_files)} Python files compile successfully")
        return True

def main():
    """Run all validation tests."""
    print("=== AI Assistant System Validation ===\n")
    
    tests = [
        test_syntax_compilation,
        test_core_imports,
        test_assistant_imports, 
        test_api_setup_imports,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}\n")
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🟢 All validation tests PASSED - System is ready!")
        print("Next steps:")
        print("1. Install dependencies: pip install -e .")
        print("2. Run the assistant: python -m src.cli")
        return True
    else:
        print("🔴 Some validation tests FAILED - See errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)