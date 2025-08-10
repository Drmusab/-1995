#!/usr/bin/env python3
"""
Test script for the consolidated configuration system organization.
Author: Drmusab
Last Modified: 2025-01-13
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_consolidated_imports():
    """Test that all imports work from the consolidated settings module."""
    print("=== Testing Consolidated Settings Module ===")
    
    try:
        # Test consolidated imports
        from src.core.config.consolidated_settings import (
            get_unified_config,
            get_logger,
            initialize_config_system,
            get_all_config,
            validate_all_config,
        )
        print("✓ Successfully imported from consolidated settings module")
        
        # Test unified config
        config = get_unified_config("development")
        print(f"✓ Unified config working: {config.get('app.name')}")
        
        # Test logger
        logger = get_logger(__name__)
        logger.info("Testing consolidated logging")
        print("✓ Consolidated logging working")
        
        # Test initialization
        config_system = initialize_config_system("development")
        print("✓ Configuration system initialization working")
        
        # Test validation
        errors = validate_all_config()
        print(f"✓ Configuration validation working: {len(errors)} validation warnings")
        
        return True
        
    except Exception as e:
        print(f"✗ Consolidated settings test failed: {e}")
        return False


def test_organized_file_structure():
    """Test that the organized file structure works."""
    print("\n=== Testing Organized File Structure ===")
    
    try:
        # Check that files are in their organized locations
        config_dir = Path("src/core/config")
        
        # Main files
        assert (config_dir / "config.yaml").exists(), "Main config file not found"
        print("✓ Main config file found in organized location")
        
        # Environment files
        env_dir = config_dir / "environments"
        assert (env_dir / "config.dev.yaml").exists(), "Dev config file not found"
        assert (env_dir / "config.prod.yaml").exists(), "Prod config file not found"
        assert (env_dir / "config.testing.yaml").exists(), "Testing config file not found"
        print("✓ Environment config files found in organized location")
        
        # New organized modules
        assert (config_dir / "consolidated_settings.py").exists(), "Consolidated settings module not found"
        assert (config_dir / "logging_config.py").exists(), "Unified logging config not found"
        assert (config_dir / "README.md").exists(), "Documentation not found"
        print("✓ New organized modules found")
        
        # Test loading from organized structure with fresh instance
        from src.core.config.unified_config import UnifiedConfigManager
        prod_config = UnifiedConfigManager("production")
        
        # Verify production overrides work
        prod_max_requests = prod_config.get("core.engine.max_concurrent_requests")
        assert prod_max_requests == 100, f"Expected 100, got {prod_max_requests}"
        print("✓ Production environment overrides working")
        
        return True
        
    except Exception as e:
        print(f"✗ Organized structure test failed: {e}")
        return False


def test_new_logging_system():
    """Test the new unified logging system."""
    print("\n=== Testing Unified Logging System ===")
    
    try:
        from src.core.config.logging_config import get_logger, setup_logging
        
        # Setup logging
        setup_logging()
        print("✓ Global logging setup successful")
        
        # Get different loggers
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")
        
        # Test logging
        logger1.info("Test message from module 1")
        logger2.debug("Test debug message from module 2")
        print("✓ Multiple loggers working")
        
        # Test that loggers have proper configuration (development environment uses DEBUG level)
        expected_level = 10  # DEBUG level in development
        assert logger1.level == expected_level, f"Expected DEBUG level (10), got {logger1.level}"
        print("✓ Logger configuration from YAML working")
        
        return True
        
    except Exception as e:
        print(f"✗ Unified logging test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that all existing APIs still work."""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test legacy config_settings still works
        from src.core.config.config_settings import get_settings, Environment
        settings = get_settings("development")
        assert settings.app_name == "AI Assistant"
        print("✓ Legacy config_settings still working")
        
        # Test legacy di_config still works
        from src.core.config.di_config import ComponentConfiguration
        component_config = ComponentConfiguration()
        print("✓ Legacy di_config still working")
        
        # Test legacy performance_config still works  
        from src.core.config.performance_config import get_default_performance_config
        perf_config = get_default_performance_config()
        assert perf_config.monitoring_enabled == True
        print("✓ Legacy performance_config still working")
        
        # Test legacy logging still works
        from src.observability.logging.simple_config import get_logger as legacy_get_logger
        legacy_logger = legacy_get_logger(__name__)
        legacy_logger.info("Legacy logging test")
        print("✓ Legacy logging still working")
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing Consolidated Configuration Organization")
    print("=" * 50)
    
    tests = [
        test_consolidated_imports,
        test_organized_file_structure,
        test_new_logging_system,
        test_backward_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print()  # Add blank line after failed test
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for i in range(total):
        status = "✓ PASSED" if i < passed else "✗ FAILED"
        print(f"Test {i+1}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Configuration organization is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration organization.")
        return 1


if __name__ == "__main__":
    sys.exit(main())