"""
Test script to verify the unified configuration consolidation works correctly.
Author: Drmusab
Last Modified: 2025-01-13
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test imports
try:
    from src.core.config.unified_config import UnifiedConfigManager, get_unified_config
    from src.core.config.config_settings import BaseSettings, get_settings
    from src.core.config.di_config import ComponentConfiguration, create_configured_container
    from src.core.config.performance_config import get_default_performance_config
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


async def test_unified_config():
    """Test the unified configuration system."""
    print("\n=== Testing Unified Configuration System ===")
    
    try:
        # Test unified configuration manager
        print("\n1. Testing UnifiedConfigManager...")
        unified_config = get_unified_config("development")
        print(f"   ✓ Created unified config for environment: {unified_config.environment}")
        
        # Test basic configuration access
        app_config = unified_config.get_app_config()
        print(f"   ✓ App name: {app_config.get('name')}")
        
        core_config = unified_config.get_core_config()
        print(f"   ✓ Core engine config loaded: {len(core_config)} keys")
        
        # Test database configuration
        db_config = unified_config.get_database_config()
        print(f"   ✓ Database URL: {db_config['url']}")
        
        # Test cache configuration
        cache_config = unified_config.get_cache_config()
        print(f"   ✓ Cache enabled: {cache_config['enabled']}")
        
        # Test configuration validation
        errors = unified_config.validate_configuration()
        print(f"   ✓ Validation completed: {len(errors)} errors found")
        
        print("   ✓ UnifiedConfigManager working correctly")
        
    except Exception as e:
        print(f"   ✗ UnifiedConfigManager error: {e}")
        return False
    
    return True


async def test_backward_compatibility():
    """Test backward compatibility with legacy APIs."""
    print("\n=== Testing Backward Compatibility ===")
    
    try:
        # Test BaseSettings (legacy API)
        print("\n2. Testing BaseSettings compatibility...")
        settings = get_settings("development")
        print(f"   ✓ Created BaseSettings for environment: {settings.environment.value}")
        print(f"   ✓ App name: {settings.app_name}")
        print(f"   ✓ Database URL: {settings.database.url}")
        print(f"   ✓ Cache enabled: {settings.cache.enabled}")
        
        # Test configuration validation
        validation_errors = settings.validate_configuration()
        print(f"   ✓ Legacy validation: {len(validation_errors)} errors")
        
        print("   ✓ BaseSettings backward compatibility working")
        
        # Test ComponentConfiguration (legacy DI)
        print("\n3. Testing ComponentConfiguration compatibility...")
        component_config = ComponentConfiguration()
        print("   ✓ Created ComponentConfiguration")
        
        # Test performance config
        print("\n4. Testing PerformanceConfiguration compatibility...")
        perf_config = get_default_performance_config()
        print(f"   ✓ Performance monitoring enabled: {perf_config.monitoring_enabled}")
        print(f"   ✓ Thresholds configured: {perf_config.thresholds.response_time_warning}s")
        
        print("   ✓ All backward compatibility tests passed")
        
    except Exception as e:
        print(f"   ✗ Backward compatibility error: {e}")
        return False
    
    return True


async def test_dependency_injection():
    """Test dependency injection integration."""
    print("\n=== Testing Dependency Injection Integration ===")
    
    try:
        print("\n5. Testing DI container creation...")
        
        # Get unified configuration 
        unified_config = get_unified_config("development")
        await unified_config.initialize()
        
        container = unified_config.get_container()
        print(f"   ✓ DI container created")
        
        # Test component registrations
        registrations = unified_config.get_component_registrations()
        total_components = sum(len(components) for components in registrations.values())
        print(f"   ✓ Component registrations loaded: {total_components} components across {len(registrations)} categories")
        
        for category, components in registrations.items():
            print(f"      - {category}: {len(components)} components")
        
        print("   ✓ Dependency injection integration working")
        
    except Exception as e:
        print(f"   ✗ DI integration error: {e}")
        return False
    
    return True


async def test_yaml_configuration():
    """Test YAML configuration loading."""
    print("\n=== Testing YAML Configuration Loading ===")
    
    try:
        print("\n6. Testing YAML configuration access...")
        
        unified_config = get_unified_config("development") 
        
        # Test direct YAML access
        app_section = unified_config.get_section("app")
        print(f"   ✓ App section: {app_section.get('name')}")
        
        core_section = unified_config.get_section("core")
        print(f"   ✓ Core section has {len(core_section)} subsections")
        
        # Test nested access
        engine_config = unified_config.get("core.engine.max_concurrent_requests")
        print(f"   ✓ Nested config access: max_concurrent_requests = {engine_config}")
        
        # Test environment interpolation
        db_url = unified_config.get("integrations.storage.database.url")
        print(f"   ✓ Environment interpolation: DB URL = {db_url}")
        
        print("   ✓ YAML configuration loading working")
        
    except Exception as e:
        print(f"   ✗ YAML configuration error: {e}")
        return False
    
    return True


async def main():
    """Run all tests."""
    print("Starting Configuration Consolidation Tests...")
    
    # Set environment for testing
    os.environ["ENVIRONMENT"] = "development"
    
    tests = [
        test_unified_config(),
        test_backward_compatibility(),
        test_dependency_injection(),
        test_yaml_configuration(),
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    success_count = sum(1 for result in results if result is True)
    total_tests = len(results)
    
    for i, result in enumerate(results, 1):
        if result is True:
            print(f"Test {i}: ✓ PASSED")
        elif isinstance(result, Exception):
            print(f"Test {i}: ✗ FAILED - {result}")
        else:
            print(f"Test {i}: ✗ FAILED")
    
    print(f"\nOverall: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Configuration consolidation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)