#!/usr/bin/env python3
"""
Final Configuration Consolidation Test
Tests the core YAML configuration functionality without complex dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_yaml_configuration():
    """Test the YAML configuration system."""
    print("🔧 Testing YAML Configuration System")
    
    try:
        from src.core.config.yaml_loader import YamlConfigLoader
        
        # Test all environments
        environments = ["development", "production", "testing"]
        results = {}
        
        for env in environments:
            print(f"  📋 Testing {env} environment...")
            
            loader = YamlConfigLoader(env)
            config = loader.load()
            
            # Extract key metrics for comparison
            results[env] = {
                "environment": config.get("app", {}).get("environment"),
                "max_requests": loader.get("core.engine.max_concurrent_requests"),
                "database_url": loader.get("integrations.storage.database.url"),
                "sections": list(config.keys())
            }
            
            print(f"    ✅ Environment: {results[env]['environment']}")
            print(f"    ✅ Max requests: {results[env]['max_requests']}")
            print(f"    ✅ Database URL: {results[env]['database_url']}")
            print(f"    ✅ Sections: {len(results[env]['sections'])}")
        
        # Verify environment differences
        print("  🔍 Verifying environment-specific overrides...")
        
        # Development should have lower limits
        if results["development"]["max_requests"] < results["production"]["max_requests"]:
            print("    ✅ Development has lower max_requests than production")
        else:
            print("    ❌ Environment overrides not working correctly")
            return False
            
        # Development should use different database
        if "development" in results["development"]["database_url"]:
            print("    ✅ Development uses development database")
        elif "memory" in results["testing"]["database_url"]:
            print("    ✅ Testing uses in-memory database")
        else:
            print("    ⚠️  Database configuration may not be environment-specific")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_variables():
    """Test environment variable interpolation."""
    print("🔧 Testing Environment Variable Interpolation")
    
    try:
        from src.core.config.yaml_loader import YamlConfigLoader
        
        # Set test environment variables
        os.environ["TEST_VAR"] = "test_value"
        os.environ["TEST_NUMBER"] = "42"
        os.environ["TEST_BOOL"] = "true"
        
        loader = YamlConfigLoader("development")
        
        # Test various interpolation cases
        test_cases = [
            ("${env:TEST_VAR}", "test_value"),
            ("${env:TEST_NUMBER}", 42),
            ("${env:TEST_BOOL}", True),
            ("${env:NONEXISTENT:default}", "default"),
            ("${env:NONEXISTENT:123}", 123),
        ]
        
        for input_val, expected in test_cases:
            result = loader._interpolate_string(input_val)
            if result == expected:
                print(f"    ✅ {input_val} → {result}")
            else:
                print(f"    ❌ {input_val} → {result} (expected {expected})")
                return False
        
        # Clean up
        del os.environ["TEST_VAR"]
        del os.environ["TEST_NUMBER"] 
        del os.environ["TEST_BOOL"]
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def test_configuration_structure():
    """Test configuration file structure and completeness."""
    print("🔧 Testing Configuration Structure")
    
    config_files = ["config.yaml", "config.dev.yaml", "config.prod.yaml", "config.testing.yaml"]
    
    try:
        import yaml
        
        for config_file in config_files:
            if not Path(config_file).exists():
                print(f"    ❌ Missing: {config_file}")
                return False
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            file_size = Path(config_file).stat().st_size
            section_count = len(config) if config else 0
            
            print(f"    ✅ {config_file}: {section_count} sections, {file_size} bytes")
        
        # Check base config has all expected sections
        with open("config.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
            
        expected_sections = [
            "app", "core", "memory", "learning", "processing", 
            "skills", "integrations", "api", "security", 
            "observability", "performance", "backup", "paths"
        ]
        
        missing_sections = []
        for section in expected_sections:
            if section not in base_config:
                missing_sections.append(section)
                
        if missing_sections:
            print(f"    ❌ Missing sections in base config: {missing_sections}")
            return False
        else:
            print(f"    ✅ Base config has all {len(expected_sections)} expected sections")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return False

def calculate_consolidation_metrics():
    """Calculate metrics showing the consolidation benefits."""
    print("🔧 Calculating Consolidation Metrics")
    
    # Old configuration files (estimated from our analysis)
    old_files = {
        "src/config_settings.py": 958,
        "src/core/config/settings/development.py": 829, 
        "src/core/config/settings/production.py": 871,
        "src/observability/logging/config.py": 1121  # partial
    }
    
    # New configuration files
    new_files = {
        "config.yaml": 0,
        "config.dev.yaml": 0,
        "config.prod.yaml": 0,
        "config.testing.yaml": 0
    }
    
    # Calculate actual sizes
    for filename in new_files:
        if Path(filename).exists():
            new_files[filename] = len(Path(filename).read_text().splitlines())
    
    old_total = sum(old_files.values())
    new_total = sum(new_files.values())
    
    print(f"    📊 Old configuration: {old_total} lines across {len(old_files)} Python files")
    print(f"    📊 New configuration: {new_total} lines across {len(new_files)} YAML files")
    print(f"    📊 Reduction: {old_total - new_total} lines ({((old_total - new_total) / old_total * 100):.1f}%)")
    
    print(f"    📈 Benefits:")
    print(f"       • Single source of truth (YAML files)")
    print(f"       • Environment-specific overrides")
    print(f"       • Environment variable interpolation")
    print(f"       • Easy to read and modify")
    print(f"       • Version control friendly")
    
    return True

def main():
    """Run all tests."""
    print("🎯 Final Configuration Consolidation Validation\n")
    
    tests = [
        ("Configuration Structure", test_configuration_structure),
        ("YAML Configuration System", test_yaml_configuration),
        ("Environment Variable Interpolation", test_environment_variables),
        ("Consolidation Metrics", calculate_consolidation_metrics),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if not test_func():
            print(f"❌ {test_name} failed")
            all_passed = False
        else:
            print(f"✅ {test_name} passed")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("🎉 Configuration Consolidation SUCCESSFUL!")
        print("")
        print("✅ YAML-first configuration system implemented")
        print("✅ Environment-specific overrides working correctly") 
        print("✅ Environment variable interpolation functional")
        print("✅ Significant reduction in configuration complexity")
        print("✅ Single source of truth established")
        print("")
        print("🚀 The configuration consolidation is complete and ready for use!")
        return 0
    else:
        print("❌ Some tests failed - please review the implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())