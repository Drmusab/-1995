#!/usr/bin/env python3
"""
Simple test for YAML configuration loading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_yaml_loading():
    """Test basic YAML configuration loading."""
    print("🔧 Testing YAML configuration loading...")
    
    try:
        from src.core.config.yaml_loader import YamlConfigLoader
        
        # Test development configuration
        print("  📖 Loading development configuration...")
        dev_loader = YamlConfigLoader("development")
        dev_config = dev_loader.load()
        
        print(f"  ✅ Loaded {len(dev_config)} top-level sections")
        print(f"  ✅ App name: {dev_config.get('app', {}).get('name')}")
        print(f"  ✅ Environment: {dev_config.get('app', {}).get('environment')}")
        
        # Test configuration access
        print("  🔍 Testing configuration access...")
        app_name = dev_loader.get("app.name")
        max_requests = dev_loader.get("core.engine.max_concurrent_requests")
        db_url = dev_loader.get("integrations.storage.database.url")
        
        print(f"  ✅ app.name: {app_name}")
        print(f"  ✅ core.engine.max_concurrent_requests: {max_requests}")
        print(f"  ✅ integrations.storage.database.url: {db_url}")
        
        # Test production configuration
        print("  📖 Loading production configuration...")
        prod_loader = YamlConfigLoader("production")
        prod_config = prod_loader.load()
        
        prod_env = prod_config.get('app', {}).get('environment')
        prod_requests = prod_loader.get("core.engine.max_concurrent_requests")
        
        print(f"  ✅ Production environment: {prod_env}")
        print(f"  ✅ Production max requests: {prod_requests}")
        
        # Verify overrides work
        if max_requests != prod_requests:
            print("  ✅ Environment-specific overrides are working!")
        else:
            print("  ⚠️  Environment overrides may not be working")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_existence():
    """Test that config files exist."""
    print("🔧 Testing configuration file existence...")
    
    config_files = [
        "config.yaml",
        "config.dev.yaml", 
        "config.prod.yaml",
        "config.testing.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✅ {config_file} exists")
        else:
            print(f"  ❌ {config_file} missing")
            return False
            
    return True

if __name__ == "__main__":
    print("🎯 YAML Configuration Consolidation Test\n")
    
    # Test file existence first
    if not test_file_existence():
        print("\n❌ Configuration files missing!")
        sys.exit(1)
    
    print()
    
    # Test YAML loading
    if not test_yaml_loading():
        print("\n❌ Configuration loading failed!")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ YAML configuration consolidation is working correctly")