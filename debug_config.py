#!/usr/bin/env python3
"""
Debug YAML configuration merging.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.config.yaml_loader import YamlConfigLoader

def debug_config_merging():
    """Debug configuration merging."""
    
    # Test base config loading
    print("🔧 Loading base configuration...")
    loader = YamlConfigLoader("production")
    
    # Load base config manually
    base_config = loader._load_yaml_file("config.yaml")
    print(f"Base max_concurrent_requests: {base_config.get('core', {}).get('engine', {}).get('max_concurrent_requests')}")
    
    # Load prod config manually  
    prod_config = loader._load_yaml_file("config.production.yaml")
    if prod_config:
        print(f"Prod max_concurrent_requests: {prod_config.get('core', {}).get('engine', {}).get('max_concurrent_requests')}")
    else:
        print("❌ Production config not found")
        
        # Try with different name
        prod_config = loader._load_yaml_file("config.prod.yaml")
        if prod_config:
            print(f"Prod max_concurrent_requests (from config.prod.yaml): {prod_config.get('core', {}).get('engine', {}).get('max_concurrent_requests')}")
    
    # Merge and check
    if prod_config:
        merged = loader._deep_merge(base_config, prod_config)
        print(f"Merged max_concurrent_requests: {merged.get('core', {}).get('engine', {}).get('max_concurrent_requests')}")
    
    # Load full config
    full_config = loader.load()
    print(f"Final max_concurrent_requests: {full_config.get('core', {}).get('engine', {}).get('max_concurrent_requests')}")

if __name__ == "__main__":
    debug_config_merging()