#!/usr/bin/env python3
"""
Enterprise Structure Validation Script

This script validates that the AI Assistant project meets enterprise-grade standards:
1. All required directories exist
2. No circular dependencies between layers
3. Clean architecture boundaries are maintained
4. All files follow naming conventions
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple


def check_directory_structure() -> List[str]:
    """Check if all required directories exist per the specification."""
    required_dirs = [
        "src/api/grpc/protos",
        "src/ui/web/templates",
        "src/ui/web/components",
        "data/models/checkpoints",
        "data/models/fine_tuned",
        "data/datasets/training",
        "data/datasets/validation", 
        "data/datasets/test",
        "data/cache/vector_cache",
        "data/cache/response_cache",
        "data/user_data/preferences",
        "data/user_data/history",
        "data/user_data/personalization",
        "data/knowledge_base/documents",
        "data/knowledge_base/embeddings",
        "data/knowledge_base/graphs",
        "data/logs/application",
        "data/logs/access",
        "data/logs/error",
        "data/logs/audit",
        "migrations/versions",
        "docs/api",
        "docs/architecture",
        "docs/deployment",
        "docs/development",
        "docs/user_guide",
        "docs/examples",
        "tests/e2e",
        "tests/smoke",
        "tests/performance",
        "tests/security",
        "tests/resilience",
        "tests/fixtures",
        "tools/code_generators",
        "tools/data_processors",
        "tools/model_converters",
        "tools/deployment_helpers",
        "infrastructure/terraform",
        "infrastructure/kubernetes",
        "infrastructure/helm",
        "infrastructure/ansible",
        "infrastructure/monitoring"
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing.append(dir_path)
    
    return missing


def check_file_naming() -> List[str]:
    """Check for correct file naming per specification."""
    issues = []
    
    # Check middleware log file
    log_file = "src/api/rest/middleware/log.py"
    if not os.path.exists(log_file):
        issues.append(f"Missing {log_file} (should be renamed from logging.py)")
    
    # Check for duplicate error_handling.py files
    error_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file == "error_handling.py":
                error_files.append(os.path.join(root, file))
    
    if len(error_files) > 1:
        issues.append(f"Duplicate error_handling.py files found: {error_files}")
    
    return issues


def extract_imports_from_file(file_path: str) -> List[str]:
    """Extract import statements from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to get imports
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception:
        return []


def check_circular_dependencies() -> List[str]:
    """Check for circular dependencies between architectural layers."""
    issues = []
    
    # Define layer hierarchy (lower layers should not import from higher)
    # Based on Clean Architecture principles
    layers = {
        'observability': 0,  # Infrastructure - logging, metrics, tracing
        'core': 1,          # Foundation - DI, config, error handling, security
        'integrations': 2,   # External integrations and utilities
        'processing': 2,     # Data processing utilities
        'reasoning': 3,      # Domain logic - reasoning and inference
        'skills': 3,        # Domain logic - skills and capabilities  
        'memory': 3,        # Domain logic - memory management
        'learning': 3,      # Domain logic - learning and adaptation
        'assistant': 4,     # Orchestration - main business logic
        'api': 5,           # Presentation - API endpoints
        'ui': 5,            # Presentation - user interfaces
    }
    
    # Check each Python file in src/
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, "src")
                
                # Determine current layer
                current_layer = None
                for layer in layers:
                    if relative_path.startswith(f"{layer}/"):
                        current_layer = layer
                        break
                
                if not current_layer:
                    continue
                
                # Check imports
                imports = extract_imports_from_file(file_path)
                for imp in imports:
                    if imp.startswith('src.'):
                        # Extract imported layer
                        parts = imp.split('.')
                        if len(parts) >= 2:
                            imported_layer = parts[1]
                            
                            if imported_layer in layers:
                                current_level = layers[current_layer]
                                imported_level = layers[imported_layer]
                                
                                # Check if lower layer imports from higher layer
                                if current_level < imported_level:
                                    issues.append(
                                        f"Circular dependency: {file_path} (layer {current_layer}, level {current_level}) "
                                        f"imports from {imp} (layer {imported_layer}, level {imported_level})"
                                    )
    
    return issues


def main():
    """Run all enterprise structure validations."""
    print("🏢 Enterprise AI Assistant Structure Validation")
    print("=" * 50)
    
    all_passed = True
    
    # 1. Check directory structure
    print("\n📁 Checking directory structure...")
    missing_dirs = check_directory_structure()
    if missing_dirs:
        print("❌ Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        all_passed = False
    else:
        print("✅ All required directories exist")
    
    # 2. Check file naming
    print("\n📝 Checking file naming conventions...")
    naming_issues = check_file_naming()
    if naming_issues:
        print("❌ File naming issues:")
        for issue in naming_issues:
            print(f"   - {issue}")
        all_passed = False
    else:
        print("✅ All files follow naming conventions")
    
    # 3. Check circular dependencies
    print("\n🔄 Checking for circular dependencies...")
    circular_deps = check_circular_dependencies()
    if circular_deps:
        print("❌ Circular dependencies found:")
        for dep in circular_deps:
            print(f"   - {dep}")
        all_passed = False
    else:
        print("✅ No circular dependencies detected")
    
    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ENTERPRISE-GRADE VALIDATION PASSED!")
        print("✅ The AI Assistant project structure meets enterprise standards")
        return 0
    else:
        print("⚠️  VALIDATION FAILED")
        print("❌ Please fix the issues above before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())