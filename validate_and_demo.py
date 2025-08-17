#!/usr/bin/env python3
"""
AI Assistant - System Validation and Demo Script

This script validates that the AI Assistant system is properly installed
and demonstrates basic functionality. Use this to verify your installation
and see the system in action.

Usage:
    python validate_and_demo.py
    python validate_and_demo.py --full-demo
"""

import os
import sys
import asyncio
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n[{step}] {description}")

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")

def validate_python_version():
    """Validate Python version."""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print_error(f"Python 3.10+ required. Current: {version.major}.{version.minor}")
        return False
    print_success(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def validate_project_structure():
    """Validate project directory structure."""
    required_dirs = [
        "src",
        "src/core",
        "src/assistant", 
        "src/api",
        "docs",
        "configs"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print_error(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print_success("Project structure validated")
    return True

def validate_core_imports():
    """Validate that core modules can be imported."""
    try:
        # Test basic imports
        import src
        print_success("Source package importable")
        
        from src.core.config.loader import ConfigLoader
        print_success("Configuration loader importable")
        
        from src.core.dependency_injection import Container
        print_success("Dependency injection importable")
        
        from src.core.events.event_bus import EventBus
        print_success("Event bus importable")
        
        return True
        
    except ImportError as e:
        print_error(f"Import failed: {e}")
        print_warning("Try: pip install -e .")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False

def validate_dependencies():
    """Validate that required dependencies are available."""
    required_packages = [
        ("yaml", "pyyaml", "YAML processing"),
        ("pydantic", "pydantic", "Data validation"),
        ("numpy", "numpy", "Numerical computing"),
        ("fastapi", "fastapi", "Web framework"),
        ("uvicorn", "uvicorn", "ASGI server"),
    ]
    
    missing_packages = []
    for import_name, package_name, description in required_packages:
        try:
            __import__(import_name)
            print_success(f"{package_name}: {description}")
        except ImportError:
            missing_packages.append(package_name)
            print_error(f"{package_name}: Missing - {description}")
    
    if missing_packages:
        print_warning(f"Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def validate_optional_dependencies():
    """Check optional dependencies."""
    optional_packages = [
        ("torch", "torch", "AI/ML framework"),
        ("redis", "redis", "Caching"),
        ("sqlalchemy", "sqlalchemy", "Database ORM"),
        ("aiohttp", "aiohttp", "Async HTTP"),
    ]
    
    for import_name, package_name, description in optional_packages:
        try:
            __import__(import_name)
            print_success(f"{package_name}: {description}")
        except ImportError:
            print_warning(f"{package_name}: Optional - {description}")

def setup_environment():
    """Set up basic environment."""
    try:
        # Create data directories
        data_dirs = [
            "data/logs",
            "data/models", 
            "data/cache",
            "data/sessions"
        ]
        
        for dir_path in data_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print_success("Data directories created")
        
        # Create config directories
        config_dirs = [
            "configs/environments",
            "configs/models",
            "configs/skills"
        ]
        
        for dir_path in config_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print_success("Config directories created")
        
        # Set up environment file if it doesn't exist
        if not Path(".env").exists():
            if Path(".env.example").exists():
                import shutil
                shutil.copy(".env.example", ".env")
                print_success("Environment file created from template")
            else:
                # Create basic .env
                with open(".env", "w") as f:
                    f.write("AI_ASSISTANT_ENV=development\n")
                    f.write("AI_ASSISTANT_DEBUG=true\n")
                    f.write("AI_ASSISTANT_LOG_LEVEL=INFO\n")
                print_success("Basic environment file created")
        else:
            print_success("Environment file already exists")
        
        return True
        
    except Exception as e:
        print_error(f"Environment setup failed: {e}")
        return False

def test_configuration_system():
    """Test the configuration system."""
    try:
        # Set environment
        os.environ["AI_ASSISTANT_ENV"] = "development"
        
        from src.core.config.loader import ConfigLoader
        config = ConfigLoader()
        
        print_success("Configuration system works")
        return True
        
    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False

def test_dependency_injection():
    """Test dependency injection system."""
    try:
        from src.core.dependency_injection import Container
        
        container = Container()
        
        # Test registration and retrieval
        container.register(str, lambda: "test")
        value = container.get(str)
        
        if value == "test":
            print_success("Dependency injection works")
            return True
        else:
            print_error("Dependency injection test failed")
            return False
            
    except Exception as e:
        print_error(f"Dependency injection test failed: {e}")
        return False

def test_event_system():
    """Test event system."""
    try:
        from src.core.events.event_bus import EventBus
        
        event_bus = EventBus()
        
        # Simple test
        print_success("Event system importable")
        return True
        
    except Exception as e:
        print_error(f"Event system test failed: {e}")
        return False

async def test_async_components():
    """Test async components."""
    try:
        from src.core.events.event_bus import EventBus
        
        event_bus = EventBus()
        await event_bus.initialize()
        
        print_success("Async components work")
        return True
        
    except Exception as e:
        print_error(f"Async test failed: {e}")
        return False

def demo_basic_usage():
    """Demonstrate basic system usage."""
    print_header("BASIC USAGE DEMO")
    
    try:
        # Import core components
        from src.core.config.loader import ConfigLoader
        from src.core.dependency_injection import Container
        
        print_step("1", "Creating configuration loader")
        config = ConfigLoader()
        print_success("Configuration loader created")
        
        print_step("2", "Creating dependency container")
        container = Container()
        print_success("Container created")
        
        print_step("3", "Registering sample service")
        container.register(str, lambda: "AI Assistant Demo")
        service = container.get(str)
        print_success(f"Retrieved service: {service}")
        
        return True
        
    except Exception as e:
        print_error(f"Demo failed: {e}")
        traceback.print_exc()
        return False

async def demo_advanced_features():
    """Demonstrate advanced features."""
    print_header("ADVANCED FEATURES DEMO")
    
    try:
        print_step("1", "Testing async event system")
        from src.core.events.event_bus import EventBus
        
        event_bus = EventBus()
        await event_bus.initialize()
        print_success("Event bus initialized")
        
        print_step("2", "Testing component integration")
        from src.core.dependency_injection import Container
        container = Container()
        container.register(EventBus, lambda: event_bus)
        
        retrieved_bus = container.get(EventBus)
        print_success("Component integration works")
        
        return True
        
    except Exception as e:
        print_error(f"Advanced demo failed: {e}")
        traceback.print_exc()
        return False

def print_next_steps():
    """Print next steps for the user."""
    print_header("NEXT STEPS")
    
    print("""
🚀 Your AI Assistant system is ready! Here's what you can do next:

1. Start the interactive assistant:
   PYTHONPATH=/path/to/-1995 python -m src.cli

2. Start the web server:
   PYTHONPATH=/path/to/-1995 python -m src.main

3. Configure API keys (optional):
   Edit .env file and add:
   OPENAI_API_KEY=your_key_here

4. Install optional features:
   pip install asyncpg psycopg2-binary  # Database
   pip install pyaudio speechrecognition  # Speech
   pip install opencv-python pillow  # Vision

5. Read the documentation:
   - docs/QUICK_START.md
   - docs/COMPREHENSIVE_BUILD_AND_USAGE_GUIDE.md
   - docs/TROUBLESHOOTING_ACTIVATION_GUIDE.md

6. Check system health:
   curl http://localhost:8000/health  (after starting server)

📚 Documentation: docs/
🐛 Issues: https://github.com/Drmusab/-1995/issues
💬 Discussions: https://github.com/Drmusab/-1995/discussions
""")

def main():
    """Main validation and demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Assistant System Validator and Demo")
    parser.add_argument("--full-demo", action="store_true", 
                       help="Run full demo including advanced features")
    args = parser.parse_args()
    
    print_header("AI ASSISTANT SYSTEM VALIDATOR")
    print("This script validates your installation and demonstrates basic functionality.")
    
    # Validation steps
    validation_steps = [
        ("Python Version", validate_python_version),
        ("Project Structure", validate_project_structure),
        ("Core Imports", validate_core_imports),
        ("Required Dependencies", validate_dependencies),
        ("Environment Setup", setup_environment),
        ("Configuration System", test_configuration_system),
        ("Dependency Injection", test_dependency_injection),
        ("Event System", test_event_system),
    ]
    
    print_header("SYSTEM VALIDATION")
    
    all_passed = True
    for step_name, step_func in validation_steps:
        print_step("✓", f"Validating {step_name}")
        if not step_func():
            all_passed = False
            break
    
    if not all_passed:
        print_error("Validation failed. Please check the errors above.")
        print_warning("See docs/TROUBLESHOOTING_ACTIVATION_GUIDE.md for help.")
        return 1
    
    print_success("All validation steps passed!")
    
    # Optional dependencies check
    print_step("✓", "Checking Optional Dependencies")
    validate_optional_dependencies()
    
    # Async validation
    print_step("✓", "Testing Async Components")
    if not asyncio.run(test_async_components()):
        print_warning("Async components have issues")
    
    # Basic demo
    if not demo_basic_usage():
        print_error("Basic demo failed")
        return 1
    
    # Advanced demo if requested
    if args.full_demo:
        if not asyncio.run(demo_advanced_features()):
            print_error("Advanced demo failed")
            return 1
    
    print_next_steps()
    
    print_header("VALIDATION COMPLETE")
    print_success("🎉 AI Assistant system is ready to use!")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)