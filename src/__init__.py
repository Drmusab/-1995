"""
AI Assistant - Advanced AI Assistant with multimodal processing, workflow orchestration, and extensible plugin system.

This package provides a comprehensive AI assistant framework with the following core capabilities:
- Multimodal processing (text, speech, vision)
- Advanced reasoning and decision making
- Extensible skill system
- Memory management and learning
- Workflow orchestration
- Real-time communication via WebSocket, REST, and GraphQL APIs
- Security and authentication
- Monitoring and observability

The package is organized into the following main modules:
- assistant: Core assistant functionality and orchestration
- core: Core system components (config, events, security, DI)
- integrations: External service integrations (LLM, cache, storage)
- processing: Data processing modules (NLP, speech, vision, multimodal)
- reasoning: Advanced reasoning capabilities and planning
- skills: Skill management and execution
- memory: Memory management and operations
- learning: Adaptive learning and feedback processing
- api: API interfaces (REST, GraphQL, WebSocket, gRPC)
- ui: User interface components
- observability: Monitoring, logging, and profiling
"""

__version__ = "1.0.0"
__author__ = "Drmusab"
__email__ = "drmusab@example.com"
__license__ = "MIT"

"""
AI Assistant - Advanced AI Assistant with multimodal processing, workflow orchestration, and extensible plugin system.

This package provides a comprehensive AI assistant framework with the following core capabilities:
- Multimodal processing (text, speech, vision)
- Advanced reasoning and decision making
- Extensible skill system
- Memory management and learning
- Workflow orchestration
- Real-time communication via WebSocket, REST, and GraphQL APIs
- Security and authentication
- Monitoring and observability

The package is organized into the following main modules:
- assistant: Core assistant functionality and orchestration
- core: Core system components (config, events, security, DI)
- integrations: External service integrations (LLM, cache, storage)
- processing: Data processing modules (NLP, speech, vision, multimodal)
- reasoning: Advanced reasoning capabilities and planning
- skills: Skill management and execution
- memory: Memory management and operations
- learning: Adaptive learning and feedback processing
- api: API interfaces (REST, GraphQL, WebSocket, gRPC)
- ui: User interface components
- observability: Monitoring, logging, and profiling
"""

__version__ = "1.0.0"
__author__ = "Drmusab"
__email__ = "drmusab@example.com"
__license__ = "MIT"

# Lazy imports to avoid dependency issues during initial import
# Submodules can be imported on-demand when needed

def get_assistant():
    """Get the assistant module."""
    from . import assistant
    return assistant

def get_core():
    """Get the core module.""" 
    from . import core
    return core

def get_integrations():
    """Get the integrations module."""
    from . import integrations
    return integrations

def get_processing():
    """Get the processing module."""
    from . import processing
    return processing

def get_reasoning():
    """Get the reasoning module."""
    from . import reasoning
    return reasoning

def get_skills():
    """Get the skills module."""
    from . import skills
    return skills

def get_memory():
    """Get the memory module."""
    from . import memory
    return memory

def get_learning():
    """Get the learning module."""
    from . import learning
    return learning

def get_api():
    """Get the API module."""
    from . import api
    return api

def get_ui():
    """Get the UI module."""
    from . import ui
    return ui

def get_observability():
    """Get the observability module."""
    from . import observability
    return observability

# Core module availability check
_available_modules = []

def check_module_availability():
    """Check which modules are available without dependency issues."""
    global _available_modules
    if _available_modules:
        return _available_modules
    
    modules_to_check = [
        'api', 'assistant', 'core', 'integrations', 'processing',
        'reasoning', 'skills', 'memory', 'learning', 'ui', 'observability'
    ]
    
    for module_name in modules_to_check:
        try:
            __import__(f'src.{module_name}')
            _available_modules.append(module_name)
        except ImportError:
            pass  # Module not available due to dependencies
    
    return _available_modules

# Export core functionality
__all__ = [
    "get_assistant",
    "get_core", 
    "get_integrations",
    "get_processing", 
    "get_reasoning",
    "get_skills",
    "get_memory",
    "get_learning",
    "get_api",
    "get_ui",
    "get_observability",
    "check_module_availability",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]