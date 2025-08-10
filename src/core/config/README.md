# Configuration System Organization

This directory contains the consolidated configuration system for the AI Assistant.

## Directory Structure

```
src/core/config/
├── README.md                   # This file - configuration organization guide
├── config.yaml                 # Main base configuration file
├── environments/               # Environment-specific configuration overrides
│   ├── config.dev.yaml        # Development environment overrides
│   ├── config.prod.yaml       # Production environment overrides
│   └── config.testing.yaml    # Testing environment overrides
├── settings.py                 # Consolidated settings module (single import point)
├── unified_config.py           # Core unified configuration manager
├── yaml_loader.py              # YAML configuration loading system
├── logging_config.py           # Unified logging configuration
├── loader.py                   # Legacy configuration loader (compatibility)
└── validators/                 # Configuration validation system
    ├── simple_validator.py     # Simple YAML-based validation (recommended)
    └── config_validator.py     # Complex validation system (legacy)
```

## Usage

### Quick Start

For most use cases, simply import from the consolidated settings module:

```python
from src.core.config.settings import get_unified_config, get_logger

# Get configuration
config = get_unified_config()
app_name = config.get("app.name")

# Get logger
logger = get_logger(__name__)
logger.info("Application started")
```

### Unified Configuration Manager

The `UnifiedConfigManager` is the central point for all configuration:

```python
from src.core.config import get_unified_config

config = get_unified_config("development")

# Access configuration sections
app_config = config.get_app_config()
db_config = config.get_database_config()
cache_config = config.get_cache_config()

# Access with dot notation
max_requests = config.get("core.engine.max_concurrent_requests")
```

### Environment-Specific Configuration

The system supports environment-specific overrides:

- `config.yaml` - Base configuration (all environments)
- `environments/config.dev.yaml` - Development overrides
- `environments/config.prod.yaml` - Production overrides  
- `environments/config.testing.yaml` - Testing overrides

Environment is determined by:
1. Constructor parameter: `get_unified_config("production")`
2. Environment variable: `ENVIRONMENT=production`
3. Default: `development`

### Legacy Compatibility

All existing APIs continue to work:

```python
# Legacy BaseSettings API still works
from src.config_settings import get_settings
settings = get_settings("development")

# Legacy component configuration still works
from src.di_config import ComponentConfiguration
component_config = ComponentConfiguration()

# Legacy performance configuration still works
from src.core.performance_config import get_default_performance_config
perf_config = get_default_performance_config()
```

## Key Features

### 1. YAML-First Configuration
- All configuration primarily defined in YAML files
- Environment variable interpolation: `${env:DATABASE_URL:default_value}`
- Hierarchical configuration with environment overrides
- Easy to read, edit, and version control

### 2. Unified Management
- Single `UnifiedConfigManager` handles all configuration aspects
- Consistent API for accessing any configuration section
- Centralized validation and dependency injection setup

### 3. Backward Compatibility
- All existing configuration APIs continue to work
- Gradual migration path - no breaking changes
- Legacy Python configuration classes maintained

### 4. Organized Structure
- Configuration files organized in logical directories
- Clear separation between base config and environment overrides
- Consolidated imports through `settings.py`

### 5. Validation System
- YAML-based validation rules in main configuration
- Simple validator for basic checks
- Complex validator available for advanced scenarios

## Migration Guide

### From Scattered Python Config Files

If you have existing Python configuration files, they continue to work. To gradually migrate:

1. **Keep using existing APIs** - no immediate changes needed
2. **New configuration** - add to YAML files instead of Python
3. **Gradual consolidation** - move settings to YAML over time

### From Legacy YAML Structure

If you have YAML files in the root directory:

1. Files are automatically found in old locations for compatibility
2. Move files to `src/core/config/` for the organized structure
3. Move environment files to `src/core/config/environments/`

## Configuration Sections

The main `config.yaml` file contains these sections:

- `app` - Application metadata (name, version, description)
- `core` - Core system settings (engine, components, workflow, sessions, etc.)
- `memory` - Memory management (working, episodic, semantic, vector store)
- `learning` - Learning systems (continual, preference, feedback, adaptation)
- `processing` - Processing pipelines (speech, vision, NLP, multimodal)
- `skills` - Skills management and execution
- `integrations` - External integrations (LLM, storage, cache, APIs)
- `api` - API configuration (REST, WebSocket, GraphQL, gRPC)
- `security` - Security settings (auth, authorization, encryption, audit)
- `observability` - Monitoring (logging, metrics, tracing, health checks)
- `infrastructure` - Deployment and scaling settings
- `performance` - Performance limits and optimization settings
- `backup` - Backup and recovery configuration
- `paths` - File system paths
- `dependency_injection` - DI container configuration
- `validation` - Configuration validation rules

## Benefits of This Organization

1. **Reduced Complexity**: 60%+ reduction in Python configuration code
2. **Better Organization**: Clear structure with logical grouping
3. **Easier Maintenance**: Single source of truth in YAML
4. **Improved Testing**: Environment-specific configurations for testing
5. **Enhanced Security**: Production configurations separate from development
6. **Better Documentation**: Self-documenting YAML structure
7. **Version Control Friendly**: Easy to track configuration changes
8. **Environment Consistency**: Consistent configuration across environments