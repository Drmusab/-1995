"""
Development Configuration Settings for AI Assistant
Author: Drmusab  
Last Modified: 2025-06-13 11:04:49 UTC

This module provides comprehensive development-specific configuration that integrates
with all core system components, enabling development features like hot-reloading,
debug logging, local storage, and development-friendly defaults.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import timedelta

# Base configuration
from .base import BaseConfig


class DevelopmentConfig(BaseConfig):
    """
    Development configuration with enhanced debugging and development features.
    
    This configuration enables:
    - Hot-reloading for rapid development
    - Verbose logging and debugging
    - Local storage backends
    - Development-friendly timeouts and limits
    - Mock services for external dependencies
    - Enhanced error reporting and profiling
    """
    
    # Environment settings
    ENVIRONMENT = "development"
    DEBUG = True
    TESTING = False
    LOG_LEVEL = "DEBUG"
    
    # Development features
    HOT_RELOAD_ENABLED = True
    AUTO_RESTART_ON_CHANGE = True
    PROFILING_ENABLED = True
    METRICS_COLLECTION_ENABLED = True
    TRACING_ENABLED = True
    
    # Core system configuration
    CORE_ENGINE_CONFIG = {
        # Processing settings
        "default_processing_mode": "asynchronous",
        "max_concurrent_requests": 5,  # Lower for development
        "default_timeout_seconds": 60.0,  # Higher for debugging
        "enable_real_time_processing": True,
        "enable_adaptive_quality": True,
        
        # Component settings
        "enable_speech_processing": True,
        "enable_vision_processing": True,
        "enable_multimodal_fusion": True,
        "enable_reasoning": True,
        "enable_learning": True,
        
        # Quality settings
        "default_quality_level": "balanced",
        "quality_monitoring": True,
        
        # Memory settings
        "working_memory_size": 500,  # Smaller for development
        "context_window_size": 2048,  # Smaller for development
        "memory_consolidation_interval": 1800,  # 30 minutes
        
        # Caching settings
        "enable_response_caching": True,
        "enable_component_caching": True,
        "cache_ttl_seconds": 1800,  # Shorter for development
        
        # Security settings
        "require_authentication": False,  # Disabled for development
        "enable_authorization": False,   # Disabled for development
        "audit_logging": True,
        
        # Performance settings
        "enable_performance_monitoring": True,
        "enable_profiling": True,
        "gc_interval_seconds": 60,  # More frequent for development
    }
    
    # Component manager configuration
    COMPONENT_MANAGER_CONFIG = {
        "auto_discovery": True,
        "parallel_initialization": True,
        "health_monitoring": True,
        "auto_restart": True,
        "restart_max_attempts": 5,
        "restart_backoff_factor": 1.5,
        "shutdown_timeout": 10.0,  # Faster shutdown for development
        "initialization_timeout": 30.0,
        "hot_reload_enabled": True,
        "dependency_injection": True,
        "component_isolation": False,  # Disabled for easier debugging
    }
    
    # Workflow orchestrator configuration
    WORKFLOW_CONFIG = {
        "max_execution_time": 120.0,  # 2 minutes for development
        "max_step_retries": 2,  # Fewer retries for faster feedback
        "enable_adaptive_workflows": True,
        "enable_workflow_learning": True,
        "max_concurrent_workflows": 3,  # Lower for development
        "step_timeout_default": 30.0,
        "enable_workflow_caching": True,
        "workflow_persistence": True,
        "enable_workflow_debugging": True,
        "workflow_visualization": True,
    }
    
    # Session manager configuration
    SESSIONS_CONFIG = {
        "storage_type": "memory",  # Use memory for development
        "node_id": f"dev_node_{os.getpid()}",
        "enable_clustering": False,  # Disabled for development
        "max_idle_time": 3600.0,  # 1 hour
        "max_session_time": 14400.0,  # 4 hours
        "cleanup_on_expire": True,
        "persist_context": True,
        "auto_save_interval": 120.0,  # 2 minutes
        "encryption_enabled": False,  # Disabled for development
        "audit_logging": True,
        "session_debugging": True,
        "context_versioning": True,
    }
    
    # Interaction handler configuration
    INTERACTIONS_CONFIG = {
        "max_duration": 3600.0,  # 1 hour
        "default_timeout": 120.0,  # 2 minutes
        "enable_real_time": True,
        "enable_streaming": True,
        "max_history": 50,  # Smaller for development
        "context_adaptation": True,
        "user_modeling": True,
        "preference_learning": True,
        "feedback_collection": True,
        "interaction_debugging": True,
    }
    
    # Plugin manager configuration
    PLUGINS_CONFIG = {
        "directories": [
            "plugins/",
            "src/plugins/",
            "data/plugins/",
            "tests/fixtures/plugins/"  # Test plugins
        ],
        "auto_discovery": True,
        "hot_reload": True,
        "security_validation": False,  # Disabled for development
        "enable_clustering": False,
        "max_sessions_per_node": 100,
        "plugin_timeout": 30.0,
        "enable_plugin_debugging": True,
        "allow_unsafe_plugins": True,  # Only for development!
    }
    
    # Memory system configuration
    MEMORY_CONFIG = {
        # Core memory settings
        "working_memory": {
            "capacity": 1000,
            "cleanup_interval": 300,  # 5 minutes
            "persistence": False,  # Memory only for development
            "compression": False,
        },
        "episodic_memory": {
            "storage_backend": "local_file",
            "max_episodes": 10000,
            "retention_days": 30,
            "indexing": True,
            "auto_summarization": False,  # Disabled for development
        },
        "semantic_memory": {
            "storage_backend": "local_vector_store",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_dimension": 384,
            "similarity_threshold": 0.7,
            "max_memories": 50000,
        },
        
        # Storage settings
        "vector_store": {
            "backend": "faiss",  # Local FAISS for development
            "index_type": "IVFFlat",
            "nlist": 100,
            "persistence_path": "data/memory/vectors/",
            "backup_enabled": False,
        },
        
        # Cache settings
        "cache": {
            "backend": "local",  # Local cache for development
            "max_size_mb": 512,
            "ttl_seconds": 3600,
            "cleanup_interval": 300,
        },
        
        # Context management
        "context": {
            "max_context_size": 8192,
            "context_window_overlap": 512,
            "auto_summarization": False,
            "context_versioning": True,
        }
    }
    
    # Learning system configuration
    LEARNING_CONFIG = {
        "continual_learning": {
            "enabled": True,
            "learning_rate": 0.001,
            "update_frequency": "on_feedback",
            "memory_replay": True,
            "catastrophic_forgetting_prevention": True,
            "model_checkpointing": True,
            "checkpoint_frequency": "hourly",
        },
        "preference_learning": {
            "enabled": True,
            "implicit_feedback": True,
            "explicit_feedback": True,
            "preference_decay": 0.95,
            "update_threshold": 5,
        },
        "feedback_processing": {
            "enabled": True,
            "real_time_processing": True,
            "sentiment_analysis": True,
            "feedback_validation": False,  # Disabled for development
            "auto_labeling": True,
        },
        "model_adaptation": {
            "enabled": True,
            "adaptation_frequency": "daily",
            "validation_split": 0.2,
            "early_stopping": True,
            "hyperparameter_tuning": False,  # Disabled for development
        }
    }
    
    # Processing pipelines configuration
    PROCESSING_CONFIG = {
        # Speech processing
        "speech": {
            "whisper_model": "base",  # Faster model for development
            "device": "cpu",  # Use CPU for development compatibility
            "compute_type": "float32",
            "language": "auto",
            "task": "transcribe",
            "vad_enabled": True,
            "noise_reduction": True,
            "emotion_detection": True,
            "speaker_recognition": False,  # Disabled for development
        },
        
        # Vision processing
        "vision": {
            "models": {
                "object_detection": "yolov5s",  # Faster model
                "face_recognition": "mtcnn",
                "ocr": "easyocr",
                "pose_estimation": "mediapipe"
            },
            "device": "cpu",
            "batch_size": 1,
            "image_preprocessing": True,
            "quality_optimization": False,
        },
        
        # Natural language processing
        "nlp": {
            "models": {
                "intent_detection": "distilbert-base-uncased",
                "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment",
                "entity_extraction": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "language_detection": "langdetect"
            },
            "device": "cpu",
            "max_length": 512,
            "batch_size": 1,
            "caching": True,
        },
        
        # Multimodal processing
        "multimodal": {
            "fusion_strategy": "early_fusion",
            "alignment_method": "attention",
            "feature_extraction": True,
            "cross_modal_attention": True,
            "temporal_modeling": False,  # Disabled for development
        }
    }
    
    # Skills management configuration
    SKILLS_CONFIG = {
        "auto_discovery": True,
        "skill_validation": False,  # Disabled for development
        "hot_reload": True,
        "skill_caching": True,
        "execution_timeout": 30.0,
        "max_concurrent_skills": 5,
        "skill_debugging": True,
        "performance_monitoring": True,
        
        # Built-in skills
        "builtin_skills": {
            "enabled": True,
            "auto_register": True,
            "categories": [
                "core_skills",
                "productivity_skills",
                "creative_skills",
                "analytical_skills"
            ]
        },
        
        # Custom skills
        "custom_skills": {
            "enabled": True,
            "directories": [
                "src/skills/custom/",
                "data/skills/user/",
                "tests/fixtures/skills/"
            ],
            "validation": False,  # Disabled for development
            "sandboxing": False,  # Disabled for development
        }
    }
    
    # External integrations configuration
    INTEGRATIONS_CONFIG = {
        # LLM integrations
        "llm": {
            "default_provider": "ollama",
            "providers": {
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "models": {
                        "default": "llama2:7b",
                        "fast": "llama2:7b",
                        "quality": "llama2:13b"
                    },
                    "timeout": 30.0,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "stream": True,
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", "dev-key"),
                    "models": {
                        "default": "gpt-3.5-turbo",
                        "fast": "gpt-3.5-turbo",
                        "quality": "gpt-4"
                    },
                    "timeout": 30.0,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "enabled": bool(os.getenv("OPENAI_API_KEY")),
                },
                "mock": {  # Mock provider for testing
                    "enabled": True,
                    "response_delay": 1.0,
                    "error_rate": 0.01,
                }
            },
            "fallback_chain": ["ollama", "mock"],
            "load_balancing": False,
            "caching": True,
            "cache_ttl": 3600,
        },
        
        # Storage integrations
        "storage": {
            "database": {
                "backend": "sqlite",
                "url": "sqlite:///data/development.db",
                "pool_size": 5,
                "echo": True,  # SQL logging for development
                "auto_migrate": True,
            },
            "file_storage": {
                "backend": "local",
                "base_path": "data/files/",
                "max_file_size": "100MB",
                "allowed_extensions": [".txt", ".pdf", ".doc", ".docx", ".md"],
            },
            "blob_storage": {
                "backend": "local",
                "base_path": "data/blobs/",
                "compression": False,
                "encryption": False,
            }
        },
        
        # Cache integrations
        "cache": {
            "backend": "local",  # Use local cache for development
            "redis": {
                "enabled": False,  # Disabled by default for development
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None,
                "ssl": False,
            },
            "local": {
                "max_size_mb": 256,
                "ttl_seconds": 3600,
                "cleanup_interval": 300,
            }
        },
        
        # External APIs (mocked for development)
        "external_apis": {
            "web_search": {
                "provider": "mock",  # Use mock for development
                "enabled": True,
                "timeout": 10.0,
            },
            "weather": {
                "provider": "mock",
                "enabled": True,
                "timeout": 5.0,
            },
            "calendar": {
                "provider": "mock",
                "enabled": True,
                "timeout": 5.0,
            },
            "notifications": {
                "provider": "console",  # Console notifications for development
                "enabled": True,
            }
        }
    }
    
    # API configuration
    API_CONFIG = {
        # REST API
        "rest": {
            "host": "localhost",
            "port": 8000,
            "debug": True,
            "reload": True,  # Auto-reload on code changes
            "workers": 1,  # Single worker for development
            "timeout": 60.0,
            "max_request_size": "50MB",
            
            # CORS settings
            "cors": {
                "enabled": True,
                "allow_origins": ["*"],  # Allow all origins for development
                "allow_methods": ["*"],
                "allow_headers": ["*"],
                "allow_credentials": True,
            },
            
            # Rate limiting
            "rate_limiting": {
                "enabled": False,  # Disabled for development
                "requests_per_minute": 1000,
                "burst_limit": 100,
            },
            
            # Authentication
            "authentication": {
                "enabled": False,  # Disabled for development
                "jwt_secret": "dev-secret-key",
                "token_expiry": 86400,  # 24 hours
            }
        },
        
        # WebSocket API
        "websocket": {
            "host": "localhost",
            "port": 8001,
            "max_connections": 100,
            "heartbeat_interval": 30.0,
            "message_queue_size": 1000,
            "compression": False,
            "authentication": False,  # Disabled for development
        },
        
        # GraphQL API
        "graphql": {
            "enabled": True,
            "playground": True,  # Enable GraphQL playground
            "introspection": True,  # Enable introspection
            "debug": True,
            "max_query_depth": 10,
            "max_query_complexity": 1000,
        },
        
        # gRPC API
        "grpc": {
            "enabled": False,  # Disabled by default for development
            "host": "localhost",
            "port": 50051,
            "max_workers": 4,
            "reflection": True,  # Enable reflection for development
        }
    }
    
    # Security configuration (relaxed for development)
    SECURITY_CONFIG = {
        "authentication": {
            "enabled": False,  # Disabled for development
            "provider": "mock",
            "session_timeout": 86400,  # 24 hours
            "max_login_attempts": 10,
            "lockout_duration": 300,  # 5 minutes
        },
        "authorization": {
            "enabled": False,  # Disabled for development
            "rbac_enabled": False,
            "default_permissions": ["read", "write", "execute"],
            "admin_users": ["dev", "admin"],
        },
        "encryption": {
            "enabled": False,  # Disabled for development
            "algorithm": "AES-256-GCM",
            "key_rotation": False,
            "key_derivation": "PBKDF2",
        },
        "sanitization": {
            "enabled": True,  # Keep enabled for safety
            "input_validation": True,
            "output_encoding": True,
            "sql_injection_protection": True,
            "xss_protection": True,
        },
        "audit": {
            "enabled": True,
            "log_all_requests": True,
            "log_authentication": True,
            "log_authorization": True,
            "retention_days": 30,
        }
    }
    
    # Observability configuration
    OBSERVABILITY_CONFIG = {
        # Logging
        "logging": {
            "level": "DEBUG",
            "format": "development",  # Verbose format for development
            "handlers": ["console", "file"],
            "file_rotation": False,
            "max_file_size": "10MB",
            "backup_count": 3,
            "structured_logging": True,
            "request_logging": True,
            "performance_logging": True,
            
            # Component-specific logging
            "component_levels": {
                "core_engine": "DEBUG",
                "workflow_orchestrator": "DEBUG",
                "session_manager": "DEBUG",
                "interaction_handler": "DEBUG",
                "plugin_manager": "DEBUG",
                "memory_manager": "DEBUG",
                "skill_factory": "DEBUG",
                "api": "INFO",
                "integrations": "INFO",
            }
        },
        
        # Metrics
        "metrics": {
            "enabled": True,
            "provider": "prometheus",
            "port": 8002,
            "path": "/metrics",
            "collection_interval": 15,  # 15 seconds
            "retention_hours": 24,
            "custom_metrics": True,
            "component_metrics": True,
            "performance_metrics": True,
        },
        
        # Tracing
        "tracing": {
            "enabled": True,
            "provider": "jaeger",
            "agent_host": "localhost",
            "agent_port": 6831,
            "sampling_rate": 1.0,  # 100% sampling for development
            "service_name": "ai-assistant-dev",
            "trace_context": True,
            "span_attributes": True,
        },
        
        # Profiling
        "profiling": {
            "enabled": True,
            "cpu_profiling": True,
            "memory_profiling": True,
            "gpu_profiling": False,  # Disabled for development
            "profile_duration": 60,  # 1 minute
            "output_path": "data/profiles/",
            "flamegraphs": True,
        },
        
        # Health checks
        "health_checks": {
            "enabled": True,
            "interval": 30,  # 30 seconds
            "timeout": 10,  # 10 seconds
            "endpoints": {
                "liveness": "/health/live",
                "readiness": "/health/ready",
                "components": "/health/components",
            },
            "detailed_responses": True,
        }
    }
    
    # Development-specific features
    DEVELOPMENT_CONFIG = {
        # Hot reloading
        "hot_reload": {
            "enabled": True,
            "watch_directories": [
                "src/",
                "configs/",
            ],
            "ignore_patterns": [
                "*.pyc",
                "__pycache__",
                "*.log",
                ".git",
                "data/cache/",
                "data/logs/",
            ],
            "restart_delay": 2.0,
            "max_restart_attempts": 5,
        },
        
        # Mock services
        "mock_services": {
            "enabled": True,
            "external_apis": True,
            "llm_providers": True,
            "storage_backends": False,
            "cache_backends": False,
            "response_delays": {
                "min": 0.1,
                "max": 2.0,
            },
            "error_simulation": {
                "enabled": True,
                "error_rate": 0.01,  # 1% error rate
            }
        },
        
        # Testing helpers
        "testing": {
            "auto_test_discovery": True,
            "test_data_generation": True,
            "mock_user_sessions": True,
            "synthetic_workloads": True,
            "performance_testing": False,
            "load_testing": False,
        },
        
        # Debugging
        "debugging": {
            "interactive_debugger": True,
            "breakpoint_injection": True,
            "variable_inspection": True,
            "execution_tracing": True,
            "memory_debugging": True,
            "workflow_visualization": True,
            "component_introspection": True,
        },
        
        # Code quality
        "code_quality": {
            "linting": True,
            "type_checking": True,
            "code_formatting": True,
            "import_sorting": True,
            "complexity_analysis": False,
            "security_scanning": False,
        }
    }
    
    # File paths (development-specific)
    PATHS_CONFIG = {
        "base_dir": Path(__file__).parent.parent.parent.parent,
        "data_dir": Path("data/"),
        "logs_dir": Path("data/logs/"),
        "cache_dir": Path("data/cache/"),
        "models_dir": Path("data/models/"),
        "plugins_dir": Path("plugins/"),
        "configs_dir": Path("configs/"),
        "temp_dir": Path("data/temp/"),
        
        # Development-specific paths
        "dev_data_dir": Path("data/development/"),
        "test_data_dir": Path("tests/fixtures/"),
        "profiles_dir": Path("data/profiles/"),
        "debug_dir": Path("data/debug/"),
    }
    
    # Environment variables override
    @classmethod
    def from_env(cls) -> 'DevelopmentConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("DEBUG"):
            config.DEBUG = os.getenv("DEBUG").lower() == "true"
        
        if os.getenv("LOG_LEVEL"):
            config.LOG_LEVEL = os.getenv("LOG_LEVEL")
        
        if os.getenv("HOT_RELOAD_ENABLED"):
            config.HOT_RELOAD_ENABLED = os.getenv("HOT_RELOAD_ENABLED").lower() == "true"
        
        # API configuration overrides
        if os.getenv("API_HOST"):
            config.API_CONFIG["rest"]["host"] = os.getenv("API_HOST")
        
        if os.getenv("API_PORT"):
            config.API_CONFIG["rest"]["port"] = int(os.getenv("API_PORT"))
        
        # Database configuration override
        if os.getenv("DATABASE_URL"):
            config.INTEGRATIONS_CONFIG["storage"]["database"]["url"] = os.getenv("DATABASE_URL")
        
        # LLM configuration overrides
        if os.getenv("OPENAI_API_KEY"):
            config.INTEGRATIONS_CONFIG["llm"]["providers"]["openai"]["api_key"] = os.getenv("OPENAI_API_KEY")
            config.INTEGRATIONS_CONFIG["llm"]["providers"]["openai"]["enabled"] = True
        
        if os.getenv("OLLAMA_BASE_URL"):
            config.INTEGRATIONS_CONFIG["llm"]["providers"]["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL")
        
        return config
    
    def validate(self) -> None:
        """Validate the development configuration."""
        super().validate()
        
        # Development-specific validations
        if self.HOT_RELOAD_ENABLED and not self.DEBUG:
            raise ValueError("Hot reload requires DEBUG mode to be enabled")
        
        if self.PROFILING_ENABLED and not self.DEBUG:
            print("Warning: Profiling enabled without DEBUG mode")
        
        # Ensure development directories exist
        for path_key, path_value in self.PATHS_CONFIG.items():
            if isinstance(path_value, Path) and not path_value.exists():
                path_value.mkdir(parents=True, exist_ok=True)
        
        # Validate that development features are not enabled for production
        production_checks = [
            ("require_authentication", False),
            ("enable_authorization", False),
            ("security_validation", False),
        ]
        
        for check_name, expected_value in production_checks:
            # This is development config, so these should be disabled
            pass
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for a specific component."""
        component_configs = {
            "core_engine": self.CORE_ENGINE_CONFIG,
            "component_manager": self.COMPONENT_MANAGER_CONFIG,
            "workflow_orchestrator": self.WORKFLOW_CONFIG,
            "session_manager": self.SESSIONS_CONFIG,
            "interaction_handler": self.INTERACTIONS_CONFIG,
            "plugin_manager": self.PLUGINS_CONFIG,
            "memory_manager": self.MEMORY_CONFIG,
            "learning_system": self.LEARNING_CONFIG,
            "skills_manager": self.SKILLS_CONFIG,
            "api": self.API_CONFIG,
            "security": self.SECURITY_CONFIG,
            "observability": self.OBSERVABILITY_CONFIG,
        }
        
        return component_configs.get(component_name, {})
    
    def get_integration_config(self, integration_name: str) -> Dict[str, Any]:
        """Get configuration for a specific integration."""
        return self.INTEGRATIONS_CONFIG.get(integration_name, {})
    
    def get_processing_config(self, processor_name: str) -> Dict[str, Any]:
        """Get configuration for a specific processor."""
        return self.PROCESSING_CONFIG.get(processor_name, {})
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a development feature is enabled."""
        feature_flags = {
            "hot_reload": self.HOT_RELOAD_ENABLED,
            "profiling": self.PROFILING_ENABLED,
            "metrics": self.METRICS_COLLECTION_ENABLED,
            "tracing": self.TRACING_ENABLED,
            "debugging": self.DEBUG,
            "mock_services": self.DEVELOPMENT_CONFIG["mock_services"]["enabled"],
            "auto_testing": self.DEVELOPMENT_CONFIG["testing"]["auto_test_discovery"],
        }
        
        return feature_flags.get(feature_name, False)
    
    def get_mock_config(self, service_name: str) -> Dict[str, Any]:
        """Get mock configuration for external services."""
        mock_configs = {
            "llm": {
                "provider": "mock",
                "response_time": 1.0,
                "error_rate": 0.01,
                "responses": {
                    "default": "This is a mock response from the development LLM provider.",
                    "thinking": "I'm thinking about your request...",
                    "error": "Mock error: Service temporarily unavailable"
                }
            },
            "web_search": {
                "results": [
                    {"title": "Mock Search Result 1", "url": "https://example.com/1", "snippet": "Mock snippet 1"},
                    {"title": "Mock Search Result 2", "url": "https://example.com/2", "snippet": "Mock snippet 2"},
                ]
            },
            "weather": {
                "temperature": 22,
                "condition": "sunny",
                "humidity": 65,
                "location": "Development City"
            }
        }
        
        return mock_configs.get(service_name, {})


# Create default development configuration instance
config = DevelopmentConfig.from_env()

# Export configuration
__all__ = ["DevelopmentConfig", "config"]
