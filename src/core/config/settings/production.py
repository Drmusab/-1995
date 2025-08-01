"""
Production Configuration Settings
Author: Drmusab
Last Modified: 2025-06-13 11:08:14 UTC

This module provides comprehensive production configuration settings for the
AI Assistant system, including security hardening, performance optimization,
monitoring, and integration with all core system components.
"""

import logging
import os
import secrets
import ssl
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core imports
from src.core.config.validators.config_validator import ConfigValidator
from src.core.security.encryption import EncryptionLevel
from src.observability.monitoring.metrics import MetricLevel


@dataclass
class SecurityConfiguration:
    """Production security configuration."""

    # Authentication settings
    jwt_secret_key: str = field(
        default_factory=lambda: os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    )
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    jwt_refresh_expiration_days: int = 30

    # Encryption settings
    encryption_key: str = field(
        default_factory=lambda: os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32))
    )
    encryption_level: str = "AES_256_GCM"
    password_salt_rounds: int = 12

    # SSL/TLS Configuration
    ssl_enabled: bool = True
    ssl_cert_path: str = os.getenv("SSL_CERT_PATH", "/etc/ssl/certs/assistant.pem")
    ssl_key_path: str = os.getenv("SSL_KEY_PATH", "/etc/ssl/private/assistant.key")
    ssl_ca_path: Optional[str] = os.getenv("SSL_CA_PATH")
    ssl_protocol: int = ssl.PROTOCOL_TLSv1_2
    ssl_ciphers: str = "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
    tls_min_version: str = "1.2"

    # CORS Configuration
    cors_enabled: bool = True
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: ["https://assistant.yourdomain.com", "https://api.yourdomain.com"]
    )
    cors_allowed_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_allowed_headers: List[str] = field(
        default_factory=lambda: ["Authorization", "Content-Type", "X-Requested-With", "X-API-Key"]
    )
    cors_max_age: int = 86400

    # Security Headers
    security_headers: Dict[str, str] = field(
        default_factory=lambda: {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }
    )

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20
    rate_limit_storage: str = "redis"

    # Input Validation
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_json_payload_size: int = 5 * 1024 * 1024  # 5MB
    input_sanitization_enabled: bool = True

    # Session Security
    session_secure_cookies: bool = True
    session_httponly_cookies: bool = True
    session_samesite: str = "strict"
    session_timeout_minutes: int = 60


@dataclass
class DatabaseConfiguration:
    """Production database configuration."""

    # Primary Database (PostgreSQL)
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://user:pass@localhost:5432/assistant_prod"
    )
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    database_pool_recycle: int = 3600
    database_echo: bool = False

    # Connection settings
    database_connect_timeout: int = 10
    database_command_timeout: int = 30
    database_ssl_mode: str = "require"
    database_ssl_cert: Optional[str] = os.getenv("DB_SSL_CERT")
    database_ssl_key: Optional[str] = os.getenv("DB_SSL_KEY")
    database_ssl_ca: Optional[str] = os.getenv("DB_SSL_CA")

    # Read replicas
    read_replica_urls: List[str] = field(
        default_factory=lambda: [
            os.getenv("READ_replica_url", "").strip()
            for url in os.getenv("READ_REPLICA_URLS", "").split(",")
            if url.strip()
        ]
    )

    # Backup settings
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    backup_storage_path: str = os.getenv("BACKUP_STORAGE_PATH", "/backups/database")

    # Migration settings
    auto_migrate: bool = False
    migration_timeout: int = 300


@dataclass
class CacheConfiguration:
    """Production cache configuration."""

    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_cluster_enabled: bool = True
    redis_cluster_nodes: List[str] = field(
        default_factory=lambda: [
            node.strip() for node in os.getenv("REDIS_CLUSTER_NODES", "").split(",") if node.strip()
        ]
    )

    # Connection settings
    redis_max_connections: int = 100
    redis_connection_timeout: int = 5
    redis_socket_timeout: int = 5
    redis_retry_on_timeout: bool = True
    redis_health_check_interval: int = 30

    # SSL Configuration
    redis_ssl_enabled: bool = True
    redis_ssl_cert_reqs: str = "required"
    redis_ssl_ca_certs: Optional[str] = os.getenv("REDIS_SSL_CA")
    redis_ssl_certfile: Optional[str] = os.getenv("REDIS_SSL_CERT")
    redis_ssl_keyfile: Optional[str] = os.getenv("REDIS_SSL_KEY")

    # Cache strategies
    default_ttl: int = 3600  # 1 hour
    session_ttl: int = 86400  # 24 hours
    response_cache_ttl: int = 1800  # 30 minutes
    vector_cache_ttl: int = 604800  # 1 week

    # Memory management
    redis_maxmemory_policy: str = "allkeys-lru"
    redis_max_memory: str = "4gb"

    # Distributed locking
    lock_timeout: int = 30
    lock_blocking_timeout: int = 10


@dataclass
class APIConfiguration:
    """Production API configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8000"))
    workers: int = int(os.getenv("WORKERS", "4"))
    max_requests: int = 1000
    max_requests_jitter: int = 100
    timeout: int = 30
    keepalive: int = 2

    # API Versioning
    default_api_version: str = "v1"
    supported_versions: List[str] = field(default_factory=lambda: ["v1", "v2"])
    version_deprecation_warnings: bool = True

    # Rate Limiting (API-specific)
    api_rate_limit_per_hour: int = 10000
    api_rate_limit_per_day: int = 100000
    api_burst_limit: int = 100

    # Request/Response settings
    max_request_timeout: int = 300  # 5 minutes
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    request_compression: bool = True
    response_compression: bool = True
    compression_level: int = 6

    # WebSocket settings
    websocket_enabled: bool = True
    websocket_max_connections: int = 1000
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 60

    # GraphQL settings
    graphql_enabled: bool = True
    graphql_introspection: bool = False  # Disabled in production
    graphql_playground: bool = False  # Disabled in production
    graphql_depth_limit: int = 10
    graphql_complexity_limit: int = 1000


@dataclass
class MonitoringConfiguration:
    """Production monitoring configuration."""

    # Metrics Configuration
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_endpoint: str = "/metrics"
    metrics_level: str = "INFO"

    # Prometheus settings
    prometheus_enabled: bool = True
    prometheus_namespace: str = "ai_assistant"
    prometheus_job_name: str = "assistant-prod"
    prometheus_scrape_interval: str = "15s"

    # Tracing Configuration
    tracing_enabled: bool = True
    tracing_service_name: str = "ai-assistant"
    tracing_environment: str = "production"
    jaeger_endpoint: str = os.getenv("JAEGER_ENDPOINT", "http://jaeger:14268/api/traces")
    tracing_sample_rate: float = 0.1  # 10% sampling in production

    # Health Check Configuration
    health_check_enabled: bool = True
    health_check_endpoint: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 10
    deep_health_check_endpoint: str = "/health/deep"

    # Alerting Configuration
    alerting_enabled: bool = True
    alert_webhook_url: str = os.getenv("ALERT_WEBHOOK_URL", "")
    critical_alert_channels: List[str] = field(
        default_factory=lambda: ["slack", "email", "pagerduty"]
    )

    # Log Aggregation
    log_aggregation_enabled: bool = True
    elasticsearch_url: str = os.getenv("ELASTICSEARCH_URL", "")
    log_index_pattern: str = "ai-assistant-logs-%Y.%m.%d"

    # Performance monitoring
    apm_enabled: bool = True
    apm_service_name: str = "ai-assistant"
    apm_environment: str = "production"
    apm_server_url: str = os.getenv("APM_SERVER_URL", "")


@dataclass
class PerformanceConfiguration:
    """Production performance optimization configuration."""

    # Resource Limits
    max_memory_usage_gb: float = 8.0
    max_cpu_usage_percent: float = 80.0
    max_concurrent_requests: int = 100
    max_concurrent_workflows: int = 50
    max_concurrent_sessions: int = 1000

    # Threading and Async settings
    thread_pool_size: int = 20
    async_pool_size: int = 100
    io_thread_pool_size: int = 10

    # Component Performance
    component_timeout_seconds: float = 30.0
    skill_execution_timeout: float = 60.0
    workflow_timeout_seconds: float = 300.0
    session_cleanup_interval: int = 300  # 5 minutes

    # Memory Management
    memory_cleanup_enabled: bool = True
    memory_cleanup_interval: int = 600  # 10 minutes
    garbage_collection_enabled: bool = True
    gc_threshold: tuple = (700, 10, 10)

    # Caching Performance
    cache_warmup_enabled: bool = True
    cache_preload_common_responses: bool = True
    vector_cache_size: int = 10000
    response_cache_size: int = 5000

    # LLM Performance
    llm_request_timeout: int = 120
    llm_max_retries: int = 3
    llm_batch_size: int = 10
    llm_concurrent_requests: int = 5

    # Auto-scaling settings
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 70.0  # CPU %
    scale_down_threshold: float = 30.0  # CPU %
    min_instances: int = 2
    max_instances: int = 20


@dataclass
class IntegrationConfiguration:
    """Production integration configuration."""

    # LLM Providers
    primary_llm_provider: str = "openai"
    fallback_llm_providers: List[str] = field(default_factory=lambda: ["anthropic", "ollama"])

    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_organization: str = os.getenv("OPENAI_ORGANIZATION", "")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout: int = 120
    openai_max_retries: int = 3

    # Anthropic Configuration
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_timeout: int = 120

    # DeepSeek Configuration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = "https://api.deepseek.com"

    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    ollama_timeout: int = 300

    # External APIs
    web_search_enabled: bool = True
    web_search_api_key: str = os.getenv("WEB_SEARCH_API_KEY", "")
    weather_api_key: str = os.getenv("WEATHER_API_KEY", "")
    calendar_api_enabled: bool = True

    # Email Service
    email_service_enabled: bool = True
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_use_tls: bool = True

    # Notification Services
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")
    teams_webhook_url: str = os.getenv("TEAMS_WEBHOOK_URL", "")


@dataclass
class InfrastructureConfiguration:
    """Production infrastructure configuration."""

    # Cloud Provider Settings
    cloud_provider: str = os.getenv("CLOUD_PROVIDER", "aws")
    region: str = os.getenv("CLOUD_REGION", "us-east-1")
    availability_zones: List[str] = field(
        default_factory=lambda: ["us-east-1a", "us-east-1b", "us-east-1c"]
    )

    # AWS Configuration
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_session_token: str = os.getenv("AWS_SESSION_TOKEN", "")
    aws_s3_bucket: str = os.getenv("AWS_S3_BUCKET", "")
    aws_s3_region: str = os.getenv("AWS_S3_REGION", "us-east-1")

    # Azure Configuration
    azure_storage_account: str = os.getenv("AZURE_STORAGE_ACCOUNT", "")
    azure_storage_key: str = os.getenv("AZURE_STORAGE_KEY", "")
    azure_container_name: str = os.getenv("AZURE_CONTAINER_NAME", "")

    # GCP Configuration
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "")
    gcp_service_account_key: str = os.getenv("GCP_SERVICE_ACCOUNT_KEY", "")
    gcp_storage_bucket: str = os.getenv("GCP_STORAGE_BUCKET", "")

    # Kubernetes Configuration
    kubernetes_enabled: bool = True
    kubernetes_namespace: str = os.getenv("KUBERNETES_NAMESPACE", "ai-assistant")
    kubernetes_service_account: str = "ai-assistant-service"

    # Container Configuration
    container_registry: str = os.getenv("CONTAINER_REGISTRY", "")
    container_image_tag: str = os.getenv("IMAGE_TAG", "latest")
    container_pull_policy: str = "Always"

    # Load Balancer Configuration
    load_balancer_enabled: bool = True
    load_balancer_type: str = "application"  # application, network
    load_balancer_ssl_policy: str = "ELBSecurityPolicy-TLS-1-2-2017-01"


@dataclass
class BackupConfiguration:
    """Production backup and disaster recovery configuration."""

    # Backup Settings
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM UTC
    backup_retention_days: int = 30
    backup_compression: bool = True
    backup_encryption: bool = True

    # Database Backup
    database_backup_enabled: bool = True
    database_backup_method: str = "pg_dump"  # pg_dump, wal-e, barman
    database_backup_parallel_jobs: int = 4

    # File System Backup
    filesystem_backup_enabled: bool = True
    backup_directories: List[str] = field(
        default_factory=lambda: [
            "/data/models",
            "/data/user_data",
            "/data/knowledge_base",
            "/configs",
        ]
    )

    # Backup Storage
    backup_storage_type: str = "s3"  # s3, azure, gcp, local
    backup_storage_bucket: str = os.getenv("BACKUP_STORAGE_BUCKET", "")
    backup_storage_path: str = "/backups/ai-assistant"

    # Disaster Recovery
    disaster_recovery_enabled: bool = True
    recovery_time_objective_hours: int = 4  # RTO
    recovery_point_objective_hours: int = 1  # RPO

    # Cross-region replication
    cross_region_replication: bool = True
    replication_regions: List[str] = field(default_factory=lambda: ["us-west-2", "eu-west-1"])

    # Backup Verification
    backup_verification_enabled: bool = True
    backup_test_restore_schedule: str = "0 3 * * 0"  # Weekly on Sunday


@dataclass
class LoggingConfiguration:
    """Production logging configuration."""

    # Log Levels
    root_log_level: str = "INFO"
    application_log_level: str = "INFO"
    security_log_level: str = "WARNING"
    performance_log_level: str = "INFO"

    # Log Formats
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    json_logging: bool = True
    include_trace_id: bool = True

    # Log Destinations
    console_logging: bool = True
    file_logging: bool = True
    remote_logging: bool = True

    # File Logging
    log_file_path: str = "/var/log/ai-assistant/application.log"
    log_file_max_size: str = "100MB"
    log_file_backup_count: int = 10
    log_file_rotation: str = "time"  # time, size

    # Remote Logging
    elasticsearch_enabled: bool = True
    elasticsearch_url: str = os.getenv("ELASTICSEARCH_URL", "")
    elasticsearch_index: str = "ai-assistant-logs"

    # Structured Logging
    structured_logging: bool = True
    log_correlation_id: bool = True
    log_user_context: bool = False  # Privacy consideration

    # Log Filtering
    log_filtering_enabled: bool = True
    sensitive_data_filtering: bool = True
    pii_filtering_enabled: bool = True

    # Audit Logging
    audit_logging_enabled: bool = True
    audit_log_file: str = "/var/log/ai-assistant/audit.log"
    audit_events: List[str] = field(
        default_factory=lambda: [
            "user_authentication",
            "permission_change",
            "data_access",
            "configuration_change",
            "system_modification",
        ]
    )


class ProductionConfig:
    """
    Comprehensive production configuration for the AI Assistant system.

    This configuration provides enterprise-grade settings for:
    - Security hardening and compliance
    - High-availability and scalability
    - Performance optimization
    - Monitoring and observability
    - Integration with external services
    - Backup and disaster recovery
    """

    def __init__(self):
        """Initialize production configuration."""
        # Environment
        self.environment = "production"
        self.debug = False
        self.testing = False

        # Application settings
        self.app_name = "AI Assistant"
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.app_description = "Advanced AI Assistant System"

        # Configuration sections
        self.security = SecurityConfiguration()
        self.database = DatabaseConfiguration()
        self.cache = CacheConfiguration()
        self.api = APIConfiguration()
        self.monitoring = MonitoringConfiguration()
        self.performance = PerformanceConfiguration()
        self.integrations = IntegrationConfiguration()
        self.infrastructure = InfrastructureConfiguration()
        self.backup = BackupConfiguration()
        self.logging = LoggingConfiguration()

        # Core System Configuration
        self.core_config = self._get_core_config()

        # Validate configuration
        self._validate_configuration()

    def _get_core_config(self) -> Dict[str, Any]:
        """Get core system configuration."""
        return {
            # Assistant Core Configuration
            "assistant": {
                "core_engine": {
                    "processing_mode": "asynchronous",
                    "max_concurrent_requests": self.performance.max_concurrent_requests,
                    "default_timeout_seconds": self.performance.component_timeout_seconds,
                    "enable_real_time_processing": True,
                    "enable_speech_processing": True,
                    "enable_vision_processing": True,
                    "enable_multimodal_fusion": True,
                    "enable_reasoning": True,
                    "enable_learning": True,
                    "default_quality_level": "balanced",
                    "adaptive_quality": True,
                    "enable_performance_monitoring": True,
                    "enable_profiling": False,  # Disabled in production
                    "gc_interval_seconds": 300,
                },
                "component_manager": {
                    "auto_discovery": True,
                    "parallel_initialization": True,
                    "health_monitoring": True,
                    "component_timeout": self.performance.component_timeout_seconds,
                    "max_restarts": 3,
                    "restart_delay": 5.0,
                },
                "workflow_orchestrator": {
                    "max_execution_time": self.performance.workflow_timeout_seconds,
                    "max_step_retries": 3,
                    "enable_adaptive_workflows": True,
                    "enable_workflow_learning": True,
                    "parallel_execution": True,
                    "max_concurrent_workflows": self.performance.max_concurrent_workflows,
                },
                "session_manager": {
                    "storage_type": "database",
                    "max_idle_time": 1800.0,  # 30 minutes
                    "max_session_time": 86400.0,  # 24 hours
                    "cleanup_interval": self.performance.session_cleanup_interval,
                    "enable_clustering": True,
                    "enable_backup": True,
                    "auto_save_interval": 300.0,
                    "encryption_enabled": True,
                    "audit_logging": True,
                },
                "interaction_handler": {
                    "max_concurrent_interactions": 50,
                    "default_timeout": 300.0,
                    "enable_real_time": True,
                    "enable_streaming": True,
                    "max_conversation_history": 100,
                    "multimodal_support": True,
                },
                "plugin_manager": {
                    "auto_discovery": True,
                    "hot_reload": False,  # Disabled in production
                    "security_validation": True,
                    "max_plugins": 100,
                    "plugin_timeout": 30.0,
                    "sandbox_mode": True,
                },
            },
            # Memory System Configuration
            "memory": {
                "working_memory": {
                    "max_size": 1000,
                    "cleanup_interval": 300,
                    "persistence_enabled": True,
                },
                "episodic_memory": {
                    "max_episodes": 10000,
                    "consolidation_interval": 3600,
                    "retention_days": 365,
                },
                "semantic_memory": {
                    "vector_dimensions": 1536,
                    "similarity_threshold": 0.8,
                    "max_vectors": 100000,
                    "index_type": "faiss",
                },
                "vector_store": {
                    "provider": "pinecone",  # or 'weaviate', 'qdrant'
                    "index_name": "ai-assistant-prod",
                    "embedding_model": "text-embedding-ada-002",
                    "batch_size": 100,
                },
            },
            # Skills Configuration
            "skills": {
                "max_execution_time": self.performance.skill_execution_timeout,
                "enable_skill_discovery": True,
                "skill_validation": True,
                "max_concurrent_skills": 20,
                "skill_timeout": 60.0,
                "enable_skill_learning": True,
            },
            # Processing Configuration
            "processing": {
                "speech": {
                    "provider": "whisper",
                    "model": "whisper-large-v3",
                    "language": "auto",
                    "quality": "high",
                    "enable_diarization": True,
                    "enable_emotion_detection": True,
                },
                "vision": {
                    "enable_ocr": True,
                    "enable_face_recognition": True,
                    "enable_object_detection": True,
                    "model_quality": "high",
                },
                "nlp": {
                    "enable_intent_detection": True,
                    "enable_entity_extraction": True,
                    "enable_sentiment_analysis": True,
                    "language_models": ["gpt-4", "claude-3"],
                },
            },
            # Learning Configuration
            "learning": {
                "continual_learning": {
                    "enabled": True,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "update_frequency": "daily",
                },
                "preference_learning": {
                    "enabled": True,
                    "adaptation_rate": 0.1,
                    "min_interactions": 10,
                },
                "feedback_processing": {
                    "enabled": True,
                    "feedback_threshold": 0.7,
                    "auto_adjustment": True,
                },
            },
        }

    def _validate_configuration(self) -> None:
        """Validate production configuration."""
        validator = ConfigValidator()

        # Critical validations
        required_env_vars = ["DATABASE_URL", "REDIS_URL", "JWT_SECRET_KEY", "ENCRYPTION_KEY"]

        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        # SSL Certificate validation
        if self.security.ssl_enabled:
            ssl_cert_path = Path(self.security.ssl_cert_path)
            ssl_key_path = Path(self.security.ssl_key_path)

            if not ssl_cert_path.exists():
                raise FileNotFoundError(f"SSL certificate not found: {ssl_cert_path}")

            if not ssl_key_path.exists():
                raise FileNotFoundError(f"SSL key not found: {ssl_key_path}")

        # Database connection validation
        if not validator.validate_database_url(self.database.database_url):
            raise ValueError("Invalid database URL format")

        # Redis connection validation
        if not validator.validate_redis_url(self.cache.redis_url):
            raise ValueError("Invalid Redis URL format")

        # Performance limits validation
        if self.performance.max_memory_usage_gb > 32:
            logging.warning("Memory limit exceeds recommended maximum (32GB)")

        if self.performance.max_cpu_usage_percent > 90:
            logging.warning("CPU limit exceeds recommended maximum (90%)")

    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "testing": self.testing,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_description": self.app_description,
            "security": self.security.__dict__,
            "database": self.database.__dict__,
            "cache": self.cache.__dict__,
            "api": self.api.__dict__,
            "monitoring": self.monitoring.__dict__,
            "performance": self.performance.__dict__,
            "integrations": self.integrations.__dict__,
            "infrastructure": self.infrastructure.__dict__,
            "backup": self.backup.__dict__,
            "logging": self.logging.__dict__,
            "core_config": self.core_config,
        }

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration for SQLAlchemy."""
        return {
            "url": self.database.database_url,
            "pool_size": self.database.database_pool_size,
            "max_overflow": self.database.database_max_overflow,
            "pool_timeout": self.database.database_pool_timeout,
            "pool_recycle": self.database.database_pool_recycle,
            "echo": self.database.database_echo,
            "connect_args": {
                "connect_timeout": self.database.database_connect_timeout,
                "command_timeout": self.database.database_command_timeout,
                "sslmode": self.database.database_ssl_mode,
                "sslcert": self.database.database_ssl_cert,
                "sslkey": self.database.database_ssl_key,
                "sslrootcert": self.database.database_ssl_ca,
            },
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        config = {
            "url": self.cache.redis_url,
            "max_connections": self.cache.redis_max_connections,
            "socket_timeout": self.cache.redis_socket_timeout,
            "socket_connect_timeout": self.cache.redis_connection_timeout,
            "retry_on_timeout": self.cache.redis_retry_on_timeout,
            "health_check_interval": self.cache.redis_health_check_interval,
        }

        if self.cache.redis_ssl_enabled:
            config["ssl_cert_reqs"] = self.cache.redis_ssl_cert_reqs
            config["ssl_ca_certs"] = self.cache.redis_ssl_ca_certs
            config["ssl_certfile"] = self.cache.redis_ssl_certfile
            config["ssl_keyfile"] = self.cache.redis_ssl_keyfile

        return config

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return self.security.security_headers.copy()

    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration."""
        return {
            "allow_origins": self.security.cors_allowed_origins,
            "allow_methods": self.security.cors_allowed_methods,
            "allow_headers": self.security.cors_allowed_headers,
            "max_age": self.security.cors_max_age,
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and observability configuration."""
        return {
            "metrics": {
                "enabled": self.monitoring.metrics_enabled,
                "port": self.monitoring.metrics_port,
                "endpoint": self.monitoring.metrics_endpoint,
                "namespace": self.monitoring.prometheus_namespace,
            },
            "tracing": {
                "enabled": self.monitoring.tracing_enabled,
                "service_name": self.monitoring.tracing_service_name,
                "environment": self.monitoring.tracing_environment,
                "jaeger_endpoint": self.monitoring.jaeger_endpoint,
                "sample_rate": self.monitoring.tracing_sample_rate,
            },
            "health_checks": {
                "enabled": self.monitoring.health_check_enabled,
                "endpoint": self.monitoring.health_check_endpoint,
                "interval": self.monitoring.health_check_interval,
                "timeout": self.monitoring.health_check_timeout,
            },
            "logging": {
                "level": self.logging.root_log_level,
                "format": self.logging.log_format,
                "json_logging": self.logging.json_logging,
                "elasticsearch_enabled": self.logging.elasticsearch_enabled,
                "elasticsearch_url": self.logging.elasticsearch_url,
            },
        }

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Create production configuration from environment variables."""
        return cls()

    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"ProductionConfig(environment={self.environment}, version={self.app_version})"


# Export the production configuration
production_config = ProductionConfig()

# Configuration validation on import
if __name__ == "__main__":
    try:
        config = ProductionConfig()
        print("Production configuration validated successfully!")
        print(f"Environment: {config.environment}")
        print(f"App Version: {config.app_version}")
        print(f"Database URL: {'*' * 20}")  # Masked for security
        print(f"Redis URL: {'*' * 20}")  # Masked for security
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        exit(1)
