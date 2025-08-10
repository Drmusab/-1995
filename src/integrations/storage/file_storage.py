"""
Advanced File Storage System for AI Assistant
Author: Drmusab
Last Modified: 2025-06-20 01:37:54 UTC

This module provides comprehensive file storage capabilities with multiple backends,
encryption, caching, and seamless integration with all core system components.
"""

import base64
import hashlib
import json
import logging
import mimetypes
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Union,
)

import aiofiles
import aiofiles.os
import asyncio

# Assistant components
from src.assistant.core import EnhancedComponentManager
from src.assistant.core import EnhancedSessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CacheHit,
    CacheMiss,
    FileAccessDenied,
    FileDeleted,
    FileDownloaded,
    FileUploaded,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager

# Integrations
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.redis_cache import RedisCache
from src.integrations.storage.backup_manager import BackupManager
from src.integrations.storage.database import DatabaseManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Optional cloud storage imports (graceful fallback)
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.core.exceptions import AzureError
    from azure.storage.blob import BlobServiceClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    from google.cloud.exceptions import GoogleCloudError

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class StorageBackend(Enum):
    """Types of storage backends."""

    LOCAL = "local"
    NETWORK = "network"
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GOOGLE_CLOUD = "google_cloud"
    FTP = "ftp"
    SFTP = "sftp"
    MEMORY = "memory"
    HYBRID = "hybrid"


class FileOperation(Enum):
    """Types of file operations."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    COPY = "copy"
    MOVE = "move"
    LIST = "list"
    EXISTS = "exists"
    METADATA = "metadata"
    BACKUP = "backup"
    RESTORE = "restore"


class StorageClass(Enum):
    """Storage classes for different use cases."""

    HOT = "hot"  # Frequently accessed
    WARM = "warm"  # Occasionally accessed
    COLD = "cold"  # Rarely accessed
    ARCHIVE = "archive"  # Long-term storage
    TEMPORARY = "temporary"  # Short-term storage


class FileStatus(Enum):
    """File status states."""

    UPLOADING = "uploading"
    AVAILABLE = "available"
    PROCESSING = "processing"
    CORRUPTED = "corrupted"
    EXPIRED = "expired"
    DELETED = "deleted"
    ARCHIVED = "archived"
    QUARANTINED = "quarantined"


@dataclass
class FileMetadata:
    """Comprehensive file metadata."""

    file_id: str
    filename: str
    original_name: str
    mime_type: str
    size_bytes: int

    # Storage information
    storage_backend: StorageBackend
    storage_path: str
    storage_class: StorageClass = StorageClass.HOT

    # Security and access
    owner_id: Optional[str] = None
    permissions: Dict[str, Set[str]] = field(default_factory=dict)
    encryption_enabled: bool = False
    encryption_key_id: Optional[str] = None

    # Content information
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    content_encoding: Optional[str] = None
    content_language: Optional[str] = None

    # Versioning
    version: int = 1
    is_latest: bool = True
    parent_file_id: Optional[str] = None

    # Lifecycle
    status: FileStatus = FileStatus.AVAILABLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    # Performance and caching
    access_count: int = 0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600

    # Business context
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageConfiguration:
    """Storage system configuration."""

    # Backend settings
    backend: StorageBackend = StorageBackend.LOCAL
    primary_backend: StorageBackend = StorageBackend.LOCAL
    fallback_backends: List[StorageBackend] = field(default_factory=list)

    # Local storage settings
    local_base_path: str = "data/files"
    local_temp_path: str = "data/temp"

    # Cloud storage settings
    aws_bucket: Optional[str] = None
    aws_region: str = "us-east-1"
    azure_container: Optional[str] = None
    gcs_bucket: Optional[str] = None

    # Security settings
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    auto_backup: bool = True
    backup_interval_hours: int = 24

    # Performance settings
    chunk_size_bytes: int = 8192
    max_file_size_mb: float = 100.0
    concurrent_operations: int = 10
    cache_enabled: bool = True

    # Lifecycle settings
    default_storage_class: StorageClass = StorageClass.HOT
    auto_archive_days: int = 90
    auto_delete_days: int = 365
    cleanup_temp_files: bool = True

    # Monitoring settings
    enable_metrics: bool = True
    enable_audit_logging: bool = True
    health_check_interval: int = 300


class StorageError(Exception):
    """Custom exception for storage operations."""

    def __init__(
        self,
        message: str,
        file_id: Optional[str] = None,
        operation: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        super().__init__(message)
        self.file_id = file_id
        self.operation = operation
        self.backend = backend
        self.timestamp = datetime.now(timezone.utc)


class StorageBackendInterface(Protocol):
    """Interface for storage backends."""

    async def store_file(self, file_path: str, content: bytes, metadata: FileMetadata) -> str:
        """Store a file and return storage path."""
        ...

    async def retrieve_file(self, storage_path: str) -> bytes:
        """Retrieve file content."""
        ...

    async def delete_file(self, storage_path: str) -> bool:
        """Delete a file."""
        ...

    async def exists(self, storage_path: str) -> bool:
        """Check if file exists."""
        ...

    async def get_file_info(self, storage_path: str) -> Dict[str, Any]:
        """Get file information."""
        ...

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix."""
        ...


class LocalStorageBackend:
    """Local filesystem storage backend."""

    def __init__(self, base_path: str, logger):
        self.base_path = Path(base_path)
        self.logger = logger
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def store_file(self, file_path: str, content: bytes, metadata: FileMetadata) -> str:
        """Store file to local filesystem."""
        storage_path = self.base_path / file_path
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(storage_path, "wb") as f:
            await f.write(content)

        return str(storage_path)

    async def retrieve_file(self, storage_path: str) -> bytes:
        """Retrieve file from local filesystem."""
        async with aiofiles.open(storage_path, "rb") as f:
            return await f.read()

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            await aiofiles.os.remove(storage_path)
            return True
        except FileNotFoundError:
            return False

    async def exists(self, storage_path: str) -> bool:
        """Check if file exists on local filesystem."""
        return Path(storage_path).exists()

    async def get_file_info(self, storage_path: str) -> Dict[str, Any]:
        """Get file information from local filesystem."""
        path = Path(storage_path)
        if not path.exists():
            raise StorageError(f"File not found: {storage_path}")

        stat = path.stat()
        return {
            "size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc),
            "created_time": datetime.fromtimestamp(stat.st_ctime, timezone.utc),
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
        }

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix."""
        search_path = self.base_path
        if prefix:
            search_path = search_path / prefix

        files = []
        if search_path.exists():
            for item in search_path.rglob("*"):
                if item.is_file():
                    files.append(str(item.relative_to(self.base_path)))

        return files


class S3StorageBackend:
    """AWS S3 storage backend."""

    def __init__(self, bucket_name: str, region: str, logger):
        if not AWS_AVAILABLE:
            raise StorageError("AWS SDK not available. Install boto3.")

        self.bucket_name = bucket_name
        self.region = region
        self.logger = logger

        try:
            self.s3_client = boto3.client("s3", region_name=region)
            # Verify bucket access
            self.s3_client.head_bucket(Bucket=bucket_name)
        except Exception as e:
            raise StorageError(f"Failed to initialize S3 backend: {str(e)}")

    async def store_file(self, file_path: str, content: bytes, metadata: FileMetadata) -> str:
        """Store file to S3."""
        try:
            extra_args = {
                "Metadata": {
                    "file-id": metadata.file_id,
                    "original-name": metadata.original_name,
                    "owner-id": metadata.owner_id or "",
                    "session-id": metadata.session_id or "",
                }
            }

            # Add server-side encryption if enabled
            if metadata.encryption_enabled:
                extra_args["ServerSideEncryption"] = "AES256"

            # Upload file
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name, Key=file_path, Body=content, **extra_args
                ),
            )

            return f"s3://{self.bucket_name}/{file_path}"

        except (BotoCoreError, ClientError) as e:
            raise StorageError(f"S3 storage failed: {str(e)}", metadata.file_id, "store", "s3")

    async def retrieve_file(self, storage_path: str) -> bytes:
        """Retrieve file from S3."""
        try:
            # Extract key from S3 path
            key = storage_path.replace(f"s3://{self.bucket_name}/", "")

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            )

            return response["Body"].read()

        except (BotoCoreError, ClientError) as e:
            raise StorageError(f"S3 retrieval failed: {str(e)}", operation="retrieve", backend="s3")

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from S3."""
        try:
            key = storage_path.replace(f"s3://{self.bucket_name}/", "")

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            )

            return True

        except (BotoCoreError, ClientError):
            return False

    async def exists(self, storage_path: str) -> bool:
        """Check if file exists in S3."""
        try:
            key = storage_path.replace(f"s3://{self.bucket_name}/", "")

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            )

            return True

        except (BotoCoreError, ClientError):
            return False

    async def get_file_info(self, storage_path: str) -> Dict[str, Any]:
        """Get file information from S3."""
        try:
            key = storage_path.replace(f"s3://{self.bucket_name}/", "")

            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            )

            return {
                "size": response.get("ContentLength", 0),
                "modified_time": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
                "metadata": response.get("Metadata", {}),
                "storage_class": response.get("StorageClass", "STANDARD"),
            }

        except (BotoCoreError, ClientError) as e:
            raise StorageError(f"S3 file info failed: {str(e)}")

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3 with optional prefix."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            )

            files = []
            for obj in response.get("Contents", []):
                files.append(obj["Key"])

            return files

        except (BotoCoreError, ClientError) as e:
            raise StorageError(f"S3 list failed: {str(e)}")


class MemoryStorageBackend:
    """In-memory storage backend for testing and temporary storage."""

    def __init__(self, logger):
        self.storage: Dict[str, bytes] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = logger
        self.lock = threading.Lock()

    async def store_file(self, file_path: str, content: bytes, metadata: FileMetadata) -> str:
        """Store file in memory."""
        with self.lock:
            self.storage[file_path] = content
            self.metadata[file_path] = {
                "size": len(content),
                "stored_at": datetime.now(timezone.utc),
                "metadata": asdict(metadata),
            }

        return f"memory://{file_path}"

    async def retrieve_file(self, storage_path: str) -> bytes:
        """Retrieve file from memory."""
        file_path = storage_path.replace("memory://", "")

        with self.lock:
            if file_path not in self.storage:
                raise StorageError(f"File not found in memory: {file_path}")

            return self.storage[file_path]

    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from memory."""
        file_path = storage_path.replace("memory://", "")

        with self.lock:
            if file_path in self.storage:
                del self.storage[file_path]
                del self.metadata[file_path]
                return True

            return False

    async def exists(self, storage_path: str) -> bool:
        """Check if file exists in memory."""
        file_path = storage_path.replace("memory://", "")

        with self.lock:
            return file_path in self.storage

    async def get_file_info(self, storage_path: str) -> Dict[str, Any]:
        """Get file information from memory."""
        file_path = storage_path.replace("memory://", "")

        with self.lock:
            if file_path not in self.metadata:
                raise StorageError(f"File not found in memory: {file_path}")

            return self.metadata[file_path]

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in memory with optional prefix."""
        with self.lock:
            files = [path for path in self.storage.keys() if path.startswith(prefix)]

            return files


class FileStorageManager:
    """
    Advanced File Storage Management System for the AI Assistant.

    This manager provides comprehensive file storage capabilities including:
    - Multiple storage backend support (local, cloud, memory)
    - Encryption and security integration
    - Intelligent caching and performance optimization
    - Event-driven file operations
    - Health monitoring and metrics
    - Session and workflow integration
    - Automatic backup and recovery
    - File versioning and lifecycle management
    - Error handling and retry mechanisms
    - Resource management and cleanup
    """

    def __init__(self, container: Container):
        """
        Initialize the file storage manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core services
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.health_check = container.get(HealthCheck)

        # Assistant components
        self.component_manager = container.get(EnhancedComponentManager)
        self.session_manager = container.get(EnhancedSessionManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)

        # Security and encryption
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.encryption_manager = None

        # Storage and caching
        try:
            self.database = container.get(DatabaseManager)
            self.cache = container.get(RedisCache)
            self.backup_manager = container.get(BackupManager)
        except Exception:
            self.database = None
            self.cache = None
            self.backup_manager = None

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Configuration
        self.config_data = self._load_configuration()

        # Storage backends
        self.backends: Dict[StorageBackend, StorageBackendInterface] = {}
        self.primary_backend: StorageBackendInterface = None

        # File tracking
        self.file_metadata: Dict[str, FileMetadata] = {}
        self.file_locks: Dict[str, asyncio.Lock] = {}

        # Performance tracking
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        self.transfer_stats: Dict[str, List[float]] = {}

        # Resource management
        self.semaphore = asyncio.Semaphore(self.config_data.concurrent_operations)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config_data.concurrent_operations, thread_name_prefix="file_storage"
        )

        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.backup_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None

        # Initialize storage system
        self._setup_backends()
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("file_storage", self._health_check_callback)

        self.logger.info("FileStorageManager initialized successfully")

    def _load_configuration(self) -> StorageConfiguration:
        """Load storage configuration from config system."""
        storage_config = self.config.get("storage", {})

        return StorageConfiguration(
            backend=StorageBackend(storage_config.get("backend", "local")),
            primary_backend=StorageBackend(storage_config.get("primary_backend", "local")),
            fallback_backends=[
                StorageBackend(b) for b in storage_config.get("fallback_backends", [])
            ],
            local_base_path=storage_config.get("local_base_path", "data/files"),
            local_temp_path=storage_config.get("local_temp_path", "data/temp"),
            aws_bucket=storage_config.get("aws_bucket"),
            aws_region=storage_config.get("aws_region", "us-east-1"),
            azure_container=storage_config.get("azure_container"),
            gcs_bucket=storage_config.get("gcs_bucket"),
            encryption_enabled=storage_config.get("encryption_enabled", True),
            auto_backup=storage_config.get("auto_backup", True),
            backup_interval_hours=storage_config.get("backup_interval_hours", 24),
            chunk_size_bytes=storage_config.get("chunk_size_bytes", 8192),
            max_file_size_mb=storage_config.get("max_file_size_mb", 100.0),
            concurrent_operations=storage_config.get("concurrent_operations", 10),
            cache_enabled=storage_config.get("cache_enabled", True),
            default_storage_class=StorageClass(storage_config.get("default_storage_class", "hot")),
            auto_archive_days=storage_config.get("auto_archive_days", 90),
            auto_delete_days=storage_config.get("auto_delete_days", 365),
            cleanup_temp_files=storage_config.get("cleanup_temp_files", True),
            enable_metrics=storage_config.get("enable_metrics", True),
            enable_audit_logging=storage_config.get("enable_audit_logging", True),
            health_check_interval=storage_config.get("health_check_interval", 300),
        )

    def _setup_backends(self) -> None:
        """Setup storage backends based on configuration."""
        try:
            # Local storage (always available)
            self.backends[StorageBackend.LOCAL] = LocalStorageBackend(
                self.config_data.local_base_path, self.logger
            )

            # Memory storage (for testing)
            self.backends[StorageBackend.MEMORY] = MemoryStorageBackend(self.logger)

            # AWS S3
            if self.config_data.aws_bucket and AWS_AVAILABLE:
                try:
                    self.backends[StorageBackend.AWS_S3] = S3StorageBackend(
                        self.config_data.aws_bucket, self.config_data.aws_region, self.logger
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to setup S3 backend: {str(e)}")

            # Set primary backend
            if self.config_data.primary_backend in self.backends:
                self.primary_backend = self.backends[self.config_data.primary_backend]
            else:
                self.primary_backend = self.backends[StorageBackend.LOCAL]
                self.logger.warning(
                    f"Primary backend {self.config_data.primary_backend} not available, using local"
                )

            self.logger.info(f"Initialized {len(self.backends)} storage backends")

        except Exception as e:
            self.logger.error(f"Failed to setup storage backends: {str(e)}")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register file storage metrics
            self.metrics.register_counter("files_uploaded_total")
            self.metrics.register_counter("files_downloaded_total")
            self.metrics.register_counter("files_deleted_total")
            self.metrics.register_histogram("file_operation_duration_seconds")
            self.metrics.register_histogram("file_transfer_speed_mbps")
            self.metrics.register_gauge("active_file_operations")
            self.metrics.register_gauge("storage_usage_bytes")
            self.metrics.register_counter("storage_errors_total")
            self.metrics.register_counter("cache_hits_total")
            self.metrics.register_counter("cache_misses_total")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the file storage manager."""
        try:
            # Initialize storage directories
            for backend_type, backend in self.backends.items():
                if hasattr(backend, "initialize"):
                    await backend.initialize()

            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())

            if self.config_data.auto_backup and self.backup_manager:
                self.backup_task = asyncio.create_task(self._backup_loop())

            # Register event handlers
            await self._register_event_handlers()

            # Load existing file metadata
            await self._load_file_metadata()

            self.logger.info("FileStorageManager initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize FileStorageManager: {str(e)}")
            raise StorageError(f"Initialization failed: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # Session events
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

        # Workflow events
        self.event_bus.subscribe("workflow_completed", self._handle_workflow_completed)

        # Component health events
        self.event_bus.subscribe("component_health_changed", self._handle_component_health_change)

        # System events
        self.event_bus.subscribe("system_shutdown_started", self._handle_system_shutdown)

    async def _load_file_metadata(self) -> None:
        """Load file metadata from database."""
        try:
            if self.database:
                # Load file metadata from database
                rows = await self.database.fetch_all(
                    "SELECT file_id, metadata FROM file_storage WHERE status != 'deleted'"
                )

                for row in rows:
                    file_id, metadata_json = row
                    try:
                        metadata_dict = json.loads(metadata_json)
                        metadata = self._deserialize_metadata(metadata_dict)
                        self.file_metadata[file_id] = metadata
                    except Exception as e:
                        self.logger.warning(f"Failed to load metadata for file {file_id}: {str(e)}")

            self.logger.info(f"Loaded metadata for {len(self.file_metadata)} files")

        except Exception as e:
            self.logger.warning(f"Failed to load file metadata: {str(e)}")

    def _get_file_lock(self, file_id: str) -> asyncio.Lock:
        """Get or create a lock for file operations."""
        if file_id not in self.file_locks:
            self.file_locks[file_id] = asyncio.Lock()
        return self.file_locks[file_id]

    @handle_exceptions
    async def store_file(
        self,
        content: Union[bytes, BinaryIO, str],
        filename: str,
        mime_type: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        storage_class: StorageClass = StorageClass.HOT,
        metadata: Optional[Dict[str, Any]] = None,
        encryption_enabled: Optional[bool] = None,
    ) -> str:
        """
        Store a file in the storage system.

        Args:
            content: File content (bytes, file-like object, or file path)
            filename: Original filename
            mime_type: MIME type (auto-detected if not provided)
            session_id: Associated session ID
            workflow_id: Associated workflow ID
            owner_id: File owner ID
            storage_class: Storage class for the file
            metadata: Custom metadata
            encryption_enabled: Override encryption setting

        Returns:
            File ID for the stored file
        """
        async with self.semaphore:
            start_time = time.time()
            file_id = str(uuid.uuid4())

            try:
                with self.tracer.trace("file_store") as span:
                    span.set_attributes(
                        {
                            "file_id": file_id,
                            "filename": filename,
                            "session_id": session_id or "unknown",
                            "workflow_id": workflow_id or "unknown",
                            "storage_class": storage_class.value,
                        }
                    )

                    # Process content
                    if isinstance(content, str):
                        # Assume it's a file path
                        async with aiofiles.open(content, "rb") as f:
                            file_content = await f.read()
                    elif hasattr(content, "read"):
                        # File-like object
                        file_content = content.read()
                        if isinstance(file_content, str):
                            file_content = file_content.encode()
                    else:
                        # Assume bytes
                        file_content = content

                    # Validate file size
                    file_size = len(file_content)
                    max_size_bytes = self.config_data.max_file_size_mb * 1024 * 1024
                    if file_size > max_size_bytes:
                        raise StorageError(
                            f"File size {file_size} exceeds maximum {max_size_bytes}"
                        )

                    # Auto-detect MIME type
                    if not mime_type:
                        mime_type, _ = mimetypes.guess_type(filename)
                        if not mime_type:
                            mime_type = "application/octet-stream"

                    # Create file metadata
                    file_metadata = FileMetadata(
                        file_id=file_id,
                        filename=self._sanitize_filename(filename),
                        original_name=filename,
                        mime_type=mime_type,
                        size_bytes=file_size,
                        storage_backend=self.config_data.primary_backend,
                        storage_path="",  # Will be set after storage
                        storage_class=storage_class,
                        owner_id=owner_id,
                        session_id=session_id,
                        workflow_id=workflow_id,
                        encryption_enabled=(
                            encryption_enabled
                            if encryption_enabled is not None
                            else self.config_data.encryption_enabled
                        ),
                        custom_metadata=metadata or {},
                    )

                    # Calculate checksums
                    file_metadata.checksum_md5 = hashlib.md5(file_content).hexdigest()
                    file_metadata.checksum_sha256 = hashlib.sha256(file_content).hexdigest()

                    # Encrypt content if enabled
                    if file_metadata.encryption_enabled and self.encryption_manager:
                        encrypted_content, key_id = await self.encryption_manager.encrypt_data(
                            file_content
                        )
                        file_content = encrypted_content
                        file_metadata.encryption_key_id = key_id

                    # Generate storage path
                    storage_path = self._generate_storage_path(file_metadata)

                    # Store file using primary backend
                    try:
                        actual_storage_path = await self.primary_backend.store_file(
                            storage_path, file_content, file_metadata
                        )
                        file_metadata.storage_path = actual_storage_path

                    except Exception as e:
                        # Try fallback backends
                        stored = False
                        for fallback_backend_type in self.config_data.fallback_backends:
                            if fallback_backend_type in self.backends:
                                try:
                                    fallback_backend = self.backends[fallback_backend_type]
                                    actual_storage_path = await fallback_backend.store_file(
                                        storage_path, file_content, file_metadata
                                    )
                                    file_metadata.storage_path = actual_storage_path
                                    file_metadata.storage_backend = fallback_backend_type
                                    stored = True
                                    break
                                except Exception as fallback_error:
                                    self.logger.warning(
                                        f"Fallback backend {fallback_backend_type} failed: {str(fallback_error)}"
                                    )

                        if not stored:
                            raise StorageError(f"Failed to store file with all backends: {str(e)}")

                    # Store metadata
                    async with self._get_file_lock(file_id):
                        self.file_metadata[file_id] = file_metadata

                        # Persist to database
                        if self.database:
                            await self._persist_file_metadata(file_metadata)

                        # Cache metadata
                        if self.cache and self.config_data.cache_enabled:
                            await self.cache.set(
                                f"file_metadata:{file_id}",
                                json.dumps(self._serialize_metadata(file_metadata)),
                                ttl=file_metadata.cache_ttl_seconds,
                            )

                    # Update metrics
                    processing_time = time.time() - start_time
                    transfer_speed = (file_size / 1024 / 1024) / max(processing_time, 0.001)  # MB/s

                    self.metrics.increment("files_uploaded_total")
                    self.metrics.record("file_operation_duration_seconds", processing_time)
                    self.metrics.record("file_transfer_speed_mbps", transfer_speed)

                    # Emit event
                    await self.event_bus.emit(
                        FileUploaded(
                            file_id=file_id,
                            filename=filename,
                            size_bytes=file_size,
                            session_id=session_id,
                            owner_id=owner_id,
                            storage_backend=file_metadata.storage_backend.value,
                        )
                    )

                    self.logger.info(
                        f"Stored file {file_id} ({filename}) - {file_size} bytes in {processing_time:.2f}s"
                    )

                    return file_id

            except Exception as e:
                self.metrics.increment("storage_errors_total")
                self.logger.error(f"Failed to store file {filename}: {str(e)}")
                raise StorageError(f"File storage failed: {str(e)}", file_id, "store")

    @handle_exceptions
    async def retrieve_file(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        decrypt: bool = True,
    ) -> bytes:
        """
        Retrieve a file from storage.

        Args:
            file_id: File identifier
            user_id: User requesting the file
            session_id: Session ID for context
            decrypt: Whether to decrypt the file content

        Returns:
            File content as bytes
        """
        async with self.semaphore:
            start_time = time.time()

            try:
                with self.tracer.trace("file_retrieve") as span:
                    span.set_attributes(
                        {
                            "file_id": file_id,
                            "user_id": user_id or "unknown",
                            "session_id": session_id or "unknown",
                        }
                    )

                    # Get file metadata
                    file_metadata = await self._get_file_metadata(file_id)
                    if not file_metadata:
                        raise StorageError(f"File {file_id} not found")

                    # Check permissions
                    if user_id and not await self._check_file_access(
                        file_metadata, user_id, FileOperation.READ
                    ):
                        await self.event_bus.emit(
                            FileAccessDenied(file_id=file_id, user_id=user_id, operation="read")
                        )
                        raise StorageError(f"Access denied for file {file_id}")

                    # Update access tracking
                    file_metadata.access_count += 1
                    file_metadata.accessed_at = datetime.now(timezone.utc)

                    # Try to get from cache first
                    cache_key = f"file_content:{file_id}"
                    if self.cache and self.config_data.cache_enabled:
                        cached_content = await self.cache.get(cache_key)
                        if cached_content:
                            await self.event_bus.emit(
                                CacheHit(key=cache_key, cache_type="file_content")
                            )

                            # Decode from base64
                            content = base64.b64decode(cached_content.encode())

                            # Decrypt if needed
                            if (
                                decrypt
                                and file_metadata.encryption_enabled
                                and self.encryption_manager
                            ):
                                content = await self.encryption_manager.decrypt_data(
                                    content, file_metadata.encryption_key_id
                                )

                            return content
                        else:
                            await self.event_bus.emit(
                                CacheMiss(key=cache_key, cache_type="file_content")
                            )

                    # Get backend for this file
                    backend = self.backends.get(file_metadata.storage_backend)
                    if not backend:
                        raise StorageError(f"Backend {file_metadata.storage_backend} not available")

                    # Retrieve file content
                    content = await backend.retrieve_file(file_metadata.storage_path)

                    # Decrypt if enabled and requested
                    if decrypt and file_metadata.encryption_enabled and self.encryption_manager:
                        content = await self.encryption_manager.decrypt_data(
                            content, file_metadata.encryption_key_id
                        )

                    # Cache the content
                    if self.cache and self.config_data.cache_enabled:
                        # Encode to base64 for caching
                        encoded_content = base64.b64encode(content).decode()
                        await self.cache.set(
                            cache_key, encoded_content, ttl=file_metadata.cache_ttl_seconds
                        )

                    # Update metrics
                    processing_time = time.time() - start_time
                    transfer_speed = (file_metadata.size_bytes / 1024 / 1024) / max(
                        processing_time, 0.001
                    )

                    self.metrics.increment("files_downloaded_total")
                    self.metrics.record("file_operation_duration_seconds", processing_time)
                    self.metrics.record("file_transfer_speed_mbps", transfer_speed)

                    # Emit event
                    await self.event_bus.emit(
                        FileDownloaded(
                            file_id=file_id,
                            filename=file_metadata.filename,
                            size_bytes=file_metadata.size_bytes,
                            user_id=user_id,
                            session_id=session_id,
                        )
                    )

                    # Update metadata in background
                    asyncio.create_task(self._update_file_metadata(file_metadata))

                    self.logger.debug(
                        f"Retrieved file {file_id} - {file_metadata.size_bytes} bytes in {processing_time:.2f}s"
                    )

                    return content

            except Exception as e:
                self.metrics.increment("storage_errors_total")
                self.logger.error(f"Failed to retrieve file {file_id}: {str(e)}")
                raise StorageError(f"File retrieval failed: {str(e)}", file_id, "retrieve")

    @handle_exceptions
    async def delete_file(
        self,
        file_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        permanent: bool = False,
    ) -> bool:
        """
        Delete a file from storage.

        Args:
            file_id: File identifier
            user_id: User requesting deletion
            session_id: Session ID for context
            permanent: Whether to permanently delete or mark as deleted

        Returns:
            True if successful
        """
        start_time = time.time()

        try:
            with self.tracer.trace("file_delete") as span:
                span.set_attributes(
                    {
                        "file_id": file_id,
                        "user_id": user_id or "unknown",
                        "session_id": session_id or "unknown",
                        "permanent": permanent,
                    }
                )

                # Get file metadata
                file_metadata = await self._get_file_metadata(file_id)
                if not file_metadata:
                    return False  # Already deleted or doesn't exist

                # Check permissions
                if user_id and not await self._check_file_access(
                    file_metadata, user_id, FileOperation.DELETE
                ):
                    await self.event_bus.emit(
                        FileAccessDenied(file_id=file_id, user_id=user_id, operation="delete")
                    )
                    raise StorageError(f"Access denied for file deletion {file_id}")

                async with self._get_file_lock(file_id):
                    if permanent:
                        # Permanently delete from storage backend
                        backend = self.backends.get(file_metadata.storage_backend)
                        if backend:
                            await backend.delete_file(file_metadata.storage_path)

                        # Remove from metadata
                        self.file_metadata.pop(file_id, None)

                        # Remove from database
                        if self.database:
                            await self.database.execute(
                                "DELETE FROM file_storage WHERE file_id = ?", (file_id,)
                            )
                    else:
                        # Mark as deleted
                        file_metadata.status = FileStatus.DELETED
                        file_metadata.modified_at = datetime.now(timezone.utc)

                        # Update in database
                        if self.database:
                            await self._persist_file_metadata(file_metadata)

                    # Remove from cache
                    if self.cache:
                        await self.cache.delete(f"file_metadata:{file_id}")
                        await self.cache.delete(f"file_content:{file_id}")

                # Update metrics
                processing_time = time.time() - start_time
                self.metrics.increment("files_deleted_total")
                self.metrics.record("file_operation_duration_seconds", processing_time)

                # Emit event
                await self.event_bus.emit(
                    FileDeleted(
                        file_id=file_id,
                        filename=file_metadata.filename,
                        user_id=user_id,
                        session_id=session_id,
                        permanent=permanent,
                    )
                )

                self.logger.info(f"Deleted file {file_id} (permanent: {permanent})")
                return True

        except Exception as e:
            self.metrics.increment("storage_errors_total")
            self.logger.error(f"Failed to delete file {file_id}: {str(e)}")
            raise StorageError(f"File deletion failed: {str(e)}", file_id, "delete")

    async def _get_file_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata with caching."""
        # Try memory first
        if file_id in self.file_metadata:
            return self.file_metadata[file_id]

        # Try cache
        if self.cache and self.config_data.cache_enabled:
            cached_data = await self.cache.get(f"file_metadata:{file_id}")
            if cached_data:
                try:
                    metadata_dict = json.loads(cached_data)
                    metadata = self._deserialize_metadata(metadata_dict)
                    self.file_metadata[file_id] = metadata
                    return metadata
                except Exception as e:
                    self.logger.warning(
                        f"Failed to deserialize cached metadata for {file_id}: {str(e)}"
                    )

        # Try database
        if self.database:
            try:
                row = await self.database.fetch_one(
                    "SELECT metadata FROM file_storage WHERE file_id = ?", (file_id,)
                )
                if row:
                    metadata_dict = json.loads(row[0])
                    metadata = self._deserialize_metadata(metadata_dict)
                    self.file_metadata[file_id] = metadata
                    return metadata
            except Exception as e:
                self.logger.warning(
                    f"Failed to load metadata from database for {file_id}: {str(e)}"
                )

        return None

    async def _check_file_access(
        self, file_metadata: FileMetadata, user_id: str, operation: FileOperation
    ) -> bool:
        """Check if user has access to perform operation on file."""
        # Owner always has access
        if file_metadata.owner_id == user_id:
            return True

        # Use authorization manager if available
        if self.authz_manager:
            try:
                return await self.authz_manager.check_permission(
                    user_id, "file", operation.value, resource_id=file_metadata.file_id
                )
            except Exception as e:
                self.logger.warning(f"Authorization check failed: {str(e)}")

        # Check permissions in metadata
        user_permissions = file_metadata.permissions.get(user_id, set())

        permission_map = {
            FileOperation.READ: "read",
            FileOperation.WRITE: "write",
            FileOperation.DELETE: "delete",
            FileOperation.COPY: "read",
            FileOperation.MOVE: "write",
        }

        required_permission = permission_map.get(operation, "read")
        return required_permission in user_permissions

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, "_")

        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            max_name_len = 255 - len(ext)
            filename = name[:max_name_len] + ext

        return filename

    def _generate_storage_path(self, metadata: FileMetadata) -> str:
        """Generate storage path for file."""
        # Create path based on date and file ID for organization
        date_path = metadata.created_at.strftime("%Y/%m/%d")
        return f"{date_path}/{metadata.file_id}/{metadata.filename}"

    def _serialize_metadata(self, metadata: FileMetadata) -> Dict[str, Any]:
        """Serialize file metadata to dictionary."""
        data = asdict(metadata)

        # Convert datetime objects to ISO strings
        datetime_fields = ["created_at", "modified_at", "accessed_at", "expires_at"]
        for field in datetime_fields:
            if data.get(field):
                data[field] = data[field].isoformat()

        # Convert enums to values
        data["storage_backend"] = data["storage_backend"].value
        data["storage_class"] = data["storage_class"].value
        data["status"] = data["status"].value

        # Convert sets to lists
        data["tags"] = list(data["tags"])
        for user_id, perms in data["permissions"].items():
            data["permissions"][user_id] = list(perms)

        return data

    def _deserialize_metadata(self, data: Dict[str, Any]) -> FileMetadata:
        """Deserialize file metadata from dictionary."""
        # Convert datetime strings back to datetime objects
        datetime_fields = ["created_at", "modified_at", "accessed_at", "expires_at"]
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])

        # Convert enum values back to enums
        data["storage_backend"] = StorageBackend(data["storage_backend"])
        data["storage_class"] = StorageClass(data["storage_class"])
        data["status"] = FileStatus(data["status"])

        # Convert lists back to sets
        data["tags"] = set(data["tags"])
        for user_id, perms in data["permissions"].items():
            data["permissions"][user_id] = set(perms)

        return FileMetadata(**data)

    async def _persist_file_metadata(self, metadata: FileMetadata) -> None:
        """Persist file metadata to database."""
        if not self.database:
            return

        try:
            metadata_json = json.dumps(self._serialize_metadata(metadata))

            await self.database.execute(
                """
                INSERT OR REPLACE INTO file_storage 
                (file_id, filename, owner_id, session_id, workflow_id, size_bytes, 
                 mime_type, storage_backend, storage_path, status, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.file_id,
                    metadata.filename,
                    metadata.owner_id,
                    metadata.session_id,
                    metadata.workflow_id,
                    metadata.size_bytes,
                    metadata.mime_type,
                    metadata.storage_backend.value,
                    metadata.storage_path,
                    metadata.status.value,
                    metadata.created_at,
                    metadata_json,
                ),
            )

        except Exception as e:
            self.logger.error(f"Failed to persist metadata for {metadata.file_id}: {str(e)}")

    async def _update_file_metadata(self, metadata: FileMetadata) -> None:
        """Update file metadata in all storage locations."""
        try:
            # Update in memory
            self.file_metadata[metadata.file_id] = metadata

            # Update in database
            await self._persist_file_metadata(metadata)

            # Update in cache
            if self.cache and self.config_data.cache_enabled:
                await self.cache.set(
                    f"file_metadata:{metadata.file_id}",
                    json.dumps(self._serialize_metadata(metadata)),
                    ttl=metadata.cache_ttl_seconds,
                )

        except Exception as e:
            self.logger.warning(f"Failed to update metadata for {metadata.file_id}: {str(e)}")

    def list_files(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        status: Optional[FileStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List files with filtering options.

        Args:
            user_id: Filter by owner
            session_id: Filter by session
            workflow_id: Filter by workflow
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of file information dictionaries
        """
        files = []

        for metadata in self.file_metadata.values():
            # Apply filters
            if user_id and metadata.owner_id != user_id:
                continue
            if session_id and metadata.session_id != session_id:
                continue
            if workflow_id and metadata.workflow_id != workflow_id:
                continue
            if status and metadata.status != status:
                continue

            files.append(
                {
                    "file_id": metadata.file_id,
                    "filename": metadata.filename,
                    "original_name": metadata.original_name,
                    "mime_type": metadata.mime_type,
                    "size_bytes": metadata.size_bytes,
                    "owner_id": metadata.owner_id,
                    "session_id": metadata.session_id,
                    "workflow_id": metadata.workflow_id,
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "modified_at": metadata.modified_at.isoformat(),
                    "access_count": metadata.access_count,
                }
            )

        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)

        # Apply pagination
        return files[offset : offset + limit]

    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed file information."""
        metadata = self.file_metadata.get(file_id)
        if not metadata:
            return None

        return {
            "file_id": metadata.file_id,
            "filename": metadata.filename,
            "original_name": metadata.original_name,
            "mime_type": metadata.mime_type,
            "size_bytes": metadata.size_bytes,
            "storage_backend": metadata.storage_backend.value,
            "storage_class": metadata.storage_class.value,
            "owner_id": metadata.owner_id,
            "session_id": metadata.session_id,
            "workflow_id": metadata.workflow_id,
            "status": metadata.status.value,
            "created_at": metadata.created_at.isoformat(),
            "modified_at": metadata.modified_at.isoformat(),
            "accessed_at": metadata.accessed_at.isoformat(),
            "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
            "access_count": metadata.access_count,
            "version": metadata.version,
            "encryption_enabled": metadata.encryption_enabled,
            "checksum_md5": metadata.checksum_md5,
            "checksum_sha256": metadata.checksum_sha256,
            "tags": list(metadata.tags),
            "custom_metadata": metadata.custom_metadata,
        }

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        stats = {
            "total_files": len(self.file_metadata),
            "total_size_bytes": 0,
            "files_by_status": {},
            "files_by_backend": {},
            "files_by_class": {},
            "recent_activity": {},
            "top_owners": {},
            "storage_health": {},
        }

        # Calculate statistics
        for metadata in self.file_metadata.values():
            stats["total_size_bytes"] += metadata.size_bytes

            # By status
            status = metadata.status.value
            stats["files_by_status"][status] = stats["files_by_status"].get(status, 0) + 1

            # By backend
            backend = metadata.storage_backend.value
            stats["files_by_backend"][backend] = stats["files_by_backend"].get(backend, 0) + 1

            # By storage class
            storage_class = metadata.storage_class.value
            stats["files_by_class"][storage_class] = (
                stats["files_by_class"].get(storage_class, 0) + 1
            )

            # Top owners
            if metadata.owner_id:
                stats["top_owners"][metadata.owner_id] = (
                    stats["top_owners"].get(metadata.owner_id, 0) + 1
                )

        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        stats["recent_activity"] = {
            "uploads": len(
                [m for m in self.file_metadata.values() if m.created_at > recent_cutoff]
            ),
            "downloads": len(
                [m for m in self.file_metadata.values() if m.accessed_at > recent_cutoff]
            ),
        }

        # Backend health
        for backend_type, backend in self.backends.items():
            try:
                # Try a simple operation to test health
                files = await backend.list_files()
                stats["storage_health"][backend_type.value] = {
                    "status": "healthy",
                    "file_count": len(files),
                }
            except Exception as e:
                stats["storage_health"][backend_type.value] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return stats

    async def _cleanup_loop(self) -> None:
        """Background task for file cleanup and maintenance."""
        while True:
            try:
                current_time = datetime.now(timezone.utc)

                # Clean up expired files
                expired_files = []
                for file_id, metadata in self.metadata_cache.items():
                    expiry = metadata.get("expires_at")
                    if expiry and current_time > expiry:
                        expired_files.append(file_id)

                # Remove expired files
                for file_id in expired_files:
                    try:
                        await self.delete_file(file_id)
                    except Exception as e:
                        self.logger.error(f"Failed to delete expired file {file_id}: {str(e)}")

                if expired_files:
                    self.logger.info(f"Cleaned up {len(expired_files)} expired files")

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(3600)
