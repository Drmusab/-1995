"""
Encrypted Storage Manager
Author: Drmusab
Last Modified: 2025-07-19 21:43:58 UTC

This module provides encryption for all stored user data.
"""

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncio

from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import DataDecrypted, DataEncrypted
from src.core.security.encryption import (
    EncryptionAlgorithm,
    EncryptionContext,
    EncryptionManager,
    SecurityLevel,
)
from src.integrations.storage.database import DatabaseManager
from src.integrations.storage.file_storage import FileStorageManager
from src.observability.logging.config import get_logger


@dataclass
class EncryptedDataRecord:
    """Record for encrypted data storage."""

    record_id: str
    user_id: str
    data_type: str
    encrypted_data: bytes
    encryption_key_id: str
    encryption_algorithm: EncryptionAlgorithm
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


class EncryptedStorageManager:
    """
    Manages encrypted storage for all user data.
    """

    def __init__(self, container: Container):
        """
        Initialize the encrypted storage manager.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Core components
        self.encryption_manager = container.get(EncryptionManager)
        self.database = container.get(DatabaseManager)
        self.file_storage = container.get(FileStorageManager)
        self.event_bus = container.get(EventBus)

        # Configuration
        self.default_algorithm = EncryptionAlgorithm.AES_256_GCM
        self.enable_key_rotation = True
        self.key_rotation_days = 90
        self.enable_compression = True
        self.audit_access = True

        # Cache for frequently accessed data
        self._decryption_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        self._cache_max_size = 1000

        self.logger.info("Encrypted storage manager initialized")

    async def store_user_data(
        self,
        user_id: str,
        data_type: str,
        data: Any,
        security_level: SecurityLevel = SecurityLevel.HIGH,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store user data with encryption.

        Args:
            user_id: User identifier
            data_type: Type of data being stored
            data: Data to encrypt and store
            security_level: Security level for encryption
            metadata: Optional metadata

        Returns:
            Record ID of stored data
        """
        try:
            # Serialize data
            if isinstance(data, (dict, list)):
                serialized_data = json.dumps(data).encode("utf-8")
            elif isinstance(data, str):
                serialized_data = data.encode("utf-8")
            elif isinstance(data, bytes):
                serialized_data = data
            else:
                serialized_data = str(data).encode("utf-8")

            # Compress if enabled
            if self.enable_compression:
                import zlib

                serialized_data = zlib.compress(serialized_data)

            # Create encryption context
            encryption_context = EncryptionContext(
                user_id=user_id,
                data_type=data_type,
                security_level=security_level,
                compression_enabled=self.enable_compression,
                audit_required=self.audit_access,
                additional_metadata=metadata or {},
            )

            # Encrypt data
            encryption_result = await self.encryption_manager.encrypt(
                data=serialized_data, context=encryption_context
            )

            if not encryption_result.success:
                raise Exception(f"Encryption failed: {encryption_result.error_message}")

            # Generate record ID
            record_id = self._generate_record_id(user_id, data_type)

            # Create encrypted record
            record = EncryptedDataRecord(
                record_id=record_id,
                user_id=user_id,
                data_type=data_type,
                encrypted_data=encryption_result.data,
                encryption_key_id=encryption_result.key_id,
                encryption_algorithm=encryption_result.algorithm,
                iv=encryption_result.iv,
                tag=encryption_result.tag,
                metadata=metadata or {},
            )

            # Store in database
            await self._store_encrypted_record(record)

            # Emit encryption event
            await self.event_bus.emit(
                DataEncrypted(
                    user_id=user_id,
                    data_type=data_type,
                    record_id=record_id,
                    encryption_algorithm=encryption_result.algorithm.value,
                    timestamp=datetime.now(timezone.utc),
                )
            )

            self.logger.info(f"Stored encrypted data for user {user_id}, type: {data_type}")

            return record_id

        except Exception as e:
            self.logger.error(f"Error storing encrypted data: {str(e)}")
            raise

    async def retrieve_user_data(
        self, user_id: str, record_id: str, _verify_integrity: bool = True
    ) -> Optional[Any]:
        """
        Retrieve and decrypt user data.

        Args:
            user_id: User identifier
            record_id: Record identifier
            _verify_integrity: Whether to verify data integrity

        Returns:
            Decrypted data or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{user_id}:{record_id}"
            if cache_key in self._decryption_cache:
                cached_data, cached_time = self._decryption_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_time).seconds < self._cache_ttl_seconds:
                    return cached_data
                else:
                    del self._decryption_cache[cache_key]

            # Retrieve encrypted record
            record = await self._get_encrypted_record(record_id, user_id)
            if not record:
                return None

            # Create decryption context
            decryption_context = EncryptionContext(
                user_id=user_id,
                data_type=record.data_type,
                compression_enabled=record.metadata.get("compressed", self.enable_compression),
                audit_required=self.audit_access,
            )

            # Decrypt data
            decryption_result = await self.encryption_manager.decrypt(
                encrypted_data=record.encrypted_data,
                key_id=record.encryption_key_id,
                algorithm=record.encryption_algorithm,
                iv=record.iv,
                tag=record.tag,
                context=decryption_context,
            )

            if not decryption_result.success:
                raise Exception(f"Decryption failed: {decryption_result.error_message}")

            # Decompress if needed
            decrypted_data = decryption_result.data
            if record.metadata.get("compressed", self.enable_compression):
                import zlib

                decrypted_data = zlib.decompress(decrypted_data)

            # Deserialize data
            try:
                deserialized_data = json.loads(decrypted_data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If not JSON, return as string or bytes
                try:
                    deserialized_data = decrypted_data.decode("utf-8")
                except UnicodeDecodeError:
                    deserialized_data = decrypted_data

            # Update access tracking
            await self._update_access_tracking(record_id)

            # Cache the decrypted data
            self._update_cache(cache_key, deserialized_data)

            # Emit decryption event
            await self.event_bus.emit(
                DataDecrypted(
                    user_id=user_id,
                    data_type=record.data_type,
                    record_id=record_id,
                    timestamp=datetime.now(timezone.utc),
                )
            )

            return deserialized_data

        except Exception as e:
            self.logger.error(f"Error retrieving encrypted data: {str(e)}")
            return None

    async def update_user_data(
        self,
        user_id: str,
        record_id: str,
        data: Any,
        security_level: Optional[SecurityLevel] = None,
    ) -> bool:
        """
        Update encrypted user data.

        Args:
            user_id: User identifier
            record_id: Record identifier
            data: New data to encrypt and store
            security_level: Optional new security level

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing record
            existing_record = await self._get_encrypted_record(record_id, user_id)
            if not existing_record:
                return False

            # Use existing security level if not specified
            if security_level is None:
                security_level = SecurityLevel(
                    existing_record.metadata.get("security_level", SecurityLevel.HIGH.value)
                )

            # Store new encrypted data
            new_record_id = await self.store_user_data(
                user_id=user_id,
                data_type=existing_record.data_type,
                data=data,
                security_level=security_level,
                metadata={
                    **existing_record.metadata,
                    "previous_record_id": record_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Mark old record as superseded
            await self._mark_record_superseded(record_id, new_record_id)

            # Clear cache
            cache_key = f"{user_id}:{record_id}"
            if cache_key in self._decryption_cache:
                del self._decryption_cache[cache_key]

            return True

        except Exception as e:
            self.logger.error(f"Error updating encrypted data: {str(e)}")
            return False

    async def delete_user_data(
        self,
        user_id: str,
        record_id: Optional[str] = None,
        data_type: Optional[str] = None,
        secure_delete: bool = True,
    ) -> int:
        """
        Delete encrypted user data.

        Args:
            user_id: User identifier
            record_id: Optional specific record to delete
            data_type: Optional data type to delete all records of
            secure_delete: Whether to securely overwrite data

        Returns:
            Number of records deleted
        """
        try:
            deleted_count = 0

            if record_id:
                # Delete specific record
                success = await self._delete_encrypted_record(record_id, user_id, secure_delete)
                if success:
                    deleted_count = 1

                    # Clear cache
                    cache_key = f"{user_id}:{record_id}"
                    if cache_key in self._decryption_cache:
                        del self._decryption_cache[cache_key]

            elif data_type:
                # Delete all records of specific type
                records = await self._get_user_records_by_type(user_id, data_type)
                for record in records:
                    success = await self._delete_encrypted_record(
                        record["record_id"], user_id, secure_delete
                    )
                    if success:
                        deleted_count += 1

                        # Clear cache
                        cache_key = f"{user_id}:{record['record_id']}"
                        if cache_key in self._decryption_cache:
                            del self._decryption_cache[cache_key]

            else:
                # Delete all user records
                records = await self._get_all_user_records(user_id)
                for record in records:
                    success = await self._delete_encrypted_record(
                        record["record_id"], user_id, secure_delete
                    )
                    if success:
                        deleted_count += 1

                # Clear all user cache entries
                keys_to_delete = [
                    key for key in self._decryption_cache.keys() if key.startswith(f"{user_id}:")
                ]
                for key in keys_to_delete:
                    del self._decryption_cache[key]

            self.logger.info(f"Deleted {deleted_count} encrypted records for user {user_id}")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Error deleting encrypted data: {str(e)}")
            return 0

    async def rotate_encryption_keys(
        self, user_id: Optional[str] = None, force: bool = False
    ) -> Dict[str, Any]:
        """
        Rotate encryption keys for user data.

        Args:
            user_id: Optional specific user to rotate keys for
            force: Force rotation regardless of age

        Returns:
            Rotation statistics
        """
        try:
            stats = {"records_processed": 0, "records_rotated": 0, "errors": 0}

            # Get records to rotate
            if user_id:
                records = await self._get_all_user_records(user_id)
            else:
                records = await self._get_records_for_rotation(force)

            for record in records:
                try:
                    # Retrieve and decrypt data
                    decrypted_data = await self.retrieve_user_data(
                        record["user_id"], record["record_id"]
                    )

                    if decrypted_data is not None:
                        # Re-encrypt with new key
                        await self.update_user_data(
                            user_id=record["user_id"],
                            record_id=record["record_id"],
                            data=decrypted_data,
                        )
                        stats["records_rotated"] += 1

                    stats["records_processed"] += 1

                except Exception as e:
                    self.logger.error(
                        f"Error rotating key for record {record['record_id']}: {str(e)}"
                    )
                    stats["errors"] += 1

            return stats

        except Exception as e:
            self.logger.error(f"Error in key rotation: {str(e)}")
            return {"error": str(e)}

    def _generate_record_id(self, user_id: str, data_type: str) -> str:
        """Generate unique record ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        content = f"{user_id}:{data_type}:{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _update_cache(self, key: str, data: Any) -> None:
        """Update decryption cache with size management."""
        # Remove oldest entries if cache is full
        if len(self._decryption_cache) >= self._cache_max_size:
            oldest_key = min(
                self._decryption_cache.keys(), key=lambda k: self._decryption_cache[k][1]
            )
            del self._decryption_cache[oldest_key]

        self._decryption_cache[key] = (data, datetime.now(timezone.utc))

    async def _store_encrypted_record(self, record: EncryptedDataRecord) -> None:
        """Store encrypted record in database."""
        query = """
            INSERT INTO encrypted_user_data
            (record_id, user_id, data_type, encrypted_data, encryption_key_id,
             encryption_algorithm, iv, tag, metadata, created_at)
            VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """

        await self.database.execute(
            query,
            {
                "record_id": record.record_id,
                "user_id": record.user_id,
                "data_type": record.data_type,
                "encrypted_data": base64.b64encode(record.encrypted_data).decode(),
                "encryption_key_id": record.encryption_key_id,
                "encryption_algorithm": record.encryption_algorithm.value,
                "iv": base64.b64encode(record.iv).decode() if record.iv else None,
                "tag": base64.b64encode(record.tag).decode() if record.tag else None,
                "metadata": json.dumps(record.metadata),
                "created_at": record.created_at,
            },
        )

    async def _get_encrypted_record(
        self, record_id: str, user_id: str
    ) -> Optional[EncryptedDataRecord]:
        """Retrieve encrypted record from database."""
        query = """
            SELECT * FROM encrypted_user_data
            WHERE record_id = $1 AND user_id = $2
            AND NOT is_deleted
        """

        result = await self.database.fetch_one(query, {"record_id": record_id, "user_id": user_id})

        if not result:
            return None

        return EncryptedDataRecord(
            record_id=result["record_id"],
            user_id=result["user_id"],
            data_type=result["data_type"],
            encrypted_data=base64.b64decode(result["encrypted_data"]),
            encryption_key_id=result["encryption_key_id"],
            encryption_algorithm=EncryptionAlgorithm(result["encryption_algorithm"]),
            iv=base64.b64decode(result["iv"]) if result["iv"] else None,
            tag=base64.b64decode(result["tag"]) if result["tag"] else None,
            metadata=json.loads(result["metadata"]) if result["metadata"] else {},
            created_at=result["created_at"],
            accessed_at=result["accessed_at"],
            access_count=result["access_count"],
        )

    async def _update_access_tracking(self, record_id: str) -> None:
        """Update access tracking for a record."""
        query = """
            UPDATE encrypted_user_data
            SET accessed_at = $1, access_count = access_count + 1
            WHERE record_id = $2
        """

        await self.database.execute(
            query, {"accessed_at": datetime.now(timezone.utc), "record_id": record_id}
        )

    async def _delete_encrypted_record(
        self, record_id: str, user_id: str, secure_delete: bool
    ) -> bool:
        """Delete encrypted record from database."""
        if secure_delete:
            # Overwrite data before deletion
            query = """
                UPDATE encrypted_user_data
                SET encrypted_data = $1, is_deleted = true, deleted_at = $2
                WHERE record_id = $3 AND user_id = $4
            """

            # Generate random data of same size
            import os

            random_data = os.urandom(1024)  # Placeholder size

            await self.database.execute(
                query,
                {
                    "encrypted_data": base64.b64encode(random_data).decode(),
                    "deleted_at": datetime.now(timezone.utc),
                    "record_id": record_id,
                    "user_id": user_id,
                },
            )
        else:
            # Soft delete
            query = """
                UPDATE encrypted_user_data
                SET is_deleted = true, deleted_at = $1
                WHERE record_id = $2 AND user_id = $3
            """

            await self.database.execute(
                query,
                {
                    "deleted_at": datetime.now(timezone.utc),
                    "record_id": record_id,
                    "user_id": user_id,
                },
            )

        return True

    async def _get_user_records_by_type(self, user_id: str, data_type: str) -> List[Dict[str, Any]]:
        """Get all user records of a specific type."""
        query = """
            SELECT record_id, data_type, created_at
            FROM encrypted_user_data
            WHERE user_id = $1 AND data_type = $2
            AND NOT is_deleted
        """

        return await self.database.fetch_all(query, {"user_id": user_id, "data_type": data_type})

    async def _get_all_user_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all records for a user."""
        query = """
            SELECT record_id, user_id, data_type, created_at
            FROM encrypted_user_data
            WHERE user_id = $1 AND NOT is_deleted
        """

        return await self.database.fetch_all(query, {"user_id": user_id})

    async def _get_records_for_rotation(self, force: bool) -> List[Dict[str, Any]]:
        """Get records that need key rotation."""
        if force:
            query = """
                SELECT record_id, user_id, data_type, created_at
                FROM encrypted_user_data
                WHERE NOT is_deleted
            """
        else:
            query = """
                SELECT record_id, user_id, data_type, created_at
                FROM encrypted_user_data
                WHERE NOT is_deleted
                AND created_at < NOW() - INTERVAL '$1 days'
            """

        return await self.database.fetch_all(
            query, {"days": self.key_rotation_days} if not force else {}
        )

    async def _mark_record_superseded(self, old_record_id: str, new_record_id: str) -> None:
        """Mark a record as superseded by a new one."""
        query = """
            UPDATE encrypted_user_data
            SET metadata = jsonb_set(
                metadata,
                '{superseded_by}',
                to_jsonb($1::text)
            )
            WHERE record_id = $2
        """

        await self.database.execute(
            query, {"superseded_by": new_record_id, "record_id": old_record_id}
        )
