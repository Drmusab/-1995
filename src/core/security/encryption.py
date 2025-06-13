"""
Advanced Encryption Management System
Author: Drmusab
Last Modified: 2025-01-13 12:17:53 UTC

This module provides comprehensive encryption and cryptographic services for the AI assistant,
including data encryption/decryption, key management, digital signatures, secure key derivation,
and integration with all core system components for end-to-end security.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Callable, Type, Union, Tuple, AsyncGenerator
import asyncio
import threading
import time
import os
import secrets
import hashlib
import hmac
import base64
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
import warnings
import weakref

# Cryptographic imports
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives.kdf.argon2 import Argon2
from cryptography.x509 import load_pem_x509_certificate
import argon2

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    EncryptionKeyGenerated, EncryptionKeyRotated, EncryptionOperationCompleted,
    EncryptionOperationFailed, EncryptionKeyExpired, SecurityViolationDetected,
    DataIntegrityCheckPassed, DataIntegrityCheckFailed, EncryptionPerformanceWarning,
    SystemStateChanged, ComponentHealthChanged, ErrorOccurred
)
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.dependency_injection import Container
from src.core.health_check import HealthCheck

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.observability.logging.config import get_logger


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"


class KeyDerivationFunction(Enum):
    """Key derivation function types."""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"
    HKDF = "hkdf"


class KeyType(Enum):
    """Types of encryption keys."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    DERIVATION_KEY = "derivation_key"
    SESSION_KEY = "session_key"
    MASTER_KEY = "master_key"


class EncryptionMode(Enum):
    """Encryption operation modes."""
    ENCRYPT = "encrypt"
    DECRYPT = "decrypt"
    SIGN = "sign"
    VERIFY = "verify"
    KEY_EXCHANGE = "key_exchange"


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"          # Basic encryption for non-sensitive data
    MEDIUM = "medium"    # Standard encryption for regular data
    HIGH = "high"        # Strong encryption for sensitive data
    CRITICAL = "critical" # Maximum encryption for highly sensitive data


@dataclass
class EncryptionKey:
    """Represents an encryption key with metadata."""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_data: bytes
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    max_usage_count: Optional[int] = None
    
    # Security properties
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    can_export: bool = False
    requires_hardware: bool = False
    
    # Key hierarchy
    parent_key_id: Optional[str] = None
    derived_keys: Set[str] = field(default_factory=set)
    
    # Additional metadata
    purpose: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptionResult:
    """Result of an encryption operation."""
    success: bool
    data: Optional[bytes] = None
    algorithm: Optional[EncryptionAlgorithm] = None
    key_id: Optional[str] = None
    
    # Additional data
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    signature: Optional[bytes] = None
    
    # Performance metrics
    processing_time: float = 0.0
    data_size: int = 0
    
    # Error information
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class EncryptionContext:
    """Context for encryption operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component_id: Optional[str] = None
    data_type: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    # Operation parameters
    algorithm_preference: Optional[EncryptionAlgorithm] = None
    key_rotation_policy: Optional[str] = None
    compression_enabled: bool = False
    
    # Audit and compliance
    audit_required: bool = True
    compliance_tags: Set[str] = field(default_factory=set)
    retention_policy: Optional[str] = None
    
    # Metadata
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_metadata: Dict[str, Any] = field(default_factory=dict)


class EncryptionError(Exception):
    """Custom exception for encryption operations."""
    
    def __init__(self, message: str, operation_id: Optional[str] = None,
                 error_code: Optional[str] = None, key_id: Optional[str] = None):
        super().__init__(message)
        self.operation_id = operation_id
        self.error_code = error_code
        self.key_id = key_id
        self.timestamp = datetime.now(timezone.utc)


class KeyManager:
    """Manages encryption keys and their lifecycle."""
    
    def __init__(self, logger, config: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.config = config or {}
        self._keys: Dict[str, EncryptionKey] = {}
        self._key_store_path = Path(self.config.get('key_store_path', 'data/keys'))
        self._master_key: Optional[bytes] = None
        self._key_lock = threading.RLock()
        
        # Ensure key store directory exists
        self._key_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize master key
        self._initialize_master_key()
    
    def _initialize_master_key(self) -> None:
        """Initialize or load the master key."""
        master_key_file = self._key_store_path / "master.key"
        
        if master_key_file.exists():
            # Load existing master key
            try:
                with open(master_key_file, 'rb') as f:
                    encrypted_master = f.read()
                
                # For security, master key should be protected by system keyring
                # For now, using a simple approach
                self._master_key = base64.b64decode(encrypted_master)
                
            except Exception as e:
                self.logger.error(f"Failed to load master key: {str(e)}")
                raise EncryptionError("Failed to load master key")
        else:
            # Generate new master key
            self._master_key = secrets.token_bytes(32)  # 256-bit key
            
            try:
                # Store master key (in production, use system keyring)
                with open(master_key_file, 'wb') as f:
                    f.write(base64.b64encode(self._master_key))
                
                # Set restrictive permissions
                os.chmod(master_key_file, 0o600)
                
                self.logger.info("Generated new master key")
                
            except Exception as e:
                self.logger.error(f"Failed to store master key: {str(e)}")
                raise EncryptionError("Failed to store master key")
    
    def generate_key(self, algorithm: EncryptionAlgorithm, key_id: Optional[str] = None,
                    security_level: SecurityLevel = SecurityLevel.MEDIUM,
                    expires_in_days: Optional[int] = None) -> str:
        """Generate a new encryption key."""
        with self._key_lock:
            if key_id is None:
                key_id = f"key_{algorithm.value}_{int(time.time())}_{secrets.token_hex(8)}"
            
            if key_id in self._keys:
                raise EncryptionError(f"Key {key_id} already exists")
            
            # Generate key data based on algorithm
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                key_data = secrets.token_bytes(32)  # 256-bit key
                key_type = KeyType.SYMMETRIC
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                key_data = secrets.token_bytes(32)  # 256-bit key
                key_type = KeyType.SYMMETRIC
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                key_data = secrets.token_bytes(32)  # 256-bit key
                key_type = KeyType.SYMMETRIC
            elif algorithm == EncryptionAlgorithm.FERNET:
                key_data = Fernet.generate_key()
                key_type = KeyType.SYMMETRIC
            elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                key_size = 2048 if algorithm == EncryptionAlgorithm.RSA_2048 else 4096
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )
                key_data = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                key_type = KeyType.ASYMMETRIC_PRIVATE
            else:
                raise EncryptionError(f"Unsupported algorithm: {algorithm}")
            
            # Calculate expiration time
            expires_at = None
            if expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
            
            # Create key object
            key = EncryptionKey(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                key_data=key_data,
                expires_at=expires_at,
                security_level=security_level
            )
            
            # Store key
            self._keys[key_id] = key
            self._persist_key(key)
            
            self.logger.info(f"Generated new {algorithm.value} key: {key_id}")
            return key_id
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get a key by ID."""
        with self._key_lock:
            key = self._keys.get(key_id)
            
            if key:
                # Check if key has expired
                if key.expires_at and datetime.now(timezone.utc) > key.expires_at:
                    self.logger.warning(f"Key {key_id} has expired")
                    return None
                
                # Update usage tracking
                key.last_used = datetime.now(timezone.utc)
                key.usage_count += 1
                
                # Check usage limits
                if key.max_usage_count and key.usage_count > key.max_usage_count:
                    self.logger.warning(f"Key {key_id} has exceeded usage limit")
                    return None
            
            return key
    
    def rotate_key(self, old_key_id: str, new_algorithm: Optional[EncryptionAlgorithm] = None) -> str:
        """Rotate an encryption key."""
        with self._key_lock:
            old_key = self._keys.get(old_key_id)
            if not old_key:
                raise EncryptionError(f"Key {old_key_id} not found")
            
            # Use same algorithm if not specified
            algorithm = new_algorithm or old_key.algorithm
            
            # Generate new key
            new_key_id = self.generate_key(
                algorithm=algorithm,
                security_level=old_key.security_level
            )
            
            # Update key hierarchy
            new_key = self._keys[new_key_id]
            new_key.parent_key_id = old_key_id
            old_key.derived_keys.add(new_key_id)
            
            self.logger.info(f"Rotated key {old_key_id} to {new_key_id}")
            return new_key_id
    
    def derive_key(self, parent_key_id: str, kdf: KeyDerivationFunction,
                  salt: Optional[bytes] = None, info: Optional[bytes] = None,
                  length: int = 32) -> str:
        """Derive a new key from an existing key."""
        with self._key_lock:
            parent_key = self.get_key(parent_key_id)
            if not parent_key:
                raise EncryptionError(f"Parent key {parent_key_id} not found")
            
            if salt is None:
                salt = secrets.token_bytes(16)
            
            # Derive key using specified KDF
            if kdf == KeyDerivationFunction.PBKDF2:
                kdf_instance = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=length,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                derived_key_data = kdf_instance.derive(parent_key.key_data)
                
            elif kdf == KeyDerivationFunction.HKDF:
                kdf_instance = HKDF(
                    algorithm=hashes.SHA256(),
                    length=length,
                    salt=salt,
                    info=info,
                    backend=default_backend()
                )
                derived_key_data = kdf_instance.derive(parent_key.key_data)
                
            elif kdf == KeyDerivationFunction.SCRYPT:
                kdf_instance = Scrypt(
                    algorithm=hashes.SHA256(),
                    length=length,
                    salt=salt,
                    n=2**14,
                    r=8,
                    p=1,
                    backend=default_backend()
                )
                derived_key_data = kdf_instance.derive(parent_key.key_data)
                
            elif kdf == KeyDerivationFunction.ARGON2:
                # Use argon2-cffi for Argon2
                hasher = argon2.PasswordHasher()
                derived_key_data = hasher.hash(parent_key.key_data, salt=salt)[:length]
                
            else:
                raise EncryptionError(f"Unsupported KDF: {kdf}")
            
            # Create derived key
            derived_key_id = f"derived_{parent_key_id}_{secrets.token_hex(8)}"
            derived_key = EncryptionKey(
                key_id=derived_key_id,
                key_type=KeyType.DERIVATION_KEY,
                algorithm=EncryptionAlgorithm.AES_256_GCM,  # Default for derived keys
                key_data=derived_key_data,
                parent_key_id=parent_key_id,
                security_level=parent_key.security_level
            )
            
            # Store derived key
            self._keys[derived_key_id] = derived_key
            parent_key.derived_keys.add(derived_key_id)
            
            self.logger.info(f"Derived key {derived_key_id} from {parent_key_id}")
            return derived_key_id
    
    def delete_key(self, key_id: str, force: bool = False) -> None:
        """Delete a key."""
        with self._key_lock:
            key = self._keys.get(key_id)
            if not key:
                return
            
            # Check if key has derived keys
            if key.derived_keys and not force:
                raise EncryptionError(f"Key {key_id} has derived keys, use force=True")
            
            # Delete derived keys recursively
            for derived_key_id in list(key.derived_keys):
                self.delete_key(derived_key_id, force=True)
            
            # Remove from parent's derived keys
            if key.parent_key_id and key.parent_key_id in self._keys:
                self._keys[key.parent_key_id].derived_keys.discard(key_id)
            
            # Delete key
            del self._keys[key_id]
            self._remove_persisted_key(key_id)
            
            # Clear sensitive data
            if isinstance(key.key_data, bytes):
                # Overwrite key data in memory (best effort)
                try:
                    import ctypes
                    ctypes.memset(id(key.key_data), 0, len(key.key_data))
                except:
                    pass
            
            self.logger.info(f"Deleted key: {key_id}")
    
    def list_keys(self, key_type: Optional[KeyType] = None,
                 algorithm: Optional[EncryptionAlgorithm] = None) -> List[str]:
        """List available keys."""
        with self._key_lock:
            keys = []
            for key_id, key in self._keys.items():
                if key_type and key.key_type != key_type:
                    continue
                if algorithm and key.algorithm != algorithm:
                    continue
                keys.append(key_id)
            return keys
    
    def _persist_key(self, key: EncryptionKey) -> None:
        """Persist a key to storage."""
        try:
            # Encrypt key data with master key
            cipher = AESGCM(self._master_key)
            nonce = secrets.token_bytes(12)
            encrypted_key_data = cipher.encrypt(nonce, key.key_data, None)
            
            # Prepare key metadata
            key_metadata = {
                'key_id': key.key_id,
                'key_type': key.key_type.value,
                'algorithm': key.algorithm.value,
                'created_at': key.created_at.isoformat(),
                'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                'security_level': key.security_level.value,
                'parent_key_id': key.parent_key_id,
                'purpose': list(key.purpose),
                'tags': list(key.tags),
                'metadata': key.metadata,
                'nonce': base64.b64encode(nonce).decode(),
                'encrypted_data': base64.b64encode(encrypted_key_data).decode()
            }
            
            # Save to file
            key_file = self._key_store_path / f"{key.key_id}.json"
            with open(key_file, 'w') as f:
                json.dump(key_metadata, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to persist key {key.key_id}: {str(e)}")
            raise EncryptionError(f"Failed to persist key: {str(e)}")
    
    def _remove_persisted_key(self, key_id: str) -> None:
        """Remove a persisted key file."""
        try:
            key_file = self._key_store_path / f"{key_id}.json"
            if key_file.exists():
                key_file.unlink()
        except Exception as e:
            self.logger.warning(f"Failed to remove key file {key_id}: {str(e)}")


class EncryptionEngine:
    """Core encryption engine for various cryptographic operations."""
    
    def __init__(self, key_manager: KeyManager, logger):
        self.key_manager = key_manager
        self.logger = logger
        self._cipher_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
    
    def encrypt_data(self, data: bytes, key_id: str, 
                    algorithm: Optional[EncryptionAlgorithm] = None,
                    context: Optional[EncryptionContext] = None) -> EncryptionResult:
        """Encrypt data using specified key and algorithm."""
        start_time = time.time()
        
        try:
            # Get encryption key
            key = self.key_manager.get_key(key_id)
            if not key:
                raise EncryptionError(f"Key {key_id} not found or invalid")
            
            # Use key's algorithm if not specified
            algo = algorithm or key.algorithm
            
            # Perform encryption based on algorithm
            if algo == EncryptionAlgorithm.AES_256_GCM:
                result = self._encrypt_aes_gcm(data, key.key_data)
            elif algo == EncryptionAlgorithm.AES_256_CBC:
                result = self._encrypt_aes_cbc(data, key.key_data)
            elif algo == EncryptionAlgorithm.CHACHA20_POLY1305:
                result = self._encrypt_chacha20(data, key.key_data)
            elif algo == EncryptionAlgorithm.FERNET:
                result = self._encrypt_fernet(data, key.key_data)
            elif algo in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                result = self._encrypt_rsa(data, key.key_data)
            else:
                raise EncryptionError(f"Unsupported encryption algorithm: {algo}")
            
            # Update result with metadata
            result.algorithm = algo
            result.key_id = key_id
            result.processing_time = time.time() - start_time
            result.data_size = len(data)
            result.success = True
            
            return result
            
        except Exception as e:
            return EncryptionResult(
                success=False,
                error_message=str(e),
                error_code="ENCRYPTION_FAILED",
                processing_time=time.time() - start_time,
                data_size=len(data)
            )
    
    def decrypt_data(self, encrypted_data: bytes, key_id: str,
                    algorithm: EncryptionAlgorithm, iv: Optional[bytes] = None,
                    tag: Optional[bytes] = None,
                    context: Optional[EncryptionContext] = None) -> EncryptionResult:
        """Decrypt data using specified key and algorithm."""
        start_time = time.time()
        
        try:
            # Get decryption key
            key = self.key_manager.get_key(key_id)
            if not key:
                raise EncryptionError(f"Key {key_id} not found or invalid")
            
            # Perform decryption based on algorithm
            if algorithm == EncryptionAlgorithm.AES_256_GCM:
                result = self._decrypt_aes_gcm(encrypted_data, key.key_data, iv, tag)
            elif algorithm == EncryptionAlgorithm.AES_256_CBC:
                result = self._decrypt_aes_cbc(encrypted_data, key.key_data, iv)
            elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
                result = self._decrypt_chacha20(encrypted_data, key.key_data)
            elif algorithm == EncryptionAlgorithm.FERNET:
                result = self._decrypt_fernet(encrypted_data, key.key_data)
            elif algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                result = self._decrypt_rsa(encrypted_data, key.key_data)
            else:
                raise EncryptionError(f"Unsupported decryption algorithm: {algorithm}")
            
            # Update result with metadata
            result.algorithm = algorithm
            result.key_id = key_id
            result.processing_time = time.time() - start_time
            result.data_size = len(encrypted_data)
            result.success = True
            
            return result
            
        except Exception as e:
            return EncryptionResult(
                success=False,
                error_message=str(e),
                error_code="DECRYPTION_FAILED",
                processing_time=time.time() - start_time,
                data_size=len(encrypted_data)
            )
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> EncryptionResult:
        """Encrypt using AES-256-GCM."""
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        cipher = AESGCM(key)
        encrypted_data = cipher.encrypt(iv, data, None)
        
        # GCM includes authentication tag in the output
        ciphertext = encrypted_data[:-16]
        tag = encrypted_data[-16:]
        
        return EncryptionResult(
            data=ciphertext,
            iv=iv,
            tag=tag
        )
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes, key: bytes, 
                        iv: bytes, tag: bytes) -> EncryptionResult:
        """Decrypt using AES-256-GCM."""
        cipher = AESGCM(key)
        # Combine ciphertext and tag for GCM
        ciphertext_with_tag = encrypted_data + tag
        decrypted_data = cipher.decrypt(iv, ciphertext_with_tag, None)
        
        return EncryptionResult(data=decrypted_data)
    
    def _encrypt_aes_cbc(self, data: bytes, key: bytes) -> EncryptionResult:
        """Encrypt using AES-256-CBC."""
        iv = secrets.token_bytes(16)  # 128-bit IV for CBC
        
        # Add PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptionResult(
            data=encrypted_data,
            iv=iv
        )
    
    def _decrypt_aes_cbc(self, encrypted_data: bytes, key: bytes, iv: bytes) -> EncryptionResult:
        """Decrypt using AES-256-CBC."""
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove PKCS7 padding
        unpadder = padding.PKCS7(128).unpadder()
        decrypted_data = unpadder.update(padded_data) + unpadder.finalize()
        
        return EncryptionResult(data=decrypted_data)
    
    def _encrypt_chacha20(self, data: bytes, key: bytes) -> EncryptionResult:
        """Encrypt using ChaCha20-Poly1305."""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        cipher = ChaCha20Poly1305(key)
        encrypted_data = cipher.encrypt(nonce, data, None)
        
        return EncryptionResult(
            data=encrypted_data,
            iv=nonce
        )
    
    def _decrypt_chacha20(self, encrypted_data: bytes, key: bytes) -> EncryptionResult:
        """Decrypt using ChaCha20-Poly1305."""
        # For ChaCha20Poly1305, the nonce should be provided separately
        # This is a simplified implementation
        cipher = ChaCha20Poly1305(key)
        # In practice, you'd need to extract the nonce from the encrypted data
        # or have it provided separately
        raise EncryptionError("ChaCha20 decryption requires separate nonce handling")
    
    def _encrypt_fernet(self, data: bytes, key: bytes) -> EncryptionResult:
        """Encrypt using Fernet."""
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        
        return EncryptionResult(data=encrypted_data)
    
    def _decrypt_fernet(self, encrypted_data: bytes, key: bytes) -> EncryptionResult:
        """Decrypt using Fernet."""
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data)
        
        return EncryptionResult(data=decrypted_data)
    
    def _encrypt_rsa(self, data: bytes, private_key_data: bytes) -> EncryptionResult:
        """Encrypt using RSA (actually signs with private key)."""
        private_key = serialization.load_pem_private_key(
            private_key_data, password=None, backend=default_backend()
        )
        
        # RSA encryption should use public key, but for signing we use private key
        signature = private_key.sign(
            data,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return EncryptionResult(data=signature)
    
    def _decrypt_rsa(self, signature: bytes, private_key_data: bytes) -> EncryptionResult:
        """Verify RSA signature."""
        private_key = serialization.load_pem_private_key(
            private_key_data, password=None, backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        try:
            # This would require the original data to verify
            # This is a simplified implementation
            public_key.verify(
                signature,
                b"",  # Original data would go here
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return EncryptionResult(data=b"verified")
        except Exception as e:
            raise EncryptionError(f"RSA verification failed: {str(e)}")


class DataIntegrityManager:
    """Manages data integrity verification using hashes and checksums."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def calculate_hash(self, data: bytes, algorithm: str = "sha256") -> str:
        """Calculate hash of data."""
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise EncryptionError(f"Unsupported hash algorithm: {algorithm}")
    
    def verify_hash(self, data: bytes, expected_hash: str, algorithm: str = "sha256") -> bool:
        """Verify data integrity using hash."""
        calculated_hash = self.calculate_hash(data, algorithm)
        return hmac.compare_digest(calculated_hash, expected_hash)
    
    def calculate_hmac(self, data: bytes, key: bytes, algorithm: str = "sha256") -> str:
        """Calculate HMAC for data."""
        if algorithm == "sha256":
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algorithm == "sha512":
            return hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise EncryptionError(f"Unsupported HMAC algorithm: {algorithm}")
    
    def verify_hmac(self, data: bytes, key: bytes, expected_hmac: str, 
                   algorithm: str = "sha256") -> bool:
        """Verify data integrity using HMAC."""
        calculated_hmac = self.calculate_hmac(data, key, algorithm)
        return hmac.compare_digest(calculated_hmac, expected_hmac)


class EncryptionManager:
    """
    Advanced Encryption Management System for the AI Assistant.
    
    This manager provides comprehensive encryption services including:
    - Multiple encryption algorithms and modes
    - Advanced key management and rotation
    - Data integrity verification
    - Performance optimization and caching
    - Integration with all core system components
    - Hardware acceleration support
    - Compliance and audit logging
    - Secure key derivation and storage
    - Real-time monitoring and alerts
    """
    
    def __init__(self, container: Container):
        """
        Initialize the encryption manager.
        
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
        
        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)
        
        # Initialize encryption components
        encryption_config = self.config.get("encryption", {})
        self.key_manager = KeyManager(self.logger, encryption_config.get("keys", {}))
        self.encryption_engine = EncryptionEngine(self.key_manager, self.logger)
        self.integrity_manager = DataIntegrityManager(self.logger)
        
        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(
            max_workers=encryption_config.get("max_workers", 4),
            thread_name_prefix="encryption"
        )
        
        # Configuration
        self.default_algorithm = EncryptionAlgorithm(
            encryption_config.get("default_algorithm", "aes_256_gcm")
        )
        self.default_security_level = SecurityLevel(
            encryption_config.get("default_security_level", "medium")
        )
        self.enable_compression = encryption_config.get("enable_compression", True)
        self.enable_integrity_check = encryption_config.get("enable_integrity_check", True)
        self.key_rotation_interval = encryption_config.get("key_rotation_days", 90)
        
        # State tracking
        self.operation_history: deque = deque(maxlen=1000)
        self.performance_stats = defaultdict(list)
        
        # Setup monitoring and health checks
        self._setup_monitoring()
        self.health_check.register_component("encryption_manager", self._health_check_callback)
        
        # Initialize default keys
        asyncio.create_task(self._initialize_default_keys())
        
        # Start background tasks
        asyncio.create_task(self._key_rotation_loop())
        asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("EncryptionManager initialized successfully")

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register encryption metrics
            self.metrics.register_counter("encryption_operations_total")
            self.metrics.register_counter("encryption_operations_successful")
            self.metrics.register_counter("encryption_operations_failed")
            self.metrics.register_histogram("encryption_operation_duration_seconds")
            self.metrics.register_gauge("encryption_keys_active")
            self.metrics.register_counter("encryption_keys_generated")
            self.metrics.register_counter("encryption_keys_rotated")
            self.metrics.register_histogram("encryption_data_size_bytes")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def _initialize_default_keys(self) -> None:
        """Initialize default encryption keys."""
        try:
            # Check if default keys already exist
            default_keys = self.key_manager.list_keys()
            
            if not default_keys:
                # Generate default system keys
                system_key_id = self.key_manager.generate_key(
                    algorithm=self.default_algorithm,
                    key_id="system_default",
                    security_level=SecurityLevel.HIGH
                )
                
                session_key_id = self.key_manager.generate_key(
                    algorithm=EncryptionAlgorithm.AES_256_GCM,
                    key_id="session_default",
                    security_level=SecurityLevel.MEDIUM
                )
                
                # Emit key generation events
                await self.event_bus.emit(EncryptionKeyGenerated(
                    key_id=system_key_id,
                    algorithm=self.default_algorithm.value,
                    key_type=KeyType.SYMMETRIC.value
                ))
                
                await self.event_bus.emit(EncryptionKeyGenerated(
                    key_id=session_key_id,
                    algorithm=EncryptionAlgorithm.AES_256_GCM.value,
                    key_type=KeyType.SYMMETRIC.value
                ))
                
                self.logger.info("Initialized default encryption keys")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize default keys: {str(e)}")

    @handle_exceptions
    async def encrypt(self, data: Union[str, bytes], context: Optional[EncryptionContext] = None,
                     algorithm: Optional[EncryptionAlgorithm] = None,
                     key_id: Optional[str] = None) -> EncryptionResult:
        """
        Encrypt data with comprehensive options and context awareness.
        
        Args:
            data: Data to encrypt (string or bytes)
            context: Encryption context for operation
            algorithm: Specific algorithm to use
            key_id: Specific key to use
            
        Returns:
            Encryption result with metadata
        """
        start_time = time.time()
        
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Create context if not provided
        if context is None:
            context = EncryptionContext()
        
        try:
            with self.tracer.trace("encryption_operation") as span:
                span.set_attributes({
                    "operation_id": context.operation_id,
                    "data_size": len(data),
                    "algorithm": (algorithm or self.default_algorithm).value,
                    "security_level": context.security_level.value
                })
                
                # Compress data if enabled and beneficial
                original_size = len(data)
                if self.enable_compression and context.compression_enabled and len(data) > 1024:
                    try:
                        import gzip
                        compressed_data = gzip.compress(data)
                        if len(compressed_data) < len(data) * 0.9:  # Only use if significant compression
                            data = compressed_data
                            context.additional_metadata['compressed'] = True
                    except Exception as e:
                        self.logger.warning(f"Compression failed: {str(e)}")
                
                # Select key if not specified
                if key_id is None:
                    key_id = await self._select_optimal_key(context)
                
                # Select algorithm if not specified
                if algorithm is None:
                    algorithm = await self._select_optimal_algorithm(context)
                
                # Calculate data integrity hash if enabled
                data_hash = None
                if self.enable_integrity_check:
                    data_hash = self.integrity_manager.calculate_hash(data)
                    context.additional_metadata['data_hash'] = data_hash
                
                # Perform encryption
                encryption_result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self.encryption_engine.encrypt_data,
                    data, key_id, algorithm, context
                )
                
                if encryption_result.success:
                    # Update metrics
                    self.metrics.increment("encryption_operations_total")
                    self.metrics.increment("encryption_operations_successful")
                    self.metrics.record("encryption_operation_duration_seconds", 
                                      encryption_result.processing_time)
                    self.metrics.record("encryption_data_size_bytes", original_size)
                    
                    # Store operation in history
                    operation_record = {
                        'operation_id': context.operation_id,
                        'operation_type': 'encrypt',
                        'timestamp': encryption_result.timestamp,
                        'key_id': key_id,
                        'algorithm': algorithm.value,
                        'data_size': original_size,
                        'processing_time': encryption_result.processing_time,
                        'success': True,
                        'context': context.additional_metadata
                    }
                    self.operation_history.append(operation_record)
                    
                    # Emit success event
                    await self.event_bus.emit(EncryptionOperationCompleted(
                        operation_id=context.operation_id,
                        operation_type="encrypt",
                        key_id=key_id,
                        algorithm=algorithm.value,
                        data_size=original_size,
                        processing_time=encryption_result.processing_time
                    ))
                    
                    self.logger.debug(
                        f"Encrypted {original_size} bytes in {encryption_result.processing_time:.3f}s "
                        f"using {algorithm.value}"
                    )
                else:
                    # Handle encryption failure
                    self.metrics.increment("encryption_operations_failed")
                    
                    await self.event_bus.emit(EncryptionOperationFailed(
                        operation_id=context.operation_id,
                        operation_type="encrypt",
                        error_message=encryption_result.error_message,
                        error_code=encryption_result.error_code
                    ))
                
                return encryption_result
                
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - start_time
            
            error_result = EncryptionResult(
                success=False,
                error_message=str(e),
                error_code="ENCRYPTION_ERROR",
                processing_time=processing_time,
                data_size=len(data),
                operation_id=context.operation_id
            )
            
            self.metrics.increment("encryption_operations_failed")
            
            await self.event_bus.emit(EncryptionOperationFailed(
                operation_id=context.operation_id,
                operation_type="encrypt",
                error_message=str(e),
                error_code="ENCRYPTION_ERROR"
            ))
            
            self.logger.error(f"Encryption failed for operation {context.operation_id}: {str(e)}")
            return error_result

    @handle_exceptions
    async def decrypt(self, encrypted_data: bytes, key_id: str, algorithm: EncryptionAlgorithm,
                     iv: Optional[bytes] = None, tag: Optional[bytes] = None,
                     context: Optional[EncryptionContext] = None) -> EncryptionResult:
        """
        Decrypt data with comprehensive options and verification.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            key_id: Key ID for decryption
            algorithm: Algorithm used for encryption
            iv: Initialization vector if required
            tag: Authentication tag if required
            context: Decryption context
            
        Returns:
            Decryption result with metadata
        """
        start_time = time.time()
        
        # Create context if not provided
        if context is None:
            context = EncryptionContext()
        
        try:
            with self.tracer.trace("decryption_operation") as span:
                span.set_attributes({
                    "operation_id": context.operation_id,
                    "data_size": len(encrypted_data),
                    "algorithm": algorithm.value,
                    "key_id": key_id
                })
                
                # Perform decryption
                decryption_result = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    self.encryption_engine.decrypt_data,
                    encrypted_data, key_id, algorithm, iv, tag, context
                )
                
                if decryption_result.success:
                    # Verify data integrity if hash is available
                    if (self.enable_integrity_check and 
                        context.additional_metadata.get('data_hash')):
                        
                        expected_hash = context.additional_metadata['data_hash']
                        if not self.integrity_manager.verify_hash(decryption_result.data, expected_hash):
                            await self.event_bus.emit(DataIntegrityCheckFailed(
                                operation_id=context.operation_id,
                                data_size=len(decryption_result.data),
                                expected_hash=expected_hash
                            ))
                            
                            decryption_result.success = False
                            decryption_result.error_message = "Data integrity check failed"
                            decryption_result.error_code = "INTEGRITY_CHECK_FAILED"
                        else:
                            await self.event_bus.emit(DataIntegrityCheckPassed(
                                operation_id=context.operation_id,
                                data_size=len(decryption_result.data)
                            ))
                    
                    # Decompress data if it was compressed
                    if (decryption_result.success and 
                        context.additional_metadata.get('compressed')):
                        try:
                            import gzip
                            decryption_result.data = gzip.decompress(decryption_result.data)
                        except Exception as e:
                            self.logger.warning(f"Decompression failed: {str(e)}")
                    
                    if decryption_result.success:
                        # Update metrics
                        self.metrics.increment("encryption_operations_total")
                        self.metrics.increment("encryption_operations_successful")
                        self.metrics.record("encryption_operation_duration_seconds", 
                                          decryption_result.processing_time)
                        
                        # Emit success event
                        await self.event_bus.emit(EncryptionOperationCompleted(
                            operation_id=context.operation_id,
                            operation_type="decrypt",
                            key_id=key_id,
                            algorithm=algorithm.value,
                            data_size=len(encrypted_data),
                            processing_time=decryption_result.processing_time
                        ))
                else:
                    # Handle decryption failure
                    self.metrics.increment("encryption_operations_failed")
                    
                    await self.event_bus.emit(EncryptionOperationFailed(
                        operation_id=context.operation_id,
                        operation_type="decrypt",
                        error_message=decryption_result.error_message,
                        error_code=decryption_result.error_code
                    ))
                
                return decryption_result
                
        except Exception as e:
            # Handle unexpected errors
            processing_time = time.time() - start_time
            
            error_result = EncryptionResult(
                success=False,
                error_message=str(e),
                error_code="DECRYPTION_ERROR",
                processing_time=processing_time,
                data_size=len(encrypted_data),
                operation_id=context.operation_id
            )
            
            self.metrics.increment("encryption_operations_failed")
            
            await self.event_bus.emit(EncryptionOperationFailed(
                operation_id=context.operation_id,
                operation_type="decrypt",
                error_message=str(e),
                error_code="DECRYPTION_ERROR"
            ))
            
            self.logger.error(f"Decryption failed for operation {context.operation_id}: {str(e)}")
            return error_result

    @handle_exceptions
    async def encrypt_json(self, data: Dict[str, Any], context: Optional[EncryptionContext] = None) -> EncryptionResult:
        """
        Encrypt JSON data with automatic serialization.
        
        Args:
            data: Dictionary to encrypt
            context: Encryption context
            
        Returns:
            Encryption result
        """
        try:
            # Serialize to JSON
            json_data = json.dumps(data, separators=(',', ':'), sort_keys=True)
            
            # Encrypt JSON string
            result = await self.encrypt(json_data, context)
            
            if result.success:
                # Add metadata indicating this is JSON data
                if context:
                    context.additional_metadata['data_type'] = 'json'
            
            return result
            
        except Exception as e:
            return EncryptionResult(
                success=False,
                error_message=f"JSON encryption failed: {str(e)}",
                error_code="JSON_ENCRYPTION_ERROR"
            )

    @handle_exceptions
    async def decrypt_json(self, encrypted_data: bytes, key_id: str, algorithm: EncryptionAlgorithm,
                          iv: Optional[bytes] = None, tag: Optional[bytes] = None,
                          context: Optional[EncryptionContext] = None) -> Dict[str, Any]:
        """
        Decrypt and deserialize JSON data.
        
        Args:
            encrypted_data: Encrypted JSON data
            key_id: Decryption key ID
            algorithm: Algorithm used for encryption
            iv: Initialization vector if required
            tag: Authentication tag if required
            context: Decryption context
            
        Returns:
            Deserialized dictionary
        """
        try:
            # Decrypt data
            result = await self.decrypt(encrypted_data, key_id, algorithm, iv, tag, context)
            
            if not result.success:
                raise EncryptionError(f"Decryption failed: {result.error_message}")
            
            # Deserialize JSON
            json_data = result.data.decode('utf-8')
            return json.loads(json_data)
            
        except Exception as e:
            self.logger.error(f"JSON decryption failed: {str(e)}")
            raise EncryptionError(f"JSON decryption failed: {str(e)}")

    async def _select_optimal_key(self, context: EncryptionContext) -> str:
        """Select the optimal encryption key based on context."""
        # Get available keys based on security level
        available_keys = []
        
        for key_id in self.key_manager.list_keys():
            key = self.key_manager.get_key(key_id)
            if key and key.security_level.value >= context.security_level.value:
                available_keys.append(key_id)
        
        if not available_keys:
            # Generate a new key if none available
            key_id = self.key_manager.generate_key(
                algorithm=self.default_algorithm,
                security_level=context.security_level
            )
            return key_id
        
        # Select key based on context (e.g., session-specific, component-specific)
        if context.session_id:
            session_keys = [k for k in available_keys if 'session' in k]
            if session_keys:
                return session_keys[0]
        
        if context.component_id:
            component_keys = [k for k in available_keys if context.component_id in k]
            if component_keys:
                return component_keys[0]
        
        # Default to first available key
        return available_keys[0]

    async def _select_optimal_algorithm(self, context: EncryptionContext) -> EncryptionAlgorithm:
        """Select the optimal encryption algorithm based on context."""
        # Return preference if specified
        if context.algorithm_preference:
            return context.algorithm_preference
        
        # Select based on security level
        if context.security_level == SecurityLevel.CRITICAL:
            return EncryptionAlgorithm.AES_256_GCM
        elif context.security_level == SecurityLevel.HIGH:
            return EncryptionAlgorithm.AES_256_GCM
        elif context.security_level == SecurityLevel.MEDIUM:
            return EncryptionAlgorithm.AES_256_GCM
        else:
            return EncryptionAlgorithm.FERNET

    @handle_exceptions
    async def generate_key(self, algorithm: EncryptionAlgorithm, 
                          security_level: SecurityLevel = SecurityLevel.MEDIUM,
                          expires_in_days: Optional[int] = None,
                          key_id: Optional[str] = None) -> str:
        """
        Generate a new encryption key.
        
        Args:
            algorithm: Encryption algorithm
            security_level: Security level for the key
            expires_in_days: Key expiration in days
            key_id: Optional custom key ID
            
        Returns:
            Generated key ID
        """
        try:
            generated_key_id = self.key_manager.generate_key(
                algorithm=algorithm,
                key_id=key_id,
                security_level=security_level,
                expires_in_days=expires_in_days
            )
            
            # Update metrics
            self.metrics.increment("encryption_keys_generated")
            self.metrics.set("encryption_keys_active", len(self.key_manager.list_keys()))
            
            # Emit key generation event
            await self.event_bus.emit(EncryptionKeyGenerated(
                key_id=generated_key_id,
                algorithm=algorithm.value,
                key_type=KeyType.SYMMETRIC.value if algorithm != EncryptionAlgorithm.RSA_2048 else KeyType.ASYMMETRIC_PRIVATE.value
            ))
            
            self.logger.info(f"Generated new encryption key: {generated_key_id}")
            return generated_key_id
            
        except Exception as e:
            self.logger.error(f"Key generation failed: {str(e)}")
            raise EncryptionError(f"Key generation failed: {str(e)}")

    @handle_exceptions
    async def rotate_key(self, old_key_id: str, 
                        new_algorithm: Optional[EncryptionAlgorithm] = None) -> str:
        """
        Rotate an encryption key.
        
        Args:
            old_key_id: ID of the key to rotate
            new_algorithm: Optional new algorithm for the rotated key
            
        Returns:
            New key ID
        """
        try:
            new_key_id = self.key_manager.rotate_key(old_key_id, new_algorithm)
            
            # Update metrics
            self.metrics.increment("encryption_keys_rotated")
            
            # Emit key rotation event
            await self.event_bus.emit(EncryptionKeyRotated(
                old_key_id=old_key_id,
                new_key_id=new_key_id,
                algorithm=new_algorithm.value if new_algorithm else "same"
            ))
            
            self.logger.info(f"Rotated key {old_key_id} to {new_key_id}")
            return new_key_id
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {str(e)}")
            raise EncryptionError(f"Key rotation failed: {str(e)}")

    @handle_exceptions
    async def derive_key(self, parent_key_id: str, kdf: KeyDerivationFunction,
                        salt: Optional[bytes] = None, info: Optional[bytes] = None,
                        length: int = 32) -> str:
        """
        Derive a new key from an existing key.
        
        Args:
            parent_key_id: Parent key ID
            kdf: Key derivation function to use
            salt: Optional salt for derivation
            info: Optional info parameter for HKDF
            length: Derived key length in bytes
            
        Returns:
            Derived key ID
        """
        try:
            derived_key_id = self.key_manager.derive_key(
                parent_key_id=parent_key_id,
                kdf=kdf,
                salt=salt,
                info=info,
                length=length
            )
            
            self.logger.info(f"Derived key {derived_key_id} from {parent_key_id}")
            return derived_key_id
            
        except Exception as e:
            self.logger.error(f"Key derivation failed: {str(e)}")
            raise EncryptionError(f"Key derivation failed: {str(e)}")

    def list_keys(self, key_type: Optional[KeyType] = None,
                 algorithm: Optional[EncryptionAlgorithm] = None) -> List[Dict[str, Any]]:
        """
        List available encryption keys.
        
        Args:
            key_type: Optional key type filter
            algorithm: Optional algorithm filter
            
        Returns:
            List of key information
        """
        key_ids = self.key_manager.list_keys(key_type, algorithm)
        keys_info = []
        
        for key_id in key_ids:
            key = self.key_manager.get_key(key_id)
            if key:
                keys_info.append({
                    'key_id': key.key_id,
                    'key_type': key.key_type.value,
                    'algorithm': key.algorithm.value,
                    'created_at': key.created_at.isoformat(),
                    'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                    'last_used': key.last_used.isoformat() if key.last_used else None,
                    'usage_count': key.usage_count,
                    'security_level': key.security_level.value,
                    'parent_key_id': key.parent_key_id,
                    'derived_keys_count': len(key.derived_keys),
                    'purpose': list(key.purpose),
                    'tags': list(key.tags)
                })
        
        return keys_info

    def get_encryption_status(self) ->
