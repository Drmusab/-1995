"""
Skill Installer Module
Author: Drmusab
Last Modified: 2025-07-05 21:44:47 UTC

This module provides functionality for installing, updating, and managing skills
for the AI assistant. It handles skill discovery, installation, validation,
dependency management, and registration with the skill registry.
"""

import hashlib
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

import aiohttp
import asyncio
import git
import pkg_resources
import yaml
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

# Assistant components
from src.assistant.core import ComponentManager
from src.assistant.core import PluginManager

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    SkillInstallationCompleted,
    SkillInstallationFailed,
    SkillInstallationStarted,
    SkillUninstallationCompleted,
    SkillUninstallationFailed,
    SkillUninstallationStarted,
    SkillUpdateCompleted,
    SkillUpdateFailed,
    SkillUpdateStarted,
    SystemConfigurationChanged,
)

# Storage components
from src.integrations.storage.file_storage import FileStorage

# Observability
from src.observability.logging.config import get_logger
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager
from src.skills.skill_factory import SkillFactory

# Skill management
from src.skills.skill_registry import SkillRegistry
from src.skills.skill_validator import SkillValidator


class SkillSource(Enum):
    """Sources from which skills can be installed."""

    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"
    GIT_REPOSITORY = "git_repository"
    MARKETPLACE = "marketplace"
    REGISTRY = "registry"
    ZIP_ARCHIVE = "zip_archive"
    URL = "url"


class SkillCategory(Enum):
    """Categories of skills."""

    CORE = "core"
    PRODUCTIVITY = "productivity"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    COMMUNICATION = "communication"
    INFORMATION = "information"
    UTILITY = "utility"
    CUSTOM = "custom"
    META = "meta"


class SkillInstallationStatus(Enum):
    """Status of skill installation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    INSTALLING_DEPENDENCIES = "installing_dependencies"
    COPYING_FILES = "copying_files"
    REGISTERING = "registering"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class SkillMetadata:
    """Metadata for a skill."""

    name: str
    version: str
    description: str
    author: str
    license: str
    category: str
    requires_auth: bool = False
    dependencies: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    compatible_platforms: List[str] = field(default_factory=list)
    min_system_version: str = "0.1.0"
    source_url: Optional[str] = None
    documentation_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    installed_at: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class SkillInstallationOptions:
    """Options for skill installation."""

    source_type: SkillSource
    source_path: str
    target_directory: Optional[str] = None
    install_dependencies: bool = True
    force_reinstall: bool = False
    skip_validation: bool = False
    user_id: Optional[str] = None
    custom_name: Optional[str] = None
    branch: Optional[str] = None
    version_tag: Optional[str] = None
    timeout: int = 300  # seconds


class SkillInstaller:
    """
    Handles the installation, updating, and management of skills.

    This class provides functionality to:
    - Install skills from various sources (local files, git repos, marketplace)
    - Validate skills for security and compatibility
    - Manage skill dependencies
    - Register skills with the skill registry
    - Update and uninstall skills
    """

    def __init__(self, container: Container):
        """
        Initialize the skill installer.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.config = container.get(ConfigLoader)
        self.event_bus = container.get(EventBus)
        self.error_handler = container.get(ErrorHandler)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Skill management components
        self.skill_registry = container.get(SkillRegistry)
        self.skill_factory = container.get(SkillFactory)
        self.skill_validator = container.get(SkillValidator)

        # System components
        self.component_manager = container.get(ComponentManager)
        self.plugin_manager = container.get(PluginManager)

        # Storage
        self.file_storage = container.get(FileStorage)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Set up paths
        self.base_skills_path = self._get_skills_directory()
        self.custom_skills_path = self.base_skills_path / "custom"
        self.builtin_skills_path = self.base_skills_path / "builtin"
        self.meta_skills_path = self.base_skills_path / "meta_skills"
        self.skill_templates_path = self.base_skills_path / "custom" / "skill_templates"

        # Ensure required directories exist
        self._ensure_directories()

        # Register metrics
        if self.metrics:
            self.metrics.register_counter("skill_installations_total")
            self.metrics.register_counter("skill_installations_failed")
            self.metrics.register_counter("skill_uninstallations_total")
            self.metrics.register_counter("skill_updates_total")
            self.metrics.register_histogram("skill_installation_time_seconds")

    def _get_skills_directory(self) -> Path:
        """Get the base directory for skills."""
        # First try to get from config
        skills_dir = self.config.get("skills.directory", None)
        if skills_dir:
            return Path(skills_dir)

        # Default to src/skills
        return Path("src") / "skills"

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.base_skills_path,
            self.custom_skills_path,
            self.builtin_skills_path,
            self.meta_skills_path,
            self.skill_templates_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            # Ensure __init__.py exists
            init_file = directory / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    async def install_skill(self, options: SkillInstallationOptions) -> Dict[str, Any]:
        """
        Install a skill from the specified source.

        Args:
            options: Installation options

        Returns:
            Installation result information
        """
        installation_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        status = SkillInstallationStatus.PENDING

        # Track metrics
        if self.metrics:
            self.metrics.increment("skill_installations_total")

        # Emit installation started event
        await self.event_bus.emit(
            SkillInstallationStarted(
                installation_id=installation_id,
                source_type=options.source_type.value,
                source_path=options.source_path,
            )
        )

        # Default target directory is custom skills if not specified
        if not options.target_directory:
            options.target_directory = str(self.custom_skills_path)

        # Create temporary directory for installation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = {
                "installation_id": installation_id,
                "source_type": options.source_type.value,
                "source_path": options.source_path,
                "target_directory": options.target_directory,
                "status": status.value,
                "started_at": start_time.isoformat(),
                "completed_at": None,
                "duration_seconds": None,
                "skill_metadata": None,
                "error": None,
            }

            try:
                # Step 1: Download/Copy skill to temporary directory
                status = SkillInstallationStatus.IN_PROGRESS
                result["status"] = status.value

                self.logger.info(
                    f"Installing skill from {options.source_type.value}: {options.source_path}"
                )
                await self._download_skill(options, temp_path)

                # Step 2: Validate the skill
                if not options.skip_validation:
                    status = SkillInstallationStatus.VALIDATING
                    result["status"] = status.value

                    self.logger.info(f"Validating skill from {temp_path}")
                    await self._validate_skill(temp_path, options)

                # Step 3: Read skill metadata
                metadata = await self._extract_skill_metadata(temp_path)
                result["skill_metadata"] = vars(metadata)

                # Generate checksum for the skill
                metadata.checksum = await self._generate_skill_checksum(temp_path)

                # Add installation timestamp
                metadata.installed_at = datetime.now(timezone.utc).isoformat()

                # Step 4: Install dependencies if needed
                if options.install_dependencies and metadata.dependencies:
                    status = SkillInstallationStatus.INSTALLING_DEPENDENCIES
                    result["status"] = status.value

                    self.logger.info(f"Installing dependencies for skill: {metadata.name}")
                    await self._install_dependencies(metadata.dependencies)

                # Step 5: Copy files to target directory
                status = SkillInstallationStatus.COPYING_FILES
                result["status"] = status.value

                target_path = await self._copy_skill_files(temp_path, options, metadata)

                # Step 6: Register the skill
                status = SkillInstallationStatus.REGISTERING
                result["status"] = status.value

                await self._register_skill(target_path, metadata)

                # Step 7: Update system configuration
                await self._update_system_configuration(metadata)

                # Successful installation
                status = SkillInstallationStatus.COMPLETED
                result["status"] = status.value

                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                result["completed_at"] = end_time.isoformat()
                result["duration_seconds"] = duration

                # Log success
                self.logger.info(
                    f"Successfully installed skill {metadata.name} v{metadata.version} "
                    f"in {duration:.2f} seconds"
                )

                # Emit completion event
                await self.event_bus.emit(
                    SkillInstallationCompleted(
                        installation_id=installation_id,
                        skill_name=metadata.name,
                        skill_version=metadata.version,
                        execution_time=duration,
                    )
                )

                if self.metrics:
                    self.metrics.record("skill_installation_time_seconds", duration)

                return result

            except Exception as e:
                # Handle installation failure
                status = SkillInstallationStatus.FAILED
                result["status"] = status.value
                result["error"] = str(e)

                end_time = datetime.now(timezone.utc)
                duration = (end_time - start_time).total_seconds()

                result["completed_at"] = end_time.isoformat()
                result["duration_seconds"] = duration

                # Log error
                self.logger.error(f"Skill installation failed: {str(e)}")
                self.logger.error(traceback.format_exc())

                # Emit failure event
                await self.event_bus.emit(
                    SkillInstallationFailed(
                        installation_id=installation_id,
                        error_message=str(e),
                        execution_time=duration,
                    )
                )

                if self.metrics:
                    self.metrics.increment("skill_installations_failed")

                # Attempt rollback
                await self._rollback_installation(options, result.get("skill_metadata"))

                return result

    async def uninstall_skill(self, skill_name: str, preserve_data: bool = False) -> Dict[str, Any]:
        """
        Uninstall a skill by name.

        Args:
            skill_name: Name of the skill to uninstall
            preserve_data: Whether to preserve skill data

        Returns:
            Uninstallation result information
        """
        uninstallation_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment("skill_uninstallations_total")

        # Emit uninstallation started event
        await self.event_bus.emit(
            SkillUninstallationStarted(uninstallation_id=uninstallation_id, skill_name=skill_name)
        )

        result = {
            "uninstallation_id": uninstallation_id,
            "skill_name": skill_name,
            "status": "pending",
            "started_at": start_time.isoformat(),
            "completed_at": None,
            "duration_seconds": None,
            "error": None,
        }

        try:
            # Check if skill exists
            skill_info = self.skill_registry.get_skill_info(skill_name)
            if not skill_info:
                raise ValueError(f"Skill '{skill_name}' is not installed")

            # Get skill path
            skill_path = Path(skill_info.get("path", ""))
            if not skill_path.exists():
                raise ValueError(f"Skill path '{skill_path}' does not exist")

            self.logger.info(f"Uninstalling skill: {skill_name} from {skill_path}")

            # Check if it's a built-in skill
            is_builtin = str(skill_path).startswith(str(self.builtin_skills_path))
            if is_builtin:
                raise ValueError(f"Cannot uninstall built-in skill '{skill_name}'")

            # Unregister the skill
            self.skill_registry.unregister_skill(skill_name)

            # Remove skill files
            if skill_path.is_dir():
                if preserve_data:
                    # Only remove Python files and keep data
                    for file in skill_path.glob("*.py"):
                        file.unlink()
                else:
                    # Remove entire directory
                    shutil.rmtree(skill_path)
            else:
                skill_path.unlink()

            # Update system configuration
            await self._update_system_configuration_after_uninstall(skill_name)

            # Successful uninstallation
            result["status"] = "completed"

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration

            # Log success
            self.logger.info(
                f"Successfully uninstalled skill {skill_name} in {duration:.2f} seconds"
            )

            # Emit completion event
            await self.event_bus.emit(
                SkillUninstallationCompleted(
                    uninstallation_id=uninstallation_id,
                    skill_name=skill_name,
                    execution_time=duration,
                )
            )

            return result

        except Exception as e:
            # Handle uninstallation failure
            result["status"] = "failed"
            result["error"] = str(e)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration

            # Log error
            self.logger.error(f"Skill uninstallation failed: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Emit failure event
            await self.event_bus.emit(
                SkillUninstallationFailed(
                    uninstallation_id=uninstallation_id,
                    skill_name=skill_name,
                    error_message=str(e),
                    execution_time=duration,
                )
            )

            return result

    async def update_skill(self, skill_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Update a skill to a newer version.

        Args:
            skill_name: Name of the skill to update
            version: Target version (if None, updates to latest)

        Returns:
            Update result information
        """
        update_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)

        # Track metrics
        if self.metrics:
            self.metrics.increment("skill_updates_total")

        # Emit update started event
        await self.event_bus.emit(
            SkillUpdateStarted(update_id=update_id, skill_name=skill_name, target_version=version)
        )

        result = {
            "update_id": update_id,
            "skill_name": skill_name,
            "status": "pending",
            "previous_version": None,
            "new_version": version,
            "started_at": start_time.isoformat(),
            "completed_at": None,
            "duration_seconds": None,
            "error": None,
        }

        try:
            # Check if skill exists
            skill_info = self.skill_registry.get_skill_info(skill_name)
            if not skill_info:
                raise ValueError(f"Skill '{skill_name}' is not installed")

            result["previous_version"] = skill_info.get("version")

            # Get skill metadata
            metadata = await self._get_skill_metadata(skill_name)

            # Get skill path
            skill_path = Path(skill_info.get("path", ""))
            if not skill_path.exists():
                raise ValueError(f"Skill path '{skill_path}' does not exist")

            self.logger.info(
                f"Updating skill: {skill_name} from version {result['previous_version']} to {version or 'latest'}"
            )

            # Check if it's a custom skill with source URL
            if not metadata.source_url:
                raise ValueError(f"Cannot update skill '{skill_name}' without source URL")

            # Create temporary directory for update
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create options for update
                update_options = SkillInstallationOptions(
                    source_type=(
                        SkillSource.GIT_REPOSITORY
                        if metadata.source_url.endswith(".git")
                        else SkillSource.URL
                    ),
                    source_path=metadata.source_url,
                    target_directory=str(skill_path.parent),
                    install_dependencies=True,
                    force_reinstall=True,
                    skip_validation=False,
                    version_tag=version,
                )

                # Download updated skill
                await self._download_skill(update_options, temp_path)

                # Validate the skill
                await self._validate_skill(temp_path, update_options)

                # Extract updated metadata
                updated_metadata = await self._extract_skill_metadata(temp_path)

                # Check if update is necessary
                if version is None or pkg_resources.parse_version(
                    updated_metadata.version
                ) > pkg_resources.parse_version(metadata.version):
                    # Backup current skill
                    backup_path = skill_path.with_suffix(".bak")
                    if skill_path.is_dir():
                        if backup_path.exists():
                            shutil.rmtree(backup_path)
                        shutil.copytree(skill_path, backup_path)
                    else:
                        shutil.copy2(skill_path, backup_path)

                    try:
                        # Unregister current skill
                        self.skill_registry.unregister_skill(skill_name)

                        # Install dependencies if needed
                        if updated_metadata.dependencies:
                            await self._install_dependencies(updated_metadata.dependencies)

                        # Copy files
                        target_path = await self._copy_skill_files(
                            temp_path, update_options, updated_metadata
                        )

                        # Register updated skill
                        await self._register_skill(target_path, updated_metadata)

                        # Update system configuration
                        await self._update_system_configuration(updated_metadata)

                        # Update result
                        result["new_version"] = updated_metadata.version
                        result["status"] = "completed"

                    except Exception as e:
                        # Restore from backup on error
                        self.logger.error(f"Error during update, restoring from backup: {str(e)}")

                        if skill_path.is_dir():
                            if skill_path.exists():
                                shutil.rmtree(skill_path)
                            shutil.copytree(backup_path, skill_path)
                        else:
                            if skill_path.exists():
                                skill_path.unlink()
                            shutil.copy2(backup_path, skill_path)

                        # Re-register original skill
                        await self._register_skill(skill_path, metadata)

                        raise

                    finally:
                        # Clean up backup
                        if backup_path.exists():
                            if backup_path.is_dir():
                                shutil.rmtree(backup_path)
                            else:
                                backup_path.unlink()
                else:
                    self.logger.info(
                        f"Skill '{skill_name}' is already at the latest version: {metadata.version}"
                    )
                    result["status"] = "completed"
                    result["new_version"] = metadata.version

            # Successful update
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration

            # Log success
            self.logger.info(
                f"Successfully updated skill {skill_name} from {result['previous_version']} "
                f"to {result['new_version']} in {duration:.2f} seconds"
            )

            # Emit completion event
            await self.event_bus.emit(
                SkillUpdateCompleted(
                    update_id=update_id,
                    skill_name=skill_name,
                    previous_version=result["previous_version"],
                    new_version=result["new_version"],
                    execution_time=duration,
                )
            )

            return result

        except Exception as e:
            # Handle update failure
            result["status"] = "failed"
            result["error"] = str(e)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result["completed_at"] = end_time.isoformat()
            result["duration_seconds"] = duration

            # Log error
            self.logger.error(f"Skill update failed: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Emit failure event
            await self.event_bus.emit(
                SkillUpdateFailed(
                    update_id=update_id,
                    skill_name=skill_name,
                    error_message=str(e),
                    execution_time=duration,
                )
            )

            return result

    async def list_installed_skills(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all installed skills, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            List of skill information dictionaries
        """
        skills = self.skill_registry.list_skills()

        if category:
            skills = [
                skill for skill in skills if skill.get("category", "").lower() == category.lower()
            ]

        # Enhance with more metadata
        for skill in skills:
            try:
                metadata = await self._get_skill_metadata(skill["name"])
                skill.update(
                    {
                        "description": metadata.description,
                        "author": metadata.author,
                        "license": metadata.license,
                        "dependencies": metadata.dependencies,
                        "installed_at": metadata.installed_at,
                        "tags": metadata.tags,
                    }
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not get full metadata for skill {skill['name']}: {str(e)}"
                )

        return skills

    async def get_skill_details(self, skill_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Detailed skill information
        """
        skill_info = self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            raise ValueError(f"Skill '{skill_name}' is not installed")

        # Get full metadata
        metadata = await self._get_skill_metadata(skill_name)

        # Check if source code is available
        skill_path = Path(skill_info.get("path", ""))
        source_code = None
        if skill_path.exists() and skill_path.is_file() and skill_path.suffix == ".py":
            with open(skill_path, "r", encoding="utf-8") as f:
                source_code = f.read()

        # Combine info
        details = {
            **skill_info,
            "metadata": vars(metadata),
            "source_code": source_code,
            "is_builtin": str(skill_path).startswith(str(self.builtin_skills_path)),
            "is_meta": str(skill_path).startswith(str(self.meta_skills_path)),
            "is_custom": str(skill_path).startswith(str(self.custom_skills_path)),
        }

        return details

    async def verify_skill_integrity(self, skill_name: str) -> Dict[str, Any]:
        """
        Verify the integrity of an installed skill.

        Args:
            skill_name: Name of the skill to verify

        Returns:
            Verification result
        """
        skill_info = self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            raise ValueError(f"Skill '{skill_name}' is not installed")

        metadata = await self._get_skill_metadata(skill_name)
        skill_path = Path(skill_info.get("path", ""))

        # Calculate current checksum
        current_checksum = await self._generate_skill_checksum(skill_path)

        # Compare with stored checksum
        original_checksum = metadata.checksum

        result = {
            "skill_name": skill_name,
            "verification_time": datetime.now(timezone.utc).isoformat(),
            "integrity_verified": current_checksum == original_checksum,
            "original_checksum": original_checksum,
            "current_checksum": current_checksum,
        }

        if not result["integrity_verified"]:
            self.logger.warning(
                f"Skill integrity check failed for {skill_name}. "
                f"Original: {original_checksum}, Current: {current_checksum}"
            )

        return result

    async def _download_skill(self, options: SkillInstallationOptions, temp_path: Path) -> None:
        """
        Download or copy skill files to temporary directory.

        Args:
            options: Installation options
            temp_path: Temporary directory path
        """
        if options.source_type == SkillSource.LOCAL_FILE:
            await self._copy_from_local_file(options.source_path, temp_path)

        elif options.source_type == SkillSource.LOCAL_DIRECTORY:
            await self._copy_from_local_directory(options.source_path, temp_path)

        elif options.source_type == SkillSource.GIT_REPOSITORY:
            await self._clone_from_git(
                options.source_path, temp_path, options.branch, options.version_tag
            )

        elif options.source_type == SkillSource.ZIP_ARCHIVE:
            await self._extract_from_zip(options.source_path, temp_path)

        elif options.source_type == SkillSource.URL:
            await self._download_from_url(options.source_path, temp_path)

        elif (
            options.source_type == SkillSource.MARKETPLACE
            or options.source_type == SkillSource.REGISTRY
        ):
            await self._download_from_marketplace(
                options.source_path, temp_path, options.version_tag
            )

        else:
            raise ValueError(f"Unsupported source type: {options.source_type}")

    async def _copy_from_local_file(self, source_path: str, temp_path: Path) -> None:
        """Copy a local file to the temporary directory."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        if source.suffix != ".py":
            raise ValueError(f"Source file must be a Python file (.py): {source_path}")

        # Copy the file
        shutil.copy2(source, temp_path / source.name)

    async def _copy_from_local_directory(self, source_path: str, temp_path: Path) -> None:
        """Copy a local directory to the temporary directory."""
        source = Path(source_path)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        # Check if there are Python files
        python_files = list(source.glob("**/*.py"))
        if not python_files:
            raise ValueError(f"No Python files found in source directory: {source_path}")

        # Copy the directory contents
        for item in source.glob("**/*"):
            if item.is_file():
                relative_path = item.relative_to(source)
                target_file = temp_path / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_file)

    async def _clone_from_git(
        self,
        repo_url: str,
        temp_path: Path,
        branch: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Clone a git repository to the temporary directory."""
        try:
            # Clone options
            clone_args = ["--depth", "1"]  # Shallow clone by default

            if branch:
                clone_args.extend(["--branch", branch])
            elif tag:
                clone_args.extend(["--branch", tag])

            # Clone the repository
            Repo.clone_from(repo_url, temp_path, multi_options=clone_args)

            # Check if there are Python files
            python_files = list(temp_path.glob("**/*.py"))
            if not python_files:
                raise ValueError(f"No Python files found in git repository: {repo_url}")

            # Remove .git directory to save space
            git_dir = temp_path / ".git"
            if git_dir.exists():
                shutil.rmtree(git_dir)

        except GitCommandError as e:
            raise ValueError(f"Git repository cloning failed: {str(e)}")

    async def _extract_from_zip(self, zip_path: str, temp_path: Path) -> None:
        """Extract a ZIP archive to the temporary directory."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Check if there are Python files
            python_files = list(temp_path.glob("**/*.py"))
            if not python_files:
                raise ValueError(f"No Python files found in ZIP archive: {zip_path}")

        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {zip_path}")

    async def _download_from_url(self, url: str, temp_path: Path) -> None:
        """Download a file from a URL to the temporary directory."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(
                            f"Failed to download from URL: {url}, status: {response.status}"
                        )

                    # Determine filename from URL or headers
                    filename = url.split("/")[-1]
                    if "content-disposition" in response.headers:
                        content_disposition = response.headers["content-disposition"]
                        if "filename=" in content_disposition:
                            filename = content_disposition.split("filename=")[1].strip('"')

                    # Save the file
                    target_file = temp_path / filename
                    content = await response.read()

                    with open(target_file, "wb") as f:
                        f.write(content)

                    # If it's a ZIP file, extract it
                    if filename.endswith(".zip"):
                        await self._extract_from_zip(str(target_file), temp_path)
                        target_file.unlink()  # Remove the original zip

                    # Check if there are Python files
                    python_files = list(temp_path.glob("**/*.py"))
                    if not python_files:
                        raise ValueError(f"No Python files found in downloaded content: {url}")

        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to download from URL: {url}, error: {str(e)}")

    async def _download_from_marketplace(
        self, skill_id: str, temp_path: Path, version: Optional[str] = None
    ) -> None:
        """Download a skill from the marketplace or registry."""
        # This is a placeholder for future marketplace/registry implementation
        # For now, we'll raise an error since this isn't implemented yet
        raise NotImplementedError(
            f"Marketplace/registry downloads are not yet implemented. "
            f"Cannot download skill: {skill_id}, version: {version or 'latest'}"
        )

    async def _validate_skill(self, skill_path: Path, options: SkillInstallationOptions) -> None:
        """
        Validate a skill for security and compatibility.

        Args:
            skill_path: Path to the skill files
            options: Installation options

        Raises:
            ValueError: If validation fails
        """
        # Use the skill validator
        validation_result = await self.skill_validator.validate_skill_source(skill_path)

        if not validation_result["valid"]:
            raise ValueError(
                f"Skill validation failed: {validation_result.get('error', 'Unknown validation error')}\n"
                f"Details: {json.dumps(validation_result.get('details', {}), indent=2)}"
            )

        # Check system compatibility
        metadata = await self._extract_skill_metadata(skill_path)
        system_version = self.config.get("system.version", "0.1.0")

        if pkg_resources.parse_version(metadata.min_system_version) > pkg_resources.parse_version(
            system_version
        ):
            raise ValueError(
                f"Skill requires minimum system version {metadata.min_system_version}, "
                f"but current version is {system_version}"
            )

        # Check if skill is already installed
        if not options.force_reinstall:
            existing_skill = self.skill_registry.get_skill_info(metadata.name)
            if existing_skill:
                raise ValueError(
                    f"Skill '{metadata.name}' is already installed. "
                    f"Use force_reinstall=True to reinstall."
                )

    async def _extract_skill_metadata(self, skill_path: Path) -> SkillMetadata:
        """
        Extract metadata from skill files.

        Args:
            skill_path: Path to the skill files

        Returns:
            Skill metadata
        """
        # Look for metadata in different possible locations
        metadata_locations = [
            skill_path / "metadata.json",
            skill_path / "metadata.yaml",
            skill_path / "metadata.yml",
            skill_path / "skill.json",
            skill_path / "skill.yaml",
            skill_path / "skill.yml",
        ]

        # Check if skill_path is a file
        if skill_path.is_file():
            return await self._extract_metadata_from_python_file(skill_path)

        # Try to find metadata file
        for metadata_file in metadata_locations:
            if metadata_file.exists():
                return await self._parse_metadata_file(metadata_file)

        # If no metadata file found, look for Python files
        python_files = list(skill_path.glob("*.py"))
        if not python_files:
            raise ValueError(f"No Python files found in: {skill_path}")

        # Try to extract from Python file
        main_file = next((f for f in python_files if f.stem.lower() == "skill"), python_files[0])
        return await self._extract_metadata_from_python_file(main_file)

    async def _parse_metadata_file(self, metadata_file: Path) -> SkillMetadata:
        """Parse metadata from a JSON or YAML file."""
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                if metadata_file.suffix in [".json"]:
                    metadata_dict = json.load(f)
                elif metadata_file.suffix in [".yaml", ".yml"]:
                    metadata_dict = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported metadata file format: {metadata_file.suffix}")

            # Create metadata object with defaults
            return SkillMetadata(
                name=metadata_dict.get("name", "unknown"),
                version=metadata_dict.get("version", "0.1.0"),
                description=metadata_dict.get("description", ""),
                author=metadata_dict.get("author", "unknown"),
                license=metadata_dict.get("license", "unknown"),
                category=metadata_dict.get("category", SkillCategory.CUSTOM.value),
                requires_auth=metadata_dict.get("requires_auth", False),
                dependencies=metadata_dict.get("dependencies", []),
                permissions=metadata_dict.get("permissions", []),
                compatible_platforms=metadata_dict.get("compatible_platforms", []),
                min_system_version=metadata_dict.get("min_system_version", "0.1.0"),
                source_url=metadata_dict.get("source_url"),
                documentation_url=metadata_dict.get("documentation_url"),
                tags=metadata_dict.get("tags", []),
                created_at=metadata_dict.get("created_at"),
                updated_at=metadata_dict.get("updated_at"),
            )

        except Exception as e:
            self.logger.error(f"Error parsing metadata file {metadata_file}: {str(e)}")
            raise ValueError(f"Invalid metadata file: {str(e)}")

    async def _extract_metadata_from_python_file(self, python_file: Path) -> SkillMetadata:
        """Extract metadata from a Python file by parsing docstrings and class attributes."""
        try:
            # Read the file content
            with open(python_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Default metadata
            metadata = {
                "name": python_file.stem,
                "version": "0.1.0",
                "description": "",
                "author": "unknown",
                "license": "unknown",
                "category": SkillCategory.CUSTOM.value,
                "requires_auth": False,
                "dependencies": [],
                "permissions": [],
                "compatible_platforms": [],
                "min_system_version": "0.1.0",
                "source_url": None,
                "documentation_url": None,
                "tags": [],
            }

            # Extract module docstring
            module_docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if module_docstring_match:
                docstring = module_docstring_match.group(1).strip()

                # Look for metadata in docstring
                author_match = re.search(r"Author:\s*(.+?)$", docstring, re.MULTILINE)
                if author_match:
                    metadata["author"] = author_match.group(1).strip()

                version_match = re.search(r"Version:\s*(.+?)$", docstring, re.MULTILINE)
                if version_match:
                    metadata["version"] = version_match.group(1).strip()

                description_lines = []
                for line in docstring.split("\n"):
                    if not line.strip() or any(
                        marker in line for marker in ["Author:", "Version:", "License:"]
                    ):
                        continue
                    description_lines.append(line.strip())

                if description_lines:
                    metadata["description"] = " ".join(description_lines)

            # Look for class attributes
            class_pattern = r"class\s+(\w+).*?:"
            class_matches = re.finditer(class_pattern, content)

            for class_match in class_matches:
                class_name = class_match.group(1)
                class_start_pos = class_match.end()

                # Find class end (next class or end of file)
                next_class_match = re.search(class_pattern, content[class_start_pos:])
                if next_class_match:
                    class_end_pos = class_start_pos + next_class_match.start()
                else:
                    class_end_pos = len(content)

                class_content = content[class_start_pos:class_end_pos]

                # Look for get_skill_id method
                skill_id_match = re.search(
                    r'def\s+get_skill_id.*?return\s+[\'"](.+?)[\'"]', class_content, re.DOTALL
                )
                if skill_id_match:
                    metadata["name"] = skill_id_match.group(1)

                # Look for get_skill_description method
                skill_desc_match = re.search(
                    r'def\s+get_skill_description.*?return\s+[\'"](.+?)[\'"]',
                    class_content,
                    re.DOTALL,
                )
                if skill_desc_match and not metadata["description"]:
                    metadata["description"] = skill_desc_match.group(1)

                # Look for get_skill_category method
                skill_cat_match = re.search(
                    r'def\s+get_skill_category.*?return\s+[\'"](.+?)[\'"]', class_content, re.DOTALL
                )
                if skill_cat_match:
                    metadata["category"] = skill_cat_match.group(1)

            # Create metadata object
            return SkillMetadata(**metadata)

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {python_file}: {str(e)}")
            # Return minimal metadata
            return SkillMetadata(
                name=python_file.stem,
                version="0.1.0",
                description="",
                author="unknown",
                license="unknown",
                category=SkillCategory.CUSTOM.value,
            )

    async def _install_dependencies(self, dependencies: List[str]) -> None:
        """
        Install Python package dependencies.

        Args:
            dependencies: List of dependencies to install
        """
        if not dependencies:
            return

        self.logger.info(f"Installing dependencies: {dependencies}")

        # Check if dependencies are already installed
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        to_install = []

        for dep in dependencies:
            pkg_name = dep.split("==")[0].split(">=")[0].split("<=")[0].strip()
            if pkg_name not in installed_packages:
                to_install.append(dep)

        if not to_install:
            self.logger.info("All dependencies are already installed")
            return

        # Install using pip in a subprocess
        try:
            # Prepare command
            cmd = [sys.executable, "-m", "pip", "install", "--user"]
            cmd.extend(to_install)

            # Run pip install
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise ValueError(f"Failed to install dependencies: {error_msg}")

            self.logger.info(f"Successfully installed dependencies: {to_install}")

        except Exception as e:
            self.logger.error(f"Error installing dependencies: {str(e)}")
            raise ValueError(f"Failed to install dependencies: {str(e)}")

    async def _copy_skill_files(
        self, source_path: Path, options: SkillInstallationOptions, metadata: SkillMetadata
    ) -> Path:
        """
        Copy skill files to the target directory.

        Args:
            source_path: Source path containing skill files
            options: Installation options
            metadata: Skill metadata

        Returns:
            Path to the installed skill
        """
        # Determine target directory
        target_dir = Path(options.target_directory)

        # Use custom name if provided
        skill_name = options.custom_name or metadata.name

        # Sanitize skill name for directory
        safe_name = re.sub(r"[^\w\-\.]", "_", skill_name)

        # Create target skill directory
        skill_dir = target_dir / safe_name

        # Check if target exists and handle accordingly
        if skill_dir.exists():
            if options.force_reinstall:
                self.logger.info(f"Removing existing skill directory: {skill_dir}")
                if skill_dir.is_dir():
                    shutil.rmtree(skill_dir)
                else:
                    skill_dir.unlink()
            else:
                raise ValueError(f"Skill directory already exists: {skill_dir}")

        # Determine if source is a file or directory
        if source_path.is_file():
            # Create parent directory
            skill_dir.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file
            target_file = target_dir / f"{safe_name}.py"
            shutil.copy2(source_path, target_file)

            return target_file
        else:
            # Copy directory contents
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for item in source_path.glob("**/*"):
                if item.is_file():
                    relative_path = item.relative_to(source_path)
                    target_file = skill_dir / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_file)

            # Create metadata file if it doesn't exist
            metadata_file = skill_dir / "metadata.json"
            if not metadata_file.exists():
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(vars(metadata), f, indent=2)

            return skill_dir

    async def _register_skill(self, skill_path: Path, metadata: SkillMetadata) -> None:
        """
        Register the skill with the skill registry.

        Args:
            skill_path: Path to the installed skill
            metadata: Skill metadata
        """
        # Find the main skill file
        if skill_path.is_file():
            main_file = skill_path
        else:
            # Look for the main skill file
            python_files = list(skill_path.glob("*.py"))
            if not python_files:
                raise ValueError(f"No Python files found in skill directory: {skill_path}")

            # Prioritize files that might be the main skill file
            for pattern in ["skill.py", f"{metadata.name.lower()}.py"]:
                main_file_candidates = [f for f in python_files if f.name.lower() == pattern]
                if main_file_candidates:
                    main_file = main_file_candidates[0]
                    break
            else:
                # Just use the first Python file
                main_file = python_files[0]

        # Register with skill registry
        module_path = str(main_file).replace(os.sep, ".").replace(".py", "")
        if module_path.startswith("src."):
            module_path = module_path[4:]  # Remove "src." prefix

        self.logger.info(f"Registering skill {metadata.name} from module {module_path}")

        self.skill_registry.register_skill(
            name=metadata.name,
            module_path=module_path,
            path=str(main_file),
            version=metadata.version,
            category=metadata.category,
        )

    async def _update_system_configuration(self, metadata: SkillMetadata) -> None:
        """
        Update system configuration after skill installation.

        Args:
            metadata: Skill metadata
        """
        # Emit configuration changed event
        await self.event_bus.emit(
            SystemConfigurationChanged(
                component="skills",
                change_type="skill_installed",
                details={
                    "skill_name": metadata.name,
                    "skill_version": metadata.version,
                    "skill_category": metadata.category,
                },
            )
        )

        # Update component manager (if needed)
        try:
            if hasattr(self.component_manager, "reload_skills"):
                await self.component_manager.reload_skills()
        except Exception as e:
            self.logger.warning(f"Error updating component manager: {str(e)}")

    async def _update_system_configuration_after_uninstall(self, skill_name: str) -> None:
        """
        Update system configuration after skill uninstallation.

        Args:
            skill_name: Name of the uninstalled skill
        """
        # Emit configuration changed event
        await self.event_bus.emit(
            SystemConfigurationChanged(
                component="skills",
                change_type="skill_uninstalled",
                details={"skill_name": skill_name},
            )
        )

        # Update component manager (if needed)
        try:
            if hasattr(self.component_manager, "reload_skills"):
                await self.component_manager.reload_skills()
        except Exception as e:
            self.logger.warning(f"Error updating component manager: {str(e)}")

    async def _rollback_installation(
        self, options: SkillInstallationOptions, metadata: Optional[Dict[str, Any]]
    ) -> None:
        """
        Roll back a failed installation.

        Args:
            options: Installation options
            metadata: Skill metadata (if available)
        """
        self.logger.info("Rolling back failed installation")

        try:
            if metadata:
                skill_name = metadata.get("name")
                if skill_name:
                    # Try to unregister from registry
                    try:
                        self.skill_registry.unregister_skill(skill_name)
                    except Exception as e:
                        self.logger.warning(f"Error unregistering skill during rollback: {str(e)}")

                    # Try to remove installed files
                    target_dir = Path(options.target_directory)
                    skill_dir = target_dir / skill_name

                    if skill_dir.exists():
                        if skill_dir.is_dir():
                            shutil.rmtree(skill_dir)
                        else:
                            skill_dir.unlink()
        except Exception as e:
            self.logger.error(f"Error during installation rollback: {str(e)}")

    async def _get_skill_metadata(self, skill_name: str) -> SkillMetadata:
        """
        Get metadata for an installed skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Skill metadata
        """
        skill_info = self.skill_registry.get_skill_info(skill_name)
        if not skill_info:
            raise ValueError(f"Skill '{skill_name}' is not installed")

        skill_path = Path(skill_info.get("path", ""))
        if not skill_path.exists():
            raise ValueError(f"Skill path '{skill_path}' does not exist")

        # Get metadata
        if skill_path.is_file():
            return await self._extract_metadata_from_python_file(skill_path)
        else:
            skill_dir = skill_path.parent if skill_path.name.endswith(".py") else skill_path

            # Look for metadata file
            metadata_file = skill_dir / "metadata.json"
            if metadata_file.exists():
                return await self._parse_metadata_file(metadata_file)

            # Try YAML alternatives
            for ext in [".yaml", ".yml"]:
                metadata_file = skill_dir / f"metadata{ext}"
                if metadata_file.exists():
                    return await self._parse_metadata_file(metadata_file)

            # Extract from Python file as fallback
            return await self._extract_metadata_from_python_file(skill_path)

    async def _generate_skill_checksum(self, skill_path: Path) -> str:
        """
        Generate a checksum for a skill's files.

        Args:
            skill_path: Path to the skill

        Returns:
            Checksum string
        """
        hasher = hashlib.sha256()

        if skill_path.is_file():
            # Single file checksum
            with open(skill_path, "rb") as f:
                hasher.update(f.read())
        else:
            # Directory checksum (combine all Python files)
            for python_file in sorted(skill_path.glob("**/*.py")):
                with open(python_file, "rb") as f:
                    hasher.update(f.read())

        return hasher.hexdigest()


class SkillInstallerFactory:
    """Factory for creating SkillInstaller instances."""

    @staticmethod
    def create(container: Container) -> SkillInstaller:
        """
        Create a SkillInstaller instance.

        Args:
            container: Dependency injection container

        Returns:
            SkillInstaller instance
        """
        return SkillInstaller(container)
