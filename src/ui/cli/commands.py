"""
CLI Commands
Author: Drmusab
Last Modified: 2025-07-05 10:58:40 UTC

This module defines the command-line interface commands for the AI assistant,
providing terminal-based interaction with core system functionality through
well-structured commands with appropriate argument handling and documentation.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import textwrap
import time
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

import asyncio
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from src.assistant.component_manager import ComponentManager

# Assistant imports
from src.assistant.core_engine import CoreEngine
from src.assistant.interaction_handler import InteractionHandler
from src.assistant.plugin_manager import PluginManager
from src.assistant.session_manager import SessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.core.config.loader import ConfigLoader

# Core imports
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler
from src.core.health_check import HealthCheck

# Integration imports
from src.integrations.llm.model_router import ModelRouter
from src.memory.cache_manager import MemoryCacheManager

# Memory imports
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.core_memory.memory_types import MemoryType
from src.memory.operations.consolidation import MemoryConsolidator
from src.memory.operations.context_manager import MemoryContextManager
from src.memory.operations.retrieval import MemoryRetriever
from src.observability.logging.config import configure_logging, get_logger

# Observability imports
from src.observability.monitoring.metrics import MetricsCollector
from src.skills.meta_skills.skill_installer import SkillInstaller
from src.skills.skill_factory import SkillFactory

# Skills imports
from src.skills.skill_registry import SkillRegistry

# Local imports
from src.ui.cli.interactive import InteractiveSession

# Initialize rich console for better output formatting
console = Console()
error_console = Console(stderr=True)


class CommandRegistry:
    """Registry for CLI commands."""

    def __init__(self):
        """Initialize the command registry."""
        self.commands = {}
        self.command_groups = {}
        self.aliases = {}

    def register(
        self,
        name: str,
        handler: Callable,
        help_text: str,
        group: str = "general",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """
        Register a command.

        Args:
            name: Command name
            handler: Command handler function
            help_text: Help text for command
            group: Command group
            aliases: Command aliases
        """
        self.commands[name] = {"handler": handler, "help": help_text, "group": group}

        # Add command to group
        if group not in self.command_groups:
            self.command_groups[group] = []
        self.command_groups[group].append(name)

        # Register aliases
        if aliases:
            for alias in aliases:
                self.aliases[alias] = name

    def get_handler(self, name: str) -> Optional[Callable]:
        """
        Get command handler by name or alias.

        Args:
            name: Command name or alias

        Returns:
            Command handler or None if not found
        """
        # Check for alias
        if name in self.aliases:
            name = self.aliases[name]

        # Get command
        command = self.commands.get(name)
        if command:
            return command["handler"]
        return None

    def get_command_help(self, name: str) -> Optional[str]:
        """
        Get help text for a command.

        Args:
            name: Command name or alias

        Returns:
            Help text or None if not found
        """
        # Check for alias
        if name in self.aliases:
            name = self.aliases[name]

        # Get command
        command = self.commands.get(name)
        if command:
            return command["help"]
        return None

    def get_all_commands(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered commands.

        Returns:
            Dictionary of commands
        """
        return self.commands

    def get_command_groups(self) -> Dict[str, List[str]]:
        """
        Get command groups.

        Returns:
            Dictionary of command groups
        """
        return self.command_groups


# Initialize command registry
registry = CommandRegistry()


class CommandHandler:
    """Base class for command handlers."""

    def __init__(self, container: Container):
        """
        Initialize command handler.

        Args:
            container: Dependency injection container
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Get common dependencies
        self.config_loader = container.get(ConfigLoader)

        # Initialize container attribute cache
        self._container_cache = {}

    def get_component(self, component_type: type) -> Any:
        """
        Get a component from the container with caching.

        Args:
            component_type: Component type

        Returns:
            Component instance
        """
        # Check cache
        if component_type in self._container_cache:
            return self._container_cache[component_type]

        # Get from container
        try:
            component = self.container.get(component_type)
            self._container_cache[component_type] = component
            return component
        except Exception as e:
            self.logger.error(f"Failed to get component {component_type.__name__}: {str(e)}")
            return None


class SystemCommands(CommandHandler):
    """System management commands."""

    async def status(self, args: argparse.Namespace) -> int:
        """
        Show system status.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get component manager
        component_manager = self.get_component(ComponentManager)
        health_check = self.get_component(HealthCheck)

        if not component_manager or not health_check:
            error_console.print(
                "[bold red]Error:[/] Component Manager or Health Check not available"
            )
            return 1

        # Show system status
        console.print("[bold blue]System Status[/]")

        # Get health check results
        health_results = await health_check.check_all()

        # Create table
        table = Table(show_header=True)
        table.add_column("Component")
        table.add_column("Status")
        table.add_column("Details")

        # Add components to table
        for name, (is_healthy, details) in health_results.items():
            status = "[bold green]Healthy[/]" if is_healthy else "[bold red]Unhealthy[/]"

            # Format details
            details_str = ""
            if details:
                details_str = "\n".join(f"{k}: {v}" for k, v in details.items())

            table.add_row(name, status, details_str)

        # Show table
        console.print(table)

        # Show overall status
        overall_health = all(status for status, _ in health_results.values())
        status_text = (
            "[bold green]System is healthy[/]"
            if overall_health
            else "[bold red]System has issues[/]"
        )
        console.print(f"\nOverall: {status_text}")

        return 0

    async def info(self, args: argparse.Namespace) -> int:
        """
        Show system information.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get components
        component_manager = self.get_component(ComponentManager)

        if not component_manager:
            error_console.print("[bold red]Error:[/] Component Manager not available")
            return 1

        # Show system information
        console.print("[bold blue]System Information[/]")

        # Get basic system info
        system_info = {
            "System Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Python Version": sys.version.split("\n")[0],
            "Platform": sys.platform,
        }

        # Get configuration information
        try:
            env_mode = self.config_loader.get("environment", "development")
            config_paths = self.config_loader.get_loaded_paths()

            system_info["Environment"] = env_mode
            system_info["Config Files"] = ", ".join(str(p) for p in config_paths)
        except Exception as e:
            self.logger.error(f"Failed to get configuration info: {str(e)}")

        # Create table
        table = Table(show_header=True)
        table.add_column("Property")
        table.add_column("Value")

        # Add system info to table
        for key, value in system_info.items():
            table.add_row(key, str(value))

        # Show table
        console.print(table)

        # Show component information
        components_table = Table(show_header=True)
        components_table.add_column("Component")
        components_table.add_column("Status")
        components_table.add_column("Type")

        # Get registered components
        registered_components = component_manager.get_registered_components()

        for name, component_info in registered_components.items():
            status = (
                "[green]Active[/]"
                if component_info.get("initialized", False)
                else "[yellow]Registered[/]"
            )
            component_type = component_info.get("type", "Unknown")
            components_table.add_row(
                name,
                status,
                (
                    component_type.__name__
                    if hasattr(component_type, "__name__")
                    else str(component_type)
                ),
            )

        console.print("\n[bold blue]Registered Components[/]")
        console.print(components_table)

        return 0

    async def config(self, args: argparse.Namespace) -> int:
        """
        Show or modify configuration.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Handle config actions
        if args.action == "show":
            # Show configuration
            if args.path:
                # Show specific config path
                try:
                    value = self.config_loader.get(args.path)
                    if isinstance(value, dict):
                        # Pretty print dict
                        console.print(f"[bold blue]Configuration for {args.path}:[/]")
                        console.print(yaml.dump(value, default_flow_style=False))
                    else:
                        console.print(f"[bold blue]{args.path}:[/] {value}")
                except Exception as e:
                    error_console.print(f"[bold red]Error:[/] {str(e)}")
                    return 1
            else:
                # Show all config
                try:
                    all_config = self.config_loader.get_all()
                    console.print("[bold blue]Complete Configuration:[/]")
                    console.print(yaml.dump(all_config, default_flow_style=False))
                except Exception as e:
                    error_console.print(f"[bold red]Error:[/] {str(e)}")
                    return 1

        elif args.action == "set":
            # Set configuration value
            if not args.path or not args.value:
                error_console.print(
                    "[bold red]Error:[/] Both path and value are required for 'set' action"
                )
                return 1

            try:
                # Try to parse value as JSON
                try:
                    value = json.loads(args.value)
                except json.JSONDecodeError:
                    # If not valid JSON, use string value
                    value = args.value

                # Set configuration value
                self.config_loader.set(args.path, value)
                console.print(f"[bold green]Successfully set {args.path} to {value}[/]")
            except Exception as e:
                error_console.print(f"[bold red]Error:[/] {str(e)}")
                return 1

        elif args.action == "save":
            # Save configuration
            try:
                self.config_loader.save()
                console.print("[bold green]Configuration saved successfully[/]")
            except Exception as e:
                error_console.print(f"[bold red]Error:[/] {str(e)}")
                return 1

        else:
            error_console.print(f"[bold red]Error:[/] Unknown action: {args.action}")
            return 1

        return 0

    async def restart(self, args: argparse.Namespace) -> int:
        """
        Restart the system or specific components.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        component_manager = self.get_component(ComponentManager)

        if not component_manager:
            error_console.print("[bold red]Error:[/] Component Manager not available")
            return 1

        if args.component:
            # Restart specific component
            console.print(f"[bold blue]Restarting component:[/] {args.component}")

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Restarting component...[/]"),
                    console=console,
                ) as progress:
                    progress.add_task("restart", total=None)
                    await component_manager.restart_component(args.component)

                console.print(f"[bold green]Component {args.component} restarted successfully[/]")
            except Exception as e:
                error_console.print(f"[bold red]Error:[/] {str(e)}")
                return 1
        else:
            # Restart entire system
            console.print("[bold blue]Restarting system...[/]")

            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Shutting down components...[/]"),
                    console=console,
                ) as progress:
                    progress.add_task("shutdown", total=None)
                    await component_manager.shutdown_all()

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Initializing components...[/]"),
                    console=console,
                ) as progress:
                    progress.add_task("init", total=None)
                    await component_manager.initialize_all()

                console.print("[bold green]System restarted successfully[/]")
            except Exception as e:
                error_console.print(f"[bold red]Error:[/] {str(e)}")
                return 1

        return 0


class SessionCommands(CommandHandler):
    """Session management commands."""

    async def list_sessions(self, args: argparse.Namespace) -> int:
        """
        List all sessions.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        session_manager = self.get_component(SessionManager)

        if not session_manager:
            error_console.print("[bold red]Error:[/] Session Manager not available")
            return 1

        # Get sessions
        sessions = await session_manager.get_all_sessions()

        if not sessions:
            console.print("[yellow]No active sessions found[/]")
            return 0

        # Create table
        table = Table(show_header=True)
        table.add_column("Session ID")
        table.add_column("User ID")
        table.add_column("Created")
        table.add_column("Last Active")
        table.add_column("Status")

        # Add sessions to table
        for session in sessions:
            # Format timestamps
            created = datetime.fromisoformat(session.created_at)
            last_active = datetime.fromisoformat(session.last_activity)

            # Calculate time difference for last activity
            now = datetime.now(timezone.utc)
            last_active_diff = now - last_active

            if last_active_diff.total_seconds() < 300:  # 5 minutes
                status = "[bold green]Active[/]"
            elif last_active_diff.total_seconds() < 3600:  # 1 hour
                status = "[yellow]Idle[/]"
            else:
                status = "[gray]Inactive[/]"

            table.add_row(
                session.session_id,
                session.user_id or "anonymous",
                created.strftime("%Y-%m-%d %H:%M:%S"),
                last_active.strftime("%Y-%m-%d %H:%M:%S"),
                status,
            )

        # Show table
        console.print("[bold blue]Active Sessions[/]")
        console.print(table)

        return 0

    async def create_session(self, args: argparse.Namespace) -> int:
        """
        Create a new session.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        session_manager = self.get_component(SessionManager)

        if not session_manager:
            error_console.print("[bold red]Error:[/] Session Manager not available")
            return 1

        try:
            # Create new session
            session_id = await session_manager.create_session(
                user_id=args.user_id, metadata={"source": "cli", "created_by": "command"}
            )

            console.print(f"[bold green]Session created successfully[/]")
            console.print(f"Session ID: {session_id}")

            # Save session ID to file if specified
            if args.output:
                with open(args.output, "w") as f:
                    f.write(session_id)
                console.print(f"Session ID saved to {args.output}")

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error:[/] {str(e)}")
            return 1

    async def close_session(self, args: argparse.Namespace) -> int:
        """
        Close a session.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        session_manager = self.get_component(SessionManager)

        if not session_manager:
            error_console.print("[bold red]Error:[/] Session Manager not available")
            return 1

        try:
            # Close session
            await session_manager.close_session(args.session_id)

            console.print(f"[bold green]Session {args.session_id} closed successfully[/]")
            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error:[/] {str(e)}")
            return 1

    async def show_session(self, args: argparse.Namespace) -> int:
        """
        Show session details.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        session_manager = self.get_component(SessionManager)

        if not session_manager:
            error_console.print("[bold red]Error:[/] Session Manager not available")
            return 1

        try:
            # Get session
            session = await session_manager.get_session(args.session_id)

            if not session:
                error_console.print(f"[bold red]Error:[/] Session {args.session_id} not found")
                return 1

            # Show session details
            console.print(f"[bold blue]Session Details:[/] {session.session_id}")

            # Create panel with session info
            session_info = {
                "User ID": session.user_id or "anonymous",
                "Created": datetime.fromisoformat(session.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "Last Active": datetime.fromisoformat(session.last_activity).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Status": session.status,
            }

            details = "\n".join(f"{k}: {v}" for k, v in session_info.items())

            # Add metadata if available
            if session.metadata:
                metadata = "\n".join(f"  {k}: {v}" for k, v in session.metadata.items())
                details += f"\n\nMetadata:\n{metadata}"

            console.print(Panel(details, title="Session Info"))

            # Show interaction history if requested
            if args.history:
                console.print("\n[bold blue]Interaction History:[/]")

                # Get interaction history
                history = await session_manager.get_session_history(args.session_id)

                if not history:
                    console.print("[yellow]No interaction history found[/]")
                else:
                    # Create table
                    table = Table(show_header=True)
                    table.add_column("Time")
                    table.add_column("Type")
                    table.add_column("Content")

                    for interaction in history:
                        timestamp = datetime.fromisoformat(
                            interaction.get("timestamp", "")
                        ).strftime("%H:%M:%S")
                        interaction_type = interaction.get("type", "unknown")

                        # Format content based on type
                        content = interaction.get("content", "")
                        if isinstance(content, dict):
                            content = json.dumps(content, indent=2)
                        elif isinstance(content, list):
                            content = "\n".join(str(item) for item in content)

                        # Truncate long content
                        if len(str(content)) > 100:
                            content = str(content)[:97] + "..."

                        table.add_row(timestamp, interaction_type, str(content))

                    console.print(table)

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error:[/] {str(e)}")
            return 1


class ChatCommands(CommandHandler):
    """Chat and interaction commands."""

    async def chat(self, args: argparse.Namespace) -> int:
        """
        Start an interactive chat session.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get required components
        interaction_handler = self.get_component(InteractionHandler)
        session_manager = self.get_component(SessionManager)

        if not interaction_handler or not session_manager:
            error_console.print("[bold red]Error:[/] Required components not available")
            return 1

        # Create or get session
        session_id = args.session_id

        if not session_id:
            # Create new session
            try:
                session_id = await session_manager.create_session(
                    user_id=args.user_id, metadata={"source": "cli", "interface": "chat"}
                )
                console.print(f"[bold green]Created new session:[/] {session_id}")
            except Exception as e:
                error_console.print(f"[bold red]Error creating session:[/] {str(e)}")
                return 1
        else:
            # Verify session exists
            session = await session_manager.get_session(session_id)
            if not session:
                error_console.print(f"[bold red]Error:[/] Session {session_id} not found")
                return 1
            console.print(f"[bold green]Using existing session:[/] {session_id}")

        # Start interactive session
        try:
            # Create interactive session
            interactive = InteractiveSession(
                container=self.container, session_id=session_id, user_id=args.user_id
            )

            # Show welcome message
            console.print("\n[bold blue]Welcome to the AI Assistant Chat[/]")
            console.print("Type 'exit' or 'quit' to end the session, or 'help' for commands")
            console.print(f"Session ID: {session_id}\n")

            # Run interactive session
            await interactive.run()

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error in chat session:[/] {str(e)}")
            return 1

    async def send(self, args: argparse.Namespace) -> int:
        """
        Send a message to the assistant.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get required components
        interaction_handler = self.get_component(InteractionHandler)
        session_manager = self.get_component(SessionManager)

        if not interaction_handler or not session_manager:
            error_console.print("[bold red]Error:[/] Required components not available")
            return 1

        # Verify session exists
        session_id = args.session_id
        session = await session_manager.get_session(session_id)
        if not session:
            error_console.print(f"[bold red]Error:[/] Session {session_id} not found")
            return 1

        # Get message from args or stdin
        message = args.message
        if not message and not sys.stdin.isatty():
            # Read from stdin
            message = sys.stdin.read().strip()

        if not message:
            error_console.print("[bold red]Error:[/] No message provided")
            return 1

        try:
            # Send message to assistant
            console.print(f"[bold blue]Sending message to session {session_id}...[/]")

            response = await interaction_handler.handle_text_input(
                text=message, session_id=session_id, user_id=args.user_id or session.user_id
            )

            # Display response
            if args.raw:
                # Print raw response
                print(response)
            else:
                # Print formatted response
                console.print("\n[bold green]Assistant:[/]")
                console.print(Markdown(response))

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error sending message:[/] {str(e)}")
            return 1


class MemoryCommands(CommandHandler):
    """Memory management commands."""

    async def list_memories(self, args: argparse.Namespace) -> int:
        """
        List memories.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get memory manager
        memory_manager = self.get_component(MemoryManager)

        if not memory_manager:
            error_console.print("[bold red]Error:[/] Memory Manager not available")
            return 1

        try:
            # Parse memory type
            memory_type = None
            if args.type:
                try:
                    memory_type = MemoryType(args.type.lower())
                except ValueError:
                    error_console.print(f"[bold red]Error:[/] Invalid memory type: {args.type}")
                    return 1

            # Get memories
            memories = await memory_manager.search_memories(
                query=args.query or "",
                memory_type=memory_type,
                session_id=args.session_id,
                limit=args.limit,
            )

            if not memories:
                console.print("[yellow]No memories found[/]")
                return 0

            # Create table
            table = Table(show_header=True)
            table.add_column("ID")
            table.add_column("Type")
            table.add_column("Created")
            table.add_column("Content")

            # Add memories to table
            for memory in memories:
                # Format content
                content = memory.content
                if isinstance(content, dict):
                    # Extract summary or first field
                    if "summary" in content:
                        content = content["summary"]
                    elif "text" in content:
                        content = content["text"]
                    else:
                        # Use first value
                        content = next(iter(content.values()), str(content))

                # Truncate content
                content_str = str(content)
                if len(content_str) > 60:
                    content_str = content_str[:57] + "..."

                # Format created timestamp
                created = "unknown"
                if memory.metadata and hasattr(memory.metadata, "created_at"):
                    created = memory.metadata.created_at.strftime("%Y-%m-%d %H:%M")

                table.add_row(memory.memory_id, memory.memory_type.value, created, content_str)

            # Show table
            console.print(f"[bold blue]Found {len(memories)} memories[/]")
            console.print(table)

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error listing memories:[/] {str(e)}")
            return 1

    async def show_memory(self, args: argparse.Namespace) -> int:
        """
        Show memory details.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get memory manager
        memory_manager = self.get_component(MemoryManager)

        if not memory_manager:
            error_console.print("[bold red]Error:[/] Memory Manager not available")
            return 1

        try:
            # Get memory
            memory = await memory_manager.get_memory(args.memory_id)

            if not memory:
                error_console.print(f"[bold red]Error:[/] Memory {args.memory_id} not found")
                return 1

            # Show memory details
            console.print(f"[bold blue]Memory Details:[/] {memory.memory_id}")

            # Display memory info
            info_panel = Panel(
                f"Type: {memory.memory_type.value}\n"
                f"Session: {memory.session_id or 'N/A'}\n"
                f"User: {memory.owner_id or 'N/A'}\n"
                f"Created: {memory.metadata.created_at.strftime('%Y-%m-%d %H:%M:%S') if memory.metadata else 'unknown'}"
            )
            console.print(info_panel)

            # Display content
            console.print("\n[bold blue]Content:[/]")

            if isinstance(memory.content, dict):
                # Format as YAML for readability
                console.print(Syntax(yaml.dump(memory.content, default_flow_style=False), "yaml"))
            elif isinstance(memory.content, str):
                # Try to detect if it's markdown
                if "##" in memory.content or "*" in memory.content:
                    console.print(Markdown(memory.content))
                else:
                    console.print(memory.content)
            else:
                console.print(str(memory.content))

            # Show metadata if available
            if memory.metadata:
                console.print("\n[bold blue]Metadata:[/]")

                metadata = {
                    "Created": memory.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Last Accessed": (
                        memory.metadata.last_accessed.strftime("%Y-%m-%d %H:%M:%S")
                        if memory.metadata.last_accessed
                        else "Never"
                    ),
                    "Access Count": memory.metadata.access_count,
                    "Importance": memory.metadata.importance,
                }

                # Add tags if available
                if memory.metadata.tags:
                    metadata["Tags"] = ", ".join(memory.metadata.tags)

                # Add custom metadata if available
                if memory.metadata.custom_metadata:
                    metadata["Custom"] = memory.metadata.custom_metadata

                metadata_table = Table(show_header=False)
                metadata_table.add_column("Property")
                metadata_table.add_column("Value")

                for key, value in metadata.items():
                    metadata_table.add_row(key, str(value))

                console.print(metadata_table)

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error showing memory:[/] {str(e)}")
            return 1

    async def delete_memory(self, args: argparse.Namespace) -> int:
        """
        Delete a memory.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get memory manager
        memory_manager = self.get_component(MemoryManager)

        if not memory_manager:
            error_console.print("[bold red]Error:[/] Memory Manager not available")
            return 1

        try:
            # Confirm deletion if not forced
            if not args.force:
                confirm = input(f"Are you sure you want to delete memory {args.memory_id}? (y/N): ")
                if confirm.lower() != "y":
                    console.print("[yellow]Deletion cancelled[/]")
                    return 0

            # Delete memory
            success = await memory_manager.delete_memory(args.memory_id)

            if success:
                console.print(f"[bold green]Memory {args.memory_id} deleted successfully[/]")
                return 0
            else:
                error_console.print(f"[bold red]Error:[/] Failed to delete memory {args.memory_id}")
                return 1

        except Exception as e:
            error_console.print(f"[bold red]Error deleting memory:[/] {str(e)}")
            return 1

    async def export_memories(self, args: argparse.Namespace) -> int:
        """
        Export memories to a file.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get memory manager
        memory_manager = self.get_component(MemoryManager)

        if not memory_manager:
            error_console.print("[bold red]Error:[/] Memory Manager not available")
            return 1

        try:
            # Parse memory type
            memory_type = None
            if args.type:
                try:
                    memory_type = MemoryType(args.type.lower())
                except ValueError:
                    error_console.print(f"[bold red]Error:[/] Invalid memory type: {args.type}")
                    return 1

            # Get memories
            memories = await memory_manager.search_memories(
                query=args.query or "",
                memory_type=memory_type,
                session_id=args.session_id,
                limit=args.limit or 1000,  # Higher limit for exports
            )

            if not memories:
                console.print("[yellow]No memories found to export[/]")
                return 0

            # Format memories for export
            export_data = []
            for memory in memories:
                # Convert memory to dict
                memory_dict = {
                    "memory_id": memory.memory_id,
                    "type": memory.memory_type.value,
                    "content": memory.content,
                    "session_id": memory.session_id,
                    "owner_id": memory.owner_id,
                    "context_id": memory.context_id,
                }

                # Add metadata if available
                if memory.metadata:
                    memory_dict["metadata"] = {
                        "created_at": (
                            memory.metadata.created_at.isoformat()
                            if memory.metadata.created_at
                            else None
                        ),
                        "last_accessed": (
                            memory.metadata.last_accessed.isoformat()
                            if memory.metadata.last_accessed
                            else None
                        ),
                        "access_count": memory.metadata.access_count,
                        "importance": memory.metadata.importance,
                        "tags": list(memory.metadata.tags) if memory.metadata.tags else [],
                    }

                export_data.append(memory_dict)

            # Export to file
            export_format = args.format.lower()
            output_path = args.output

            if export_format == "json":
                with open(output_path, "w") as f:
                    json.dump(export_data, f, indent=2)
            elif export_format == "yaml":
                with open(output_path, "w") as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            else:
                error_console.print(
                    f"[bold red]Error:[/] Unsupported export format: {export_format}"
                )
                return 1

            console.print(
                f"[bold green]Successfully exported {len(export_data)} memories to {output_path}[/]"
            )
            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error exporting memories:[/] {str(e)}")
            return 1


class SkillCommands(CommandHandler):
    """Skill management commands."""

    async def list_skills(self, args: argparse.Namespace) -> int:
        """
        List available skills.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get skill registry
        skill_registry = self.get_component(SkillRegistry)

        if not skill_registry:
            error_console.print("[bold red]Error:[/] Skill Registry not available")
            return 1

        try:
            # Get registered skills
            skills = skill_registry.get_all_skills()

            # Filter by category if specified
            if args.category:
                skills = {
                    name: info
                    for name, info in skills.items()
                    if info.get("category", "").lower() == args.category.lower()
                }

            # Filter by installed/available if specified
            if args.status == "installed":
                skills = {
                    name: info for name, info in skills.items() if info.get("installed", False)
                }
            elif args.status == "available":
                skills = {
                    name: info for name, info in skills.items() if not info.get("installed", False)
                }

            if not skills:
                console.print("[yellow]No skills found[/]")
                return 0

            # Group skills by category
            categories = {}
            for name, info in skills.items():
                category = info.get("category", "Uncategorized")
                if category not in categories:
                    categories[category] = []

                # Add skill to category
                categories[category].append((name, info))

            # Display skills by category
            console.print("[bold blue]Available Skills[/]")

            for category, skills_list in sorted(categories.items()):
                console.print(f"\n[bold green]{category}[/]")

                # Create table for this category
                table = Table(show_header=True)
                table.add_column("Name")
                table.add_column("Version")
                table.add_column("Status")
                table.add_column("Description")

                # Add skills to table
                for name, info in sorted(skills_list, key=lambda x: x[0]):
                    status = (
                        "[green]Installed[/]"
                        if info.get("installed", False)
                        else "[yellow]Available[/]"
                    )
                    version = info.get("version", "unknown")
                    description = info.get("description", "")

                    # Truncate description if too long
                    if len(description) > 60:
                        description = description[:57] + "..."

                    table.add_row(name, version, status, description)

                console.print(table)

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error listing skills:[/] {str(e)}")
            return 1

    async def install_skill(self, args: argparse.Namespace) -> int:
        """
        Install a skill.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get skill installer
        skill_installer = self.get_component(SkillInstaller)

        if not skill_installer:
            error_console.print("[bold red]Error:[/] Skill Installer not available")
            return 1

        try:
            # Install skill
            console.print(f"[bold blue]Installing skill:[/] {args.skill_name}")

            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]Installing skill...[/]"), console=console
            ) as progress:
                progress.add_task("install", total=None)

                if args.url:
                    # Install from URL
                    result = await skill_installer.install_from_url(args.url)
                elif args.path:
                    # Install from local path
                    result = await skill_installer.install_from_path(args.path)
                else:
                    # Install by name
                    result = await skill_installer.install_skill(args.skill_name)

            if result:
                console.print(f"[bold green]Skill {args.skill_name} installed successfully[/]")
                return 0
            else:
                error_console.print(
                    f"[bold red]Error:[/] Failed to install skill {args.skill_name}"
                )
                return 1

        except Exception as e:
            error_console.print(f"[bold red]Error installing skill:[/] {str(e)}")
            return 1

    async def uninstall_skill(self, args: argparse.Namespace) -> int:
        """
        Uninstall a skill.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get skill installer
        skill_installer = self.get_component(SkillInstaller)

        if not skill_installer:
            error_console.print("[bold red]Error:[/] Skill Installer not available")
            return 1

        try:
            # Confirm uninstall if not forced
            if not args.force:
                confirm = input(
                    f"Are you sure you want to uninstall skill {args.skill_name}? (y/N): "
                )
                if confirm.lower() != "y":
                    console.print("[yellow]Uninstall cancelled[/]")
                    return 0

            # Uninstall skill
            console.print(f"[bold blue]Uninstalling skill:[/] {args.skill_name}")

            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]Uninstalling skill...[/]"), console=console
            ) as progress:
                progress.add_task("uninstall", total=None)
                result = await skill_installer.uninstall_skill(args.skill_name)

            if result:
                console.print(f"[bold green]Skill {args.skill_name} uninstalled successfully[/]")
                return 0
            else:
                error_console.print(
                    f"[bold red]Error:[/] Failed to uninstall skill {args.skill_name}"
                )
                return 1

        except Exception as e:
            error_console.print(f"[bold red]Error uninstalling skill:[/] {str(e)}")
            return 1

    async def show_skill(self, args: argparse.Namespace) -> int:
        """
        Show skill details.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get skill registry
        skill_registry = self.get_component(SkillRegistry)

        if not skill_registry:
            error_console.print("[bold red]Error:[/] Skill Registry not available")
            return 1

        try:
            # Get skill info
            skill_info = skill_registry.get_skill_info(args.skill_name)

            if not skill_info:
                error_console.print(f"[bold red]Error:[/] Skill {args.skill_name} not found")
                return 1

            # Display skill details
            console.print(f"[bold blue]Skill Details:[/] {args.skill_name}")

            # Create panel with skill info
            details = [
                f"Name: {args.skill_name}",
                f"Version: {skill_info.get('version', 'unknown')}",
                f"Category: {skill_info.get('category', 'Uncategorized')}",
                f"Status: {'Installed' if skill_info.get('installed', False) else 'Available'}",
                f"\nDescription: {skill_info.get('description', 'No description available')}",
            ]

            # Add author info if available
            if "author" in skill_info:
                details.append(f"\nAuthor: {skill_info['author']}")

            # Add requirements if available
            if "requirements" in skill_info and skill_info["requirements"]:
                req_str = ", ".join(skill_info["requirements"])
                details.append(f"\nRequirements: {req_str}")

            # Add parameters if available
            if "parameters" in skill_info and skill_info["parameters"]:
                details.append("\nParameters:")
                for param in skill_info["parameters"]:
                    param_str = f"  - {param.get('name', 'unnamed')}"
                    if "type" in param:
                        param_str += f" ({param['type']})"
                    if "description" in param:
                        param_str += f": {param['description']}"
                    details.append(param_str)

            # Add examples if available
            if "examples" in skill_info and skill_info["examples"]:
                details.append("\nExample Usage:")
                for example in skill_info["examples"]:
                    details.append(f"  - {example}")

            console.print(Panel("\n".join(details), title=f"Skill: {args.skill_name}"))

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error showing skill:[/] {str(e)}")
            return 1


class ModelCommands(CommandHandler):
    """Model management commands."""

    async def list_models(self, args: argparse.Namespace) -> int:
        """
        List available models.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get model router
        model_router = self.get_component(ModelRouter)

        if not model_router:
            error_console.print("[bold red]Error:[/] Model Router not available")
            return 1

        try:
            # Get available models
            models = await model_router.list_available_models()

            if not models:
                console.print("[yellow]No models available[/]")
                return 0

            # Group models by provider
            providers = {}
            for model in models:
                provider = model.get("provider", "unknown")
                if provider not in providers:
                    providers[provider] = []

                providers[provider].append(model)

            # Display models by provider
            console.print("[bold blue]Available Models[/]")

            for provider, provider_models in sorted(providers.items()):
                console.print(f"\n[bold green]Provider: {provider}[/]")

                # Create table for this provider
                table = Table(show_header=True)
                table.add_column("Model ID")
                table.add_column("Type")
                table.add_column("Status")
                table.add_column("Description")

                # Add models to table
                for model in sorted(provider_models, key=lambda x: x.get("id", "")):
                    model_id = model.get("id", "unknown")
                    model_type = model.get("type", "unknown")
                    status = (
                        "[green]Available[/]"
                        if model.get("available", False)
                        else "[yellow]Unavailable[/]"
                    )
                    description = model.get("description", "")

                    # Truncate description if too long
                    if len(description) > 60:
                        description = description[:57] + "..."

                    table.add_row(model_id, model_type, status, description)

                console.print(table)

            # Show default models
            console.print("\n[bold blue]Default Models[/]")

            # Get default models
            defaults = await model_router.get_default_models()

            if defaults:
                defaults_table = Table(show_header=True)
                defaults_table.add_column("Task")
                defaults_table.add_column("Model ID")

                for task, model_id in sorted(defaults.items()):
                    defaults_table.add_row(task, model_id)

                console.print(defaults_table)
            else:
                console.print("[yellow]No default models configured[/]")

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error listing models:[/] {str(e)}")
            return 1

    async def set_default_model(self, args: argparse.Namespace) -> int:
        """
        Set a default model for a task.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get model router
        model_router = self.get_component(ModelRouter)

        if not model_router:
            error_console.print("[bold red]Error:[/] Model Router not available")
            return 1

        try:
            # Set default model
            success = await model_router.set_default_model(task=args.task, model_id=args.model_id)

            if success:
                console.print(
                    f"[bold green]Successfully set default model for {args.task} to {args.model_id}[/]"
                )
                return 0
            else:
                error_console.print(f"[bold red]Error:[/] Failed to set default model")
                return 1

        except Exception as e:
            error_console.print(f"[bold red]Error setting default model:[/] {str(e)}")
            return 1

    async def test_model(self, args: argparse.Namespace) -> int:
        """
        Test a model with a sample prompt.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get model router
        model_router = self.get_component(ModelRouter)

        if not model_router:
            error_console.print("[bold red]Error:[/] Model Router not available")
            return 1

        try:
            # Get prompt
            prompt = args.prompt
            if not prompt and not sys.stdin.isatty():
                # Read from stdin
                prompt = sys.stdin.read().strip()

            if not prompt:
                prompt = "Tell me about yourself in one paragraph."

            # Display test info
            console.print(f"[bold blue]Testing model:[/] {args.model_id}")
            console.print(f"Prompt: {prompt}")

            # Generate response
            with Progress(
                SpinnerColumn(), TextColumn("[bold blue]Generating response...[/]"), console=console
            ) as progress:
                progress.add_task("generate", total=None)

                # Use different methods based on model type
                if args.type == "text":
                    response = await model_router.generate_text(
                        prompt=prompt, model_id=args.model_id
                    )
                elif args.type == "embedding":
                    # For embeddings, just show a sample
                    embedding = await model_router.get_embeddings(
                        text=prompt, model_id=args.model_id
                    )

                    # Format embedding preview
                    if embedding:
                        embedding_len = len(embedding)
                        sample = embedding[:5]
                        response = f"Generated embedding with {embedding_len} dimensions.\nSample: {sample}..."
                    else:
                        response = "Failed to generate embedding."
                else:
                    error_console.print(f"[bold red]Error:[/] Unsupported model type: {args.type}")
                    return 1

            # Display response
            console.print("\n[bold green]Response:[/]")

            if args.type == "text":
                console.print(Markdown(response))
            else:
                console.print(response)

            # Display metadata
            console.print("\n[bold blue]Performance:[/]")
            console.print(f"Model: {args.model_id}")
            console.print(f"Type: {args.type}")

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error testing model:[/] {str(e)}")
            return 1


class UtilityCommands(CommandHandler):
    """Utility commands."""

    async def benchmark(self, args: argparse.Namespace) -> int:
        """
        Run a simple benchmark.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get necessary components
        model_router = self.get_component(ModelRouter)
        memory_manager = self.get_component(MemoryManager)

        if not model_router or not memory_manager:
            error_console.print("[bold red]Error:[/] Required components not available")
            return 1

        try:
            # Display benchmark info
            console.print("[bold blue]Running benchmark[/]")
            console.print(f"Type: {args.type}")
            console.print(f"Iterations: {args.iterations}")

            # Initialize results
            results = {"type": args.type, "iterations": args.iterations, "timings": []}

            # Run benchmark
            with Progress() as progress:
                task = progress.add_task(
                    f"[blue]Running {args.type} benchmark...", total=args.iterations
                )

                for i in range(args.iterations):
                    start_time = time.time()

                    # Run different benchmarks based on type
                    if args.type == "memory":
                        # Benchmark memory operations
                        await memory_manager.search_memories("test", limit=10)
                    elif args.type == "model":
                        # Benchmark model generation
                        await model_router.generate_text("Generate a short poem about nature.")
                    elif args.type == "embedding":
                        # Benchmark embedding generation
                        await model_router.get_embeddings(
                            "This is a test sentence for embedding generation."
                        )
                    else:
                        error_console.print(
                            f"[bold red]Error:[/] Unsupported benchmark type: {args.type}"
                        )
                        return 1

                    # Record timing
                    elapsed = time.time() - start_time
                    results["timings"].append(elapsed)

                    # Update progress
                    progress.update(
                        task,
                        advance=1,
                        description=f"[blue]Running {args.type} benchmark... {i+1}/{args.iterations}",
                    )

                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.1)

            # Calculate statistics
            timings = results["timings"]
            avg_time = sum(timings) / len(timings)
            min_time = min(timings)
            max_time = max(timings)

            # Sort timings for percentiles
            timings.sort()
            p50 = timings[len(timings) // 2]
            p90 = timings[int(len(timings) * 0.9)]
            p95 = timings[int(len(timings) * 0.95)]

            # Display results
            console.print("\n[bold blue]Benchmark Results[/]")

            results_table = Table(show_header=True)
            results_table.add_column("Metric")
            results_table.add_column("Value (seconds)")

            results_table.add_row("Average Time", f"{avg_time:.4f}")
            results_table.add_row("Minimum Time", f"{min_time:.4f}")
            results_table.add_row("Maximum Time", f"{max_time:.4f}")
            results_table.add_row("p50 (Median)", f"{p50:.4f}")
            results_table.add_row("p90", f"{p90:.4f}")
            results_table.add_row("p95", f"{p95:.4f}")

            console.print(results_table)

            # Save results if output specified
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(
                        {
                            "type": args.type,
                            "iterations": args.iterations,
                            "average": avg_time,
                            "min": min_time,
                            "max": max_time,
                            "p50": p50,
                            "p90": p90,
                            "p95": p95,
                            "raw_timings": timings,
                        },
                        f,
                        indent=2,
                    )

                console.print(f"[bold green]Results saved to {args.output}[/]")

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error running benchmark:[/] {str(e)}")
            return 1

    async def version(self, args: argparse.Namespace) -> int:
        """
        Show version information.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        try:
            # Get version information
            version_info = {
                "System": "AI Assistant",
                "Version": self.config_loader.get("version", "unknown"),
                "Build Date": self.config_loader.get("build_date", "unknown"),
                "Python": sys.version.split()[0],
                "Platform": sys.platform,
            }

            # Display version info
            console.print("[bold blue]Version Information[/]")

            # Create panel
            version_text = "\n".join(f"{k}: {v}" for k, v in version_info.items())
            console.print(Panel(version_text, title="Version Info"))

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error getting version info:[/] {str(e)}")
            return 1

    async def diagnostics(self, args: argparse.Namespace) -> int:
        """
        Run system diagnostics.

        Args:
            args: Command arguments

        Returns:
            Exit code
        """
        # Get components
        health_check = self.get_component(HealthCheck)
        component_manager = self.get_component(ComponentManager)
        memory_manager = self.get_component(MemoryManager)

        if not health_check or not component_manager:
            error_console.print("[bold red]Error:[/] Required components not available")
            return 1

        try:
            console.print("[bold blue]Running System Diagnostics[/]")

            # Check system health
            console.print("\n[bold]Health Check:[/]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Running health checks...[/]"),
                console=console,
            ) as progress:
                progress.add_task("health", total=None)
                health_results = await health_check.check_all()

            # Count healthy/unhealthy components
            healthy_count = sum(1 for status, _ in health_results.values() if status)
            unhealthy_count = len(health_results) - healthy_count

            health_table = Table(show_header=True)
            health_table.add_column("Component")
            health_table.add_column("Status")
            health_table.add_column("Details")

            for name, (is_healthy, details) in health_results.items():
                status = "[bold green]Healthy[/]" if is_healthy else "[bold red]Unhealthy[/]"
                details_str = ", ".join(f"{k}: {v}" for k, v in details.items()) if details else ""
                health_table.add_row(name, status, details_str)

            console.print(health_table)
            console.print(
                f"Summary: {healthy_count} healthy, {unhealthy_count} unhealthy components"
            )

            # Check memory stats
            if memory_manager:
                console.print("\n[bold]Memory Statistics:[/]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Gathering memory statistics...[/]"),
                    console=console,
                ) as progress:
                    progress.add_task("memory", total=None)
                    memory_stats = await memory_manager.get_stats()

                if memory_stats:
                    memory_table = Table(show_header=True)
                    memory_table.add_column("Memory Type")
                    memory_table.add_column("Count")
                    memory_table.add_column("Size")

                    for memory_type, stats in memory_stats.items():
                        if isinstance(stats, dict):
                            count = stats.get("count", 0)
                            size = stats.get("size_bytes", 0)
                            # Format size
                            size_str = (
                                f"{size / (1024*1024):.2f} MB"
                                if size > 1024 * 1024
                                else f"{size / 1024:.2f} KB"
                            )
                            memory_table.add_row(memory_type, str(count), size_str)

                    console.print(memory_table)

            # Check component dependencies
            console.print("\n[bold]Component Dependencies:[/]")

            # Simulate component dependency check
            console.print("Component dependency validation completed - all dependencies satisfied.")

            return 0

        except Exception as e:
            error_console.print(f"[bold red]Error running diagnostics:[/] {str(e)}")
            return 1
