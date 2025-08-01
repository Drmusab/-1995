"""
Advanced AI Assistant Command Line Interface
Author: Drmusab
Last Modified: 2025-07-05 11:16:08 UTC

This module provides a comprehensive command-line interface for the AI assistant,
supporting both interactive conversation mode and command execution mode, with
access to all core system functionality including workflows, plugins, and diagnostics.
"""

import os
import sys
import asyncio
import argparse
import json
import readline
import signal
import atexit
import time
import logging
import tempfile
import subprocess
import uuid
import shutil
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set, Union, Callable
from datetime import datetime, timezone
from enum import Enum
import inspect
import traceback
# Lazy import for numpy - only import when needed
# import numpy as np  # Moved to lazy import
from concurrent.futures import ThreadPoolExecutor

# Lazy import for heavy dependencies - only import when needed
_numpy = None
_rich_modules = None

def get_numpy():
    """Lazy import numpy only when needed."""
    global _numpy
    if _numpy is None:
        import numpy as np
        _numpy = np
    return _numpy

def get_rich_modules():
    """Lazy import rich modules only when needed."""
    global _rich_modules
    if _rich_modules is None:
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.syntax import Syntax
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn
            from rich.table import Table
            from rich.prompt import Prompt, Confirm
            from rich import print as rich_print
            _rich_modules = {
                'Console': Console,
                'Markdown': Markdown,
                'Syntax': Syntax,
                'Panel': Panel,
                'Progress': Progress,
                'SpinnerColumn': SpinnerColumn,
                'TextColumn': TextColumn,
                'Table': Table,
                'Prompt': Prompt,
                'Confirm': Confirm,
                'rich_print': rich_print,
                'available': True
            }
        except ImportError:
            _rich_modules = {'available': False}
    return _rich_modules

# Check if rich is available (but don't import yet)
try:
    import importlib.util
    RICH_AVAILABLE = importlib.util.find_spec("rich") is not None
except:
    RICH_AVAILABLE = False

# Import from main application entry point
from src.main import AIAssistant

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    UserInteractionStarted, UserInteractionCompleted,
    SessionStarted, SessionEnded, WorkflowStarted, WorkflowCompleted,
    PluginLoaded, PluginUnloaded, SystemStarted, SystemShutdownStarted
)
from src.core.dependency_injection import Container

# Assistant components
from src.assistant.interaction_handler import (
    InteractionHandler, UserMessage, InteractionMode, InputModality, OutputModality
)
from src.assistant.session_manager import EnhancedSessionManager
from src.assistant.workflow_orchestrator import WorkflowOrchestrator
from src.assistant.plugin_manager import EnhancedPluginManager
from src.assistant.component_manager import EnhancedComponentManager

# Observability
from src.observability.logging.config import get_logger, setup_logging

# Import UI-specific components
from src.ui.cli.commands import CommandRegistry, Command
from src.ui.cli.interactive import InteractiveSession


class CLIMode(Enum):
    """CLI operation modes."""
    INTERACTIVE = "interactive"  # Interactive conversation mode
    COMMAND = "command"          # Single command execution mode
    REPL = "repl"                # Read-Eval-Print Loop mode
    SCRIPT = "script"            # Script execution mode
    MONITOR = "monitor"          # System monitoring mode


class CLITheme(Enum):
    """CLI color themes."""
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"
    MONOCHROME = "monochrome"


class AssistantCLI:
    """
    Advanced command-line interface for the AI Assistant.
    
    This CLI provides multiple interaction modes:
    - Interactive conversation with the assistant
    - Command mode for executing specific functions
    - REPL mode for programmatic interaction
    - Script mode for running commands from files
    - Monitor mode for real-time system monitoring
    
    Features:
    - Rich text formatting and colored output
    - Command history and auto-completion
    - Integrated help system
    - File upload/download capabilities
    - Plugin management
    - Workflow execution
    - System diagnostics and monitoring
    - Configuration management
    """
    
    def __init__(self, assistant: Optional[AIAssistant] = None):
        """
        Initialize the CLI with an optional existing assistant instance.
        
        Args:
            assistant: Optional existing AIAssistant instance
        """
        # Setup logging
        setup_logging()
        self.logger = get_logger(__name__)
        
        # Initialize rich console if available
        self.console = Console() if RICH_AVAILABLE else None
        
        # Store or create assistant instance
        self.assistant = assistant
        self._assistant_initialized = assistant is not None
        
        # CLI state
        self.mode = CLIMode.INTERACTIVE
        self.current_session_id = None
        self.current_user_id = None
        self.current_conversation_id = None
        self.exit_requested = False
        self.command_history_file = os.path.expanduser("~/.ai_assistant_history")
        
        # Command registry
        self.command_registry = CommandRegistry()
        self._register_commands()
        
        # Settings
        self.settings = {
            "theme": CLITheme.DARK,
            "max_output_length": 2000,
            "show_timestamps": True,
            "save_history": True,
            "verbose": False,
            "auto_suggestions": True,
            "use_markdown": True,
            "upload_dir": tempfile.gettempdir(),
            "download_dir": os.path.expanduser("~/Downloads"),
            "max_history": 1000,
            "prompt_style": ">>> ",
            "user_color": "cyan",
            "assistant_color": "green",
            "system_color": "yellow",
            "error_color": "red"
        }
        
        # Setup CLI environment
        self._setup_readline()
        self._setup_signal_handlers()
    
    def _setup_readline(self):
        """Setup readline for command history and tab completion."""
        # Set up command history
        if self.settings["save_history"] and os.path.exists(self.command_history_file):
            try:
                readline.read_history_file(self.command_history_file)
                readline.set_history_length(self.settings["max_history"])
            except Exception as e:
                self.logger.warning(f"Could not read history file: {str(e)}")
        
        # Save history on exit
        if self.settings["save_history"]:
            atexit.register(lambda: readline.write_history_file(self.command_history_file))
        
        # Set up tab completion
        readline.parse_and_bind("tab: complete")
        readline.set_completer(self._command_completer)
    
    def _command_completer(self, text, state):
        """Provide tab completion for commands."""
        # Get all command names
        command_names = list(self.command_registry.get_commands().keys())
        
        # Filter commands that start with the text
        matches = [cmd for cmd in command_names if cmd.startswith(text)]
        
        # Return the state-th match, or None if no match
        return matches[state] if state < len(matches) else None
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful exit."""
        signal.signal(signal.SIGINT, self._handle_sigint)
        if platform.system() != "Windows":
            signal.signal(signal.SIGTERM, self._handle_sigterm)
    
    def _handle_sigint(self, signum, frame):
        """Handle keyboard interrupt (Ctrl+C)."""
        print("\nInterrupted. Use 'exit' or 'quit' to exit the program.")
    
    def _handle_sigterm(self, signum, frame):
        """Handle termination signal."""
        print("\nReceived termination signal. Shutting down...")
        self.exit_requested = True
        asyncio.create_task(self._shutdown())
    
    def _register_commands(self):
        """Register all available CLI commands."""
        # System commands
        self.command_registry.register(Command(
            name="help",
            description="Show help information about available commands",
            callback=self._cmd_help,
            usage="help [command]"
        ))
        
        self.command_registry.register(Command(
            name="exit",
            description="Exit the application",
            callback=self._cmd_exit,
            aliases=["quit", "q"]
        ))
        
        self.command_registry.register(Command(
            name="clear",
            description="Clear the terminal screen",
            callback=self._cmd_clear,
            aliases=["cls"]
        ))
        
        self.command_registry.register(Command(
            name="mode",
            description="Change the CLI mode",
            callback=self._cmd_mode,
            usage="mode [interactive|command|repl|script|monitor]"
        ))
        
        self.command_registry.register(Command(
            name="status",
            description="Show system status information",
            callback=self._cmd_status
        ))
        
        # Session commands
        self.command_registry.register(Command(
            name="session",
            description="Manage sessions",
            callback=self._cmd_session,
            usage="session [new|list|switch|end] [id]"
        ))
        
        # Workflow commands
        self.command_registry.register(Command(
            name="workflow",
            description="Manage and execute workflows",
            callback=self._cmd_workflow,
            usage="workflow [list|execute|status] [id] [params]"
        ))
        
        # Plugin commands
        self.command_registry.register(Command(
            name="plugin",
            description="Manage plugins",
            callback=self._cmd_plugin,
            usage="plugin [list|load|unload|info] [id]"
        ))
        
        # Component commands
        self.command_registry.register(Command(
            name="component",
            description="Manage system components",
            callback=self._cmd_component,
            usage="component [list|status|restart] [id]"
        ))
        
        # Configuration commands
        self.command_registry.register(Command(
            name="config",
            description="View or modify configuration",
            callback=self._cmd_config,
            usage="config [get|set|list] [key] [value]"
        ))
        
        # Debug commands
        self.command_registry.register(Command(
            name="debug",
            description="Debug tools and diagnostics",
            callback=self._cmd_debug,
            usage="debug [log|trace|profile|memory] [options]"
        ))
        
        # File commands
        self.command_registry.register(Command(
            name="upload",
            description="Upload a file for processing",
            callback=self._cmd_upload,
            usage="upload <filepath>"
        ))
        
        self.command_registry.register(Command(
            name="download",
            description="Download generated content to a file",
            callback=self._cmd_download,
            usage="download <content_id> <filepath>"
        ))
        
        # Settings commands
        self.command_registry.register(Command(
            name="settings",
            description="View or modify CLI settings",
            callback=self._cmd_settings,
            usage="settings [get|set|list] [key] [value]"
        ))
        
        # History commands
        self.command_registry.register(Command(
            name="history",
            description="View or manage command history",
            callback=self._cmd_history,
            usage="history [list|clear|save|load] [count]"
        ))
    
    async def initialize(self):
        """Initialize the CLI and the assistant if not already initialized."""
        if not self._assistant_initialized:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Initializing AI Assistant..."),
                    console=self.console
                ) as progress:
                    progress.add_task("init", total=None)
                    self.assistant = AIAssistant()
                    await self.assistant.initialize()
            else:
                print("Initializing AI Assistant...")
                self.assistant = AIAssistant()
                await self.assistant.initialize()
            
            self._assistant_initialized = True
        
        # Create initial session
        self.current_session_id = await self.assistant.session_manager.create_session(
            user_id=self.current_user_id
        )
        
        self.print_system_message("AI Assistant initialized and ready.")
        self.print_system_message(f"Session ID: {self.current_session_id}")
        self.print_system_message("Type 'help' to see available commands.")
    
    async def run(self, args: Optional[List[str]] = None):
        """
        Run the CLI with the given arguments.
        
        Args:
            args: Command line arguments
        """
        parser = self._create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Set current user if provided
        if parsed_args.user:
            self.current_user_id = parsed_args.user
        
        # Set CLI mode
        if parsed_args.command:
            self.mode = CLIMode.COMMAND
        elif parsed_args.repl:
            self.mode = CLIMode.REPL
        elif parsed_args.script:
            self.mode = CLIMode.SCRIPT
        elif parsed_args.monitor:
            self.mode = CLIMode.MONITOR
        else:
            self.mode = CLIMode.INTERACTIVE
        
        # Initialize the assistant
        await self.initialize()
        
        # Run in the appropriate mode
        if self.mode == CLIMode.INTERACTIVE:
            await self._run_interactive_mode()
        elif self.mode == CLIMode.COMMAND:
            await self._run_command_mode(parsed_args.command)
        elif self.mode == CLIMode.REPL:
            await self._run_repl_mode()
        elif self.mode == CLIMode.SCRIPT:
            await self._run_script_mode(parsed_args.script)
        elif self.mode == CLIMode.MONITOR:
            await self._run_monitor_mode()
    
    def _create_argument_parser(self):
        """Create the argument parser for the CLI."""
        parser = argparse.ArgumentParser(description="AI Assistant Command Line Interface")
        
        # Mode selection (mutually exclusive)
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "-i", "--interactive", 
            action="store_true", 
            help="Run in interactive conversation mode (default)"
        )
        mode_group.add_argument(
            "-c", "--command", 
            type=str, 
            help="Execute a single command and exit"
        )
        mode_group.add_argument(
            "-r", "--repl", 
            action="store_true", 
            help="Run in REPL (Read-Eval-Print Loop) mode"
        )
        mode_group.add_argument(
            "-s", "--script", 
            type=str, 
            help="Execute commands from a script file"
        )
        mode_group.add_argument(
            "-m", "--monitor", 
            action="store_true", 
            help="Run in system monitoring mode"
        )
        
        # Other options
        parser.add_argument(
            "-u", "--user", 
            type=str, 
            help="User ID for authentication"
        )
        parser.add_argument(
            "-v", "--verbose", 
            action="store_true", 
            help="Enable verbose output"
        )
        parser.add_argument(
            "--no-color", 
            action="store_true", 
            help="Disable colored output"
        )
        parser.add_argument(
            "--config", 
            type=str, 
            help="Path to configuration file"
        )
        parser.add_argument(
            "--debug", 
            action="store_true", 
            help="Enable debug mode"
        )
        parser.add_argument(
            "--version", 
            action="store_true", 
            help="Show version information and exit"
        )
        
        return parser
    
    async def _run_interactive_mode(self):
        """Run the CLI in interactive conversation mode."""
        self.print_system_message("Starting interactive conversation mode.")
        self.print_system_message("Type a message to chat with the assistant, or start with '/' for commands.")
        self.print_system_message("Type '/exit' or '/quit' to exit.")
        
        interactive_session = InteractiveSession(
            assistant=self.assistant,
            session_id=self.current_session_id,
            user_id=self.current_user_id,
            command_registry=self.command_registry,
            console=self.console if RICH_AVAILABLE else None
        )
        
        await interactive_session.run()
    
    async def _run_command_mode(self, command_str: str):
        """
        Run the CLI in command mode, executing a single command.
        
        Args:
            command_str: Command string to execute
        """
        if not command_str:
            self.print_error("No command specified")
            return
        
        # Parse command
        parts = command_str.split(" ", 1)
        command_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        # Execute command
        await self._execute_command(command_name, args)
    
    async def _run_repl_mode(self):
        """Run the CLI in REPL (Read-Eval-Print Loop) mode."""
        self.print_system_message("Starting REPL mode.")
        self.print_system_message("Enter commands to execute, or 'exit' to quit.")
        
        while not self.exit_requested:
            try:
                # Get command input
                if RICH_AVAILABLE:
                    command_str = Prompt.ask(self.settings["prompt_style"])
                else:
                    command_str = input(self.settings["prompt_style"])
                
                # Exit check
                if command_str.lower() in ["exit", "quit", "q"]:
                    break
                
                # Skip empty lines
                if not command_str.strip():
                    continue
                
                # Parse command
                parts = command_str.split(" ", 1)
                command_name = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                
                # Execute command
                await self._execute_command(command_name, args)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to exit")
            except Exception as e:
                self.print_error(f"Error: {str(e)}")
                if self.settings["verbose"]:
                    traceback.print_exc()
    
    async def _run_script_mode(self, script_path: str):
        """
        Run the CLI in script mode, executing commands from a file.
        
        Args:
            script_path: Path to the script file
        """
        if not os.path.exists(script_path):
            self.print_error(f"Script file not found: {script_path}")
            return
        
        self.print_system_message(f"Executing script: {script_path}")
        
        try:
            with open(script_path, 'r') as f:
                script_lines = f.readlines()
            
            for line_num, line in enumerate(script_lines, 1):
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse command
                parts = line.split(" ", 1)
                command_name = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                
                # Print the command being executed
                self.print_system_message(f"Line {line_num}: {line}")
                
                # Execute command
                await self._execute_command(command_name, args)
                
                # Add a small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.print_error(f"Error executing script: {str(e)}")
            if self.settings["verbose"]:
                traceback.print_exc()
    
    async def _run_monitor_mode(self):
        """Run the CLI in system monitoring mode."""
        self.print_system_message("Starting system monitoring mode.")
        self.print_system_message("Press Ctrl+C to exit.")
        
        update_interval = 2.0  # seconds
        
        try:
            while not self.exit_requested:
                # Clear the screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Get system status
                status = self.assistant.get_status()
                
                # Display system status
                if RICH_AVAILABLE:
                    table = Table(title="AI Assistant System Monitor")
                    table.add_column("Component", style="cyan")
                    table.add_column("Status", style="green")
                    table.add_column("Details", style="yellow")
                    
                    # System status
                    table.add_row(
                        "System", 
                        status.get("status", "unknown"),
                        f"Uptime: {status.get('uptime_seconds', 0):.1f}s"
                    )
                    
                    # Component statuses
                    components = status.get("components", {})
                    for component_name, component_status in components.items():
                        if component_status is None:
                            continue
                        
                        details = ""
                        if component_name == "core_engine" and component_status:
                            details = f"State: {component_status.get('state', 'unknown')}"
                        elif component_name == "session_manager" and component_status:
                            details = f"Active sessions: {component_status.get('total_active_sessions', 0)}"
                        elif component_name == "workflow_orchestrator" and component_status:
                            details = f"Active workflows: {component_status.get('active_executions', 0)}"
                        
                        table.add_row(component_name, "active", details)
                    
                    self.console.print(table)
                else:
                    print(f"=== AI Assistant System Monitor ===")
                    print(f"System status: {status.get('status', 'unknown')}")
                    print(f"Uptime: {status.get('uptime_seconds', 0):.1f}s")
                    print()
                    print("Components:")
                    
                    components = status.get("components", {})
                    for component_name, component_status in components.items():
                        if component_status is None:
                            continue
                        print(f"- {component_name}: active")
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nExiting monitoring mode")
    
    async def _execute_command(self, command_name: str, args_str: str) -> bool:
        """
        Execute a command with the given arguments.
        
        Args:
            command_name: Name of the command to execute
            args_str: Arguments string for the command
            
        Returns:
            True if command executed successfully, False otherwise
        """
        # Handle command aliases
        command = self.command_registry.get_command_by_name_or_alias(command_name)
        
        if not command:
            self.print_error(f"Unknown command: {command_name}")
            self.print_system_message("Type 'help' to see available commands.")
            return False
        
        try:
            # Parse arguments based on the command's signature
            parsed_args = self._parse_command_args(command, args_str)
            
            # Execute the command with parsed arguments
            if asyncio.iscoroutinefunction(command.callback):
                result = await command.callback(**parsed_args)
            else:
                result = command.callback(**parsed_args)
            
            return True
            
        except Exception as e:
            self.print_error(f"Error executing command '{command_name}': {str(e)}")
            if self.settings["verbose"]:
                traceback.print_exc()
            return False
    
    def _parse_command_args(self, command: Command, args_str: str) -> Dict[str, Any]:
        """
        Parse command arguments based on the command's callback signature.
        
        Args:
            command: Command object
            args_str: Arguments string
            
        Returns:
            Dictionary of parsed arguments
        """
        # Get the signature of the callback function
        sig = inspect.signature(command.callback)
        
        # Create a simple parser for the arguments
        parsed_args = {}
        
        # If no parameters expected (besides self), return empty dict
        param_count = sum(1 for p in sig.parameters.values() 
                        if p.name != 'self' and p.kind != inspect.Parameter.VAR_KEYWORD)
        
        if param_count == 0:
            return parsed_args
        
        # If first parameter is args_str, pass the entire string
        if 'args_str' in sig.parameters:
            parsed_args['args_str'] = args_str
            return parsed_args
        
        # Otherwise, split by whitespace and assign positionally
        args_list = args_str.split()
        
        param_names = [p.name for p in sig.parameters.values() 
                     if p.name != 'self' and p.kind != inspect.Parameter.VAR_KEYWORD]
        
        for i, param_name in enumerate(param_names):
            if i < len(args_list):
                parsed_args[param_name] = args_list[i]
            else:
                # Check if parameter has a default value
                param = sig.parameters[param_name]
                if param.default != inspect.Parameter.empty:
                    parsed_args[param_name] = param.default
        
        return parsed_args
    
    # ----- Command Implementations -----
    
    def _cmd_help(self, command_name: str = None):
        """
        Show help information for available commands.
        
        Args:
            command_name: Optional command name to get help for
        """
        if command_name:
            # Show help for specific command
            command = self.command_registry.get_command_by_name_or_alias(command_name)
            if not command:
                self.print_error(f"Unknown command: {command_name}")
                return
            
            if RICH_AVAILABLE:
                self.console.print(f"[bold]{command.name}[/bold]: {command.description}")
                if command.usage:
                    self.console.print(f"Usage: [cyan]{command.usage}[/cyan]")
                if command.aliases:
                    self.console.print(f"Aliases: [yellow]{', '.join(command.aliases)}[/yellow]")
            else:
                print(f"{command.name}: {command.description}")
                if command.usage:
                    print(f"Usage: {command.usage}")
                if command.aliases:
                    print(f"Aliases: {', '.join(command.aliases)}")
        else:
            # Show all commands
            commands = self.command_registry.get_commands()
            
            if RICH_AVAILABLE:
                table = Table(title="Available Commands")
                table.add_column("Command", style="cyan")
                table.add_column("Description", style="green")
                table.add_column("Usage", style="yellow")
                
                for name, cmd in sorted(commands.items()):
                    table.add_row(name, cmd.description, cmd.usage or "")
                
                self.console.print(table)
            else:
                print("Available Commands:")
                for name, cmd in sorted(commands.items()):
                    print(f"  {name}: {cmd.description}")
                    if cmd.usage:
                        print(f"    Usage: {cmd.usage}")
    
    async def _cmd_exit(self):
        """Exit the application."""
        self.print_system_message("Exiting the application...")
        self.exit_requested = True
        await self._shutdown()
    
    def _cmd_clear(self):
        """Clear the terminal screen."""
        if RICH_AVAILABLE:
            self.console.clear()
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
    
    def _cmd_mode(self, mode: str = None):
        """
        Change the CLI mode.
        
        Args:
            mode: Mode to switch to (interactive, command, repl, script, monitor)
        """
        if not mode:
            self.print_system_message(f"Current mode: {self.mode.value}")
            self.print_system_message("Available modes: interactive, command, repl, script, monitor")
            return
        
        try:
            new_mode = CLIMode(mode.lower())
            self.print_system_message(f"Switching to {new_mode.value} mode...")
            self.mode = new_mode
            
            # Restart in the new mode
            if new_mode == CLIMode.INTERACTIVE:
                asyncio.create_task(self._run_interactive_mode())
            elif new_mode == CLIMode.REPL:
                asyncio.create_task(self._run_repl_mode())
            elif new_mode == CLIMode.MONITOR:
                asyncio.create_task(self._run_monitor_mode())
            else:
                self.print_error(f"Can't directly switch to {new_mode.value} mode. Please restart the CLI.")
                
        except ValueError:
            self.print_error(f"Invalid mode: {mode}")
            self.print_system_message("Available modes: interactive, command, repl, script, monitor")
    
    def _cmd_status(self):
        """Show system status information."""
        status = self.assistant.get_status()
        
        if RICH_AVAILABLE:
            table = Table(title="System Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="yellow")
            
            # System info
            table.add_row(
                "System",
                status.get("status", "unknown"),
                f"Version: {status.get('version', 'unknown')}, "
                f"Uptime: {status.get('uptime_seconds', 0):.1f}s"
            )
            
            # Component statuses
            components = status.get("components", {})
            for component_name, component_status in components.items():
                if component_status is None:
                    continue
                
                if isinstance(component_status, dict):
                    # Format details based on component type
                    details = ""
                    if component_name == "component_manager":
                        total = component_status.get("total_components", 0)
                        running = component_status.get("running_components", 0)
                        details = f"{running}/{total} components running"
                    elif component_name == "session_manager":
                        details = f"{component_status.get('total_active_sessions', 0)} active sessions"
                    elif component_name == "workflow_orchestrator":
                        details = f"{component_status.get('active_executions', 0)} active workflows"
                    elif component_name == "plugin_manager":
                        total = component_status.get("total_plugins", 0)
                        enabled = component_status.get("enabled_plugins", 0)
                        details = f"{enabled}/{total} plugins enabled"
                        
                    table.add_row(component_name, "active", details)
                else:
                    table.add_row(component_name, "unknown", "")
            
            self.console.print(table)
        else:
            print("=== System Status ===")
            print(f"Status: {status.get('status', 'unknown')}")
            print(f"Version: {status.get('version', 'unknown')}")
            print(f"Uptime: {status.get('uptime_seconds', 0):.1f}s")
            print()
            print("Components:")
            
            components = status.get("components", {})
            for component_name, component_status in components.items():
                if component_status is None:
                    continue
                
                if isinstance(component_status, dict):
                    if component_name == "component_manager":
                        total = component_status.get("total_components", 0)
                        running = component_status.get("running_components", 0)
                        print(f"- {component_name}: {running}/{total} components running")
                    elif component_name == "session_manager":
                        print(f"- {component_name}: {component_status.get('total_active_sessions', 0)} active sessions")
                    elif component_name == "workflow_orchestrator":
                        print(f"- {component_name}: {component_status.get('active_executions', 0)} active workflows")
                    elif component_name == "plugin_manager":
                        total = component_status.get("total_plugins", 0)
                        enabled = component_status.get("enabled_plugins", 0)
                        print(f"- {component_name}: {enabled}/{total} plugins enabled")
                    else:
                        print(f"- {component_name}: active")
                else:
                    print(f"- {component_name}: unknown")
    
    async def _cmd_session(self, action: str = "list", session_id: str = None):
        """
        Manage sessions.
        
        Args:
            action: Action to perform (new, list, switch, end)
            session_id: Optional session ID for switch/end actions
        """
        if action == "new":
            # Create a new session
            new_session_id = await self.assistant.session_manager.create_session(
                user_id=self.current_user_id
            )
            self.current_session_id = new_session_id
            self.print_system_message(f"Created new session: {new_session_id}")
            
        elif action == "list":
            # List active sessions
            sessions = self.assistant.session_manager.get_active_sessions()
            
            if RICH_AVAILABLE:
                table = Table(title="Active Sessions")
                table.add_column("Session ID", style="cyan")
                table.add_column("User ID", style="green")
                table.add_column("Created At", style="yellow")
                table.add_column("Status", style="blue")
                
                for session in sessions:
                    current_marker = "* " if session["session_id"] == self.current_session_id else ""
                    table.add_row(
                        f"{current_marker}{session['session_id']}", 
                        session.get("user_id", "anonymous"),
                        session.get("created_at", "unknown"),
                        session.get("state", "unknown")
                    )
                
                self.console.print(table)
            else:
                print("Active Sessions:")
                for session in sessions:
                    current_marker = "* " if session["session_id"] == self.current_session_id else ""
                    print(f"{current_marker}{session['session_id']} - User: {session.get('user_id', 'anonymous')}, "
                          f"State: {session.get('state', 'unknown')}")
            
        elif action == "switch":
            # Switch to a different session
            if not session_id:
                self.print_error("Session ID required for switch action")
                return
            
            sessions = self.assistant.session_manager.get_active_sessions()
            session_ids = [s["session_id"] for s in sessions]
            
            if session_id not in session_ids:
                self.print_error(f"Session {session_id} not found")
                return
            
            self.current_session_id = session_id
            self.print_system_message(f"Switched to session: {session_id}")
            
        elif action == "end":
            # End a session
            if not session_id:
                session_id = self.current_session_id
            
            await self.assistant.session_manager.end_session(session_id)
            
            if session_id == self.current_session_id:
                # Create a new session if we ended the current one
                new_session_id = await self.assistant.session_manager.create_session(
                    user_id=self.current_user_id
                )
                self.current_session_id = new_session_id
                self.print_system_message(f"Ended previous session and created new session: {new_session_id}")
            else:
                self.print_system_message(f"Ended session: {session_id}")
            
        else:
            self.print_error(f"Unknown session action: {action}")
            self.print_system_message("Available actions: new, list, switch, end")
    
    async def _cmd_workflow(self, action: str = "list", workflow_id: str = None, params_json: str = None):
        """
        Manage and execute workflows.
        
        Args:
            action: Action to perform (list, execute, status)
            workflow_id: Workflow ID for execute/status actions
            params_json: JSON string of parameters for execute action
        """
        if action == "list":
            # List available workflows
            workflows = self.assistant.workflow_orchestrator.list_workflows()
            
            if RICH_AVAILABLE:
                table = Table(title="Available Workflows")
                table.add_column("Workflow ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Version", style="yellow")
                table.add_column("Type", style="blue")
                
                for workflow in workflows:
                    table.add_row(
                        workflow["workflow_id"],
                        workflow.get("name", "unknown"),
                        workflow.get("version", "unknown"),
                        workflow.get("type", "unknown")
                    )
                
                self.console.print(table)
            else:
                print("Available Workflows:")
                for workflow in workflows:
                    print(f"{workflow['workflow_id']} - {workflow.get('name', 'unknown')} "
                          f"(v{workflow.get('version', 'unknown')}, {workflow.get('type', 'unknown')})")
            
        elif action == "execute":
            # Execute a workflow
            if not workflow_id:
                self.print_error("Workflow ID required for execute action")
                return
            
            # Parse parameters if provided
            params = {}
            if params_json:
                try:
                    params = json.loads(params_json)
                except json.JSONDecodeError:
                    self.print_error("Invalid JSON parameters")
                    return
            
            # Execute workflow
            self.print_system_message(f"Executing workflow: {workflow_id}")
            
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Executing workflow..."),
                    console=self.console
                ) as progress:
                    task = progress.add_task("execute", total=None)
                    execution_id = await self.assistant.workflow_orchestrator.execute_workflow(
                        workflow_id=workflow_id,
                        input_data=params,
                        session_id=self.current_session_id,
                        user_id=self.current_user_id
                    )
            else:
                print("Executing workflow...")
                execution_id = await self.assistant.workflow_orchestrator.execute_workflow(
                    workflow_id=workflow_id,
                    input_data=params,
                    session_id=self.current_session_id,
                    user_id=self.current_user_id
                )
            
            self.print_system_message(f"Workflow execution started. Execution ID: {execution_id}")
            
            # Wait for execution to complete
            status = await self.assistant.workflow_orchestrator.get_execution_status(execution_id)
            
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Waiting for workflow completion..."),
                    console=self.console
                ) as progress:
                    task = progress.add_task("wait", total=None)
                    
                    # Poll for completion
                    while status["state"] in ["running", "planning", "ready"]:
                        await asyncio.sleep(1)
                        status = await self.assistant.workflow_orchestrator.get_execution_status(execution_id)
            else:
                print("Waiting for workflow completion...")
                
                # Poll for completion
                while status["state"] in ["running", "planning", "ready"]:
                    await asyncio.sleep(1)
                    status = await self.assistant.workflow_orchestrator.get_execution_status(execution_id)
            
            # Show execution result
            if status["state"] == "completed":
                self.print_system_message(f"Workflow completed successfully in {status.get('execution_time', 0):.2f}s")
            else:
                self.print_error(f"Workflow execution failed: {status.get('state', 'unknown')}")
                if status.get("errors"):
                    for error in status["errors"]:
                        self.print_error(f"Error: {error}")
            
        elif action == "status":
            # Check workflow execution status
            if not workflow_id:
                # List active executions
                executions = self.assistant.workflow_orchestrator.get_active_executions()
                
                if RICH_AVAILABLE:
                    table = Table(title="Active Workflow Executions")
                    table.add_column("Execution ID", style="cyan")
                    table.add_column("Workflow ID", style="green")
                    table.add_column("State", style="yellow")
                    table.add_column("Progress", style="blue")
                    
                    for execution in executions:
                        table.add_row(
                            execution["execution_id"],
                            execution.get("workflow_id", "unknown"),
                            execution.get("state", "unknown"),
                            f"{execution.get('progress', 0):.0%}"
                        )
                    
                    self.console.print(table)
                else:
                    print("Active Workflow Executions:")
                    for execution in executions:
                        print(f"{execution['execution_id']} - Workflow: {execution.get('workflow_id', 'unknown')}, "
                              f"State: {execution.get('state', 'unknown')}, "
                              f"Progress: {execution.get('progress', 0):.0%}")
            else:
                # Check specific execution status
                try:
                    status = await self.assistant.workflow_orchestrator.get_execution_status(workflow_id)
                    
                    if RICH_AVAILABLE:
                        table = Table(title=f"Workflow Execution: {workflow_id}")
                        table.add_column("Property", style="cyan")
                        table.add_column("Value", style="green")
                        
                        for key, value in status.items():
                            if key not in ["current_steps", "completed_steps", "failed_steps", "errors", "warnings"]:
                                table.add_row(key, str(value))
                        
                        self.console.print(table)
                        
                        # Show steps information
                        if status.get("current_steps"):
                            self.console.print("[bold]Current Steps:[/bold]")
                            for step in status["current_steps"]:
                                self.console.print(f"- {step}")
                        
                        if status.get("completed_steps"):
                            self.console.print("[bold]Completed Steps:[/bold]")
                            for step in status["completed_steps"]:
                                self.console.print(f"- {step}")
                        
                        if status.get("failed_steps"):
                            self.console.print("[bold red]Failed Steps:[/bold red]")
                            for step in status["failed_steps"]:
                                self.console.print(f"- {step}")
                        
                        if status.get("errors"):
                            self.console.print("[bold red]Errors:[/bold red]")
                            for error in status["errors"]:
                                self.console.print(f"- {error}")
                    else:
                        print(f"Workflow Execution: {workflow_id}")
                        print(f"State: {status.get('state', 'unknown')}")
                        print(f"Progress: {status.get('progress', 0):.0%}")
                        print(f"Execution Time: {status.get('execution_time', 0):.2f}s")
                        
                        if status.get("current_steps"):
                            print("Current Steps:")
                            for step in status["current_steps"]:
                                print(f"- {step}")
                        
                        if status.get("completed_steps"):
                            print("Completed Steps:")
                            for step in status["completed_steps"]:
                                print(f"- {step}")
                        
                        if status.get("failed_steps"):
                            print("Failed Steps:")
                            for step in status["failed_steps"]:
                                print(f"- {step}")
                        
                        if status.get("errors"):
                            print("Errors:")
                            for error in status["errors"]:
                                print(f"- {error}")
                
                except Exception as e:
                    self.print_error(f"Error getting execution status: {str(e)}")
            
        else:
            self.print_error(f"Unknown workflow action: {action}")
            self.print_system_message("Available actions: list, execute, status")
    
    async def _cmd_plugin(self, action: str = "list", plugin_id: str = None):
        """
        Manage plugins.
        
        Args:
            action: Action to perform (list, load, unload, info)
            plugin_id: Plugin ID for load/unload/info actions
        """
        if action == "list":
            # List available plugins
            plugins = self.assistant.plugin_manager.list_plugins()
            
            if RICH_AVAILABLE:
                table = Table(title="Installed Plugins")
                table.add_column("Plugin ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Version", style="yellow")
                table.add_column("Type", style="blue")
                table.add_column("State", style="magenta")
                
                for plugin in plugins:
                    table.add_row(
                        plugin["plugin_id"],
                        plugin.get("name", "unknown"),
                        plugin.get("version", "unknown"),
                        plugin.get("type", "unknown"),
                        plugin.get("state", "unknown")
                    )
                
                self.console.print(table)
            else:
                print("Installed Plugins:")
                for plugin in plugins:
                    print(f"{plugin['plugin_id']} - {plugin.get('name', 'unknown')} "
                          f"(v{plugin.get('version', 'unknown')}, {plugin.get('type', 'unknown')}, "
                          f"{plugin.get('state', 'unknown')})")
            
        elif action == "load":
            # Load a plugin
            if not plugin_id:
                self.print_error("Plugin ID required for load action")
                return
            
            self.print_system_message(f"Loading plugin: {plugin_id}")
            
            try:
                await self.assistant.plugin_manager.load_plugin(plugin_id)
                self.print_system_message(f"Successfully loaded plugin: {plugin_id}")
            except Exception as e:
                self.print_error(f"Failed to load plugin: {str(e)}")
            
        elif action == "unload":
            # Unload a plugin
            if not plugin_id:
                self.print_error("Plugin ID required for unload action")
                return
            
            self.print_system_message(f"Unloading plugin: {plugin_id}")
            
            try:
                await self.assistant.plugin_manager.unload_plugin(plugin_id)
                self.print_system_message(f"Successfully unloaded plugin: {plugin_id}")
            except Exception as e:
                self.print_error(f"Failed to unload plugin: {str(e)}")
            
        elif action == "info":
            # Show plugin details
            if not plugin_id:
                self.print_error("Plugin ID required for info action")
                return
            
            try:
                plugin_info = self.assistant.plugin_manager.get_plugin_info(plugin_id)
                
                if not plugin_info:
                    self.print_error(f"Plugin not found: {plugin_id}")
                    return
                
                if RICH_AVAILABLE:
                    table = Table(title=f"Plugin Details: {plugin_id}")
                    table.add_column("Property", style="cyan")
                    table.add_column("Value", style="green")
                    
                    metadata = plugin_info.metadata
                    
                    table.add_row("Name", metadata.name)
                    table.add_row("Version", metadata.version)
                    table.add_row("Type", metadata.plugin_type.value)
                    table.add_row("State", plugin_info.state.value)
                    table.add_row("Author", metadata.author)
                    table.add_row("Description", metadata.description)
                    
                    if metadata.provides:
                        capabilities = ", ".join(c.name for c in metadata.provides)
                        table.add_row("Capabilities", capabilities)
                    
                    if metadata.dependencies:
                        dependencies = ", ".join(d.plugin_id for d in metadata.dependencies)
                        table.add_row("Dependencies", dependencies)
                    
                    if plugin_info.error_count > 0:
                        table.add_row("Error Count", str(plugin_info.error_count))
                        if plugin_info.last_error:
                            table.add_row("Last Error", str(plugin_info.last_error))
                    
                    self.console.print(table)
                else:
                    metadata = plugin_info.metadata
                    print(f"Plugin Details: {plugin_id}")
                    print(f"Name: {metadata.name}")
                    print(f"Version: {metadata.version}")
                    print(f"Type: {metadata.plugin_type.value}")
                    print(f"State: {plugin_info.state.value}")
                    print(f"Author: {metadata.author}")
                    print(f"Description: {metadata.description}")
                    
                    if metadata.provides:
                        capabilities = ", ".join(c.name for c in metadata.provides)
                        print(f"Capabilities: {capabilities}")
                    
                    if metadata.dependencies:
                        dependencies = ", ".join(d.plugin_id for d in metadata.dependencies)
                        print(f"Dependencies: {dependencies}")
                    
                    if plugin_info.error_count > 0:
                        print(f"Error Count: {plugin_info.error_count}")
                        if plugin_info.last_error:
                            print(f"Last Error: {plugin_info.last_error}")
            
            except Exception as e:
                self.print_error(f"Error getting plugin info: {str(e)}")
            
        else:
            self.print_error(f"Unknown plugin action: {action}")
            self.print_system_message("Available actions: list, load, unload, info")
    
    async def _cmd_component(self, action: str = "list", component_id: str = None):
        """
        Manage system components.
        
        Args:
            action: Action to perform (list, status, restart)
            component_id: Component ID for status/restart actions
        """
        if action == "list":
            # List all components
            components = self.assistant.component_manager.list_components()
            
            if RICH_AVAILABLE:
                table = Table(title="System Components")
                table.add_column("Component ID", style="cyan")
                table.add_column("State", style="green")
                
                for component_id in components:
                    state = "active"  # We'd need to get actual state
                    table.add_row(component_id, state)
                
                self.console.print(table)
            else:
                print("System Components:")
                for component_id in components:
                    state = "active"  # We'd need to get actual state
                    print(f"{component_id} - {state}")
            
        elif action == "status":
            # Show component status
            if not component_id:
                # Show status of all components
                status = self.assistant.component_manager.get_component_status()
                
                if RICH_AVAILABLE:
                    table = Table(title="Component Status")
                    table.add_column("Component", style="cyan")
                    table.add_column("State", style="green")
                    
                    for comp_id, comp_status in status.get("components", {}).items():
                        table.add_row(comp_id, comp_status.get("state", "unknown"))
                    
                    self.console.print(table)
                    self.console.print(f"Total components: {status.get('total_components', 0)}")
                    self.console.print(f"Running components: {status.get('running_components', 0)}")
                    self.console.print(f"Failed components: {status.get('failed_components', 0)}")
                else:
                    print("Component Status:")
                    for comp_id, comp_status in status.get("components", {}).items():
                        print(f"{comp_id} - {comp_status.get('state', 'unknown')}")
                    
                    print(f"Total components: {status.get('total_components', 0)}")
                    print(f"Running components: {status.get('running_components', 0)}")
                    print(f"Failed components: {status.get('failed_components', 0)}")
            else:
                # Show status of specific component
                try:
                    status = self.assistant.component_manager.get_component_status(component_id)
                    
                    if RICH_AVAILABLE:
                        table = Table(title=f"Component Status: {component_id}")
                        table.add_column("Property", style="cyan")
                        table.add_column("Value", style="green")
                        
                        for key, value in status.items():
                            table.add_row(key, str(value))
                        
                        self.console.print(table)
                    else:
                        print(f"Component Status: {component_id}")
                        for key, value in status.items():
                            print(f"{key}: {value}")
                
                except Exception as e:
                    self.print_error(f"Error getting component status: {str(e)}")
            
        elif action == "restart":
            # Restart a component
            if not component_id:
                self.print_error("Component ID required for restart action")
                return
            
            self.print_system_message(f"Restarting component: {component_id}")
            
            try:
                await self.assistant.component_manager.restart_component(component_id)
                self.print_system_message(f"Successfully restarted component: {component_id}")
            except Exception as e:
                self.print_error(f"Failed to restart component: {str(e)}")
            
        else:
            self.print_error(f"Unknown component action: {action}")
            self.print_system_message("Available actions: list, status, restart")
    
    def _cmd_config(self, action: str = "list", key: str = None, value: str = None):
        """
        View or modify configuration.
        
        Args:
            action: Action to perform (get, set, list)
            key: Configuration key
            value: Configuration value for set action
        """
        config_loader = self.assistant.config_loader
        
        if action == "list":
            # List all configuration values
            config = config_loader.get_all()
            
            if RICH_AVAILABLE:
                table = Table(title="System Configuration")
                table.add_column("Key", style="cyan")
                table.add_column("Value", style="green")
                
                for key, value in sorted(config.items()):
                    table.add_row(key, str(value))
                
                self.console.print(table)
            else:
                print("System Configuration:")
                for key, value in sorted(config.items()):
                    print(f"{key}: {value}")
            
        elif action == "get":
            # Get a specific configuration value
            if not key:
                self.print_error("Configuration key required for get action")
                return
            
            value = config_loader.get(key)
            self.print_system_message(f"{key}: {value}")
            
        elif action == "set":
            # Set a configuration value
            if not key:
                self.print_error("Configuration key required for set action")
                return
            
            if value is None:
                self.print_error("Configuration value required for set action")
                return
            
            # Try to convert value to appropriate type
            try:
                # Try as number
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                # Try as boolean
                elif value.lower() in ["true", "yes", "on"]:
                    value = True
                elif value.lower() in ["false", "no", "off"]:
                    value = False
            except (ValueError, AttributeError):
                # Keep as string if conversion fails
                pass
            
            # Set the configuration value
            config_loader.set(key, value)
            self.print_system_message(f"Set {key} = {value}")
            
        else:
            self.print_error(f"Unknown config action: {action}")
            self.print_system_message("Available actions: get, set, list")
    
    def _cmd_debug(self, action: str = "log", level: str = "info", target: str = None):
        """
        Debug tools and diagnostics.
        
        Args:
            action: Action to perform (log, trace, profile, memory)
            level: Logging level for log action
            target: Target component for profiling/debugging
        """
        if action == "log":
            # Set logging level
            log_levels = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL
            }
            
            if level not in log_levels:
                self.print_error(f"Unknown log level: {level}")
                self.print_system_message(f"Available levels: {', '.join(log_levels.keys())}")
                return
            
            # Set log level for the root logger
            logging.getLogger().setLevel(log_levels[level])
            self.print_system_message(f"Set logging level to {level}")
            
        elif action == "trace":
            # Enable/disable tracing
            if level.lower() in ["on", "enable", "true"]:
                # Enable tracing
                if hasattr(self.assistant, 'tracer'):
                    self.assistant.tracer.enable()
                    self.print_system_message("Tracing enabled")
                else:
                    self.print_error("Tracing not available")
            elif level.lower() in ["off", "disable", "false"]:
                # Disable tracing
                if hasattr(self.assistant, 'tracer'):
                    self.assistant.tracer.disable()
                    self.print_system_message("Tracing disabled")
                else:
                    self.print_error("Tracing not available")
            else:
                self.print_error(f"Unknown trace option: {level}")
                self.print_system_message("Available options: on, off")
            
        elif action == "profile":
            # Profile a component or operation
            self.print_system_message("Profiling not yet implemented")
            
        elif action == "memory":
            # Show memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                
                if RICH_AVAILABLE:
                    table = Table(title="Memory Usage")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    
                    table.add_row("RSS", f"{memory_info.rss / (1024 * 1024):.2f} MB")
                    table.add_row("VMS", f"{memory_info.vms / (1024 * 1024):.2f} MB")
                    table.add_row("Percent", f"{process.memory_percent():.2f}%")
                    
                    self.console.print(table)
                else:
                    print("Memory Usage:")
                    print(f"RSS: {memory_info.rss / (1024 * 1024):.2f} MB")
                    print(f"VMS: {memory_info.vms / (1024 * 1024):.2f} MB")
                    print(f"Percent: {process.memory_percent():.2f}%")
            except ImportError:
                self.print_error("psutil not available, cannot retrieve memory usage")
            
        else:
            self.print_error(f"Unknown debug action: {action}")
            self.print_system_message("Available actions: log, trace, profile, memory")
    
    def _cmd_upload(self, filepath: str = None):
        """
        Upload a file for processing.
        
        Args:
            filepath: Path to the file to upload
        """
        if not filepath:
            self.print_error("Filepath required for upload action")
            return
        
        filepath = os.path.expanduser(filepath)
        
        if not os.path.exists(filepath):
            self.print_error(f"File not found: {filepath}")
            return
        
        try:
            # Copy file to upload directory
            filename = os.path.basename(filepath)
            dest_path = os.path.join(self.settings["upload_dir"], filename)
            
            shutil.copy2(filepath, dest_path)
            
            self.print_system_message(f"Uploaded file to: {dest_path}")
            self.print_system_message(f"File reference: {filename}")
            
        except Exception as e:
            self.print_error(f"Error uploading file: {str(e)}")
    
    def _cmd_download(self, content_id: str = None, filepath: str = None):
        """
        Download generated content to a file.
        
        Args:
            content_id: ID of the content to download
            filepath: Path to save the file
        """
        self.print_system_message("Download functionality not yet implemented")
    
    def _cmd_settings(self, action: str = "list", key: str = None, value: str = None):
        """
        View or modify CLI settings.
        
        Args:
            action: Action to perform (get, set, list)
            key: Setting key
            value: Setting value for set action
        """
        if action == "list":
            # List all settings
            if RICH_AVAILABLE:
                table = Table(title="CLI Settings")
                table.add_column("Setting", style="cyan")
                table.add_column("Value", style="green")
                
                for key, value in sorted(self.settings.items()):
                    table.add_row(key, str(value))
                
                self.console.print(table)
            else:
                print("CLI Settings:")
                for key, value in sorted(self.settings.items()):
                    print(f"{key}: {value}")
        
        elif action == "get":
            # Get a specific setting
            if not key:
                self.print_error("Setting key required for get action")
                return
            
            value = self.settings.get(key)
            if value is not None:
                self.print_system_message(f"{key}: {value}")
            else:
                self.print_error(f"Setting not found: {key}")
        
        elif action == "set":
            # Set a setting value
            if not key:
                self.print_error("Setting key required for set action")
                return
            
            if value is None:
                self.print_error("Setting value required for set action")
                return
            
            # Try to convert value to appropriate type
            try:
                # Try as number
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                # Try as boolean
                elif value.lower() in ["true", "yes", "on"]:
                    value = True
                elif value.lower() in ["false", "no", "off"]:
                    value = False
            except (ValueError, AttributeError):
                # Keep as string if conversion fails
                pass
            
            # Set the setting value
            self.settings[key] = value
            self.print_system_message(f"Set {key} = {value}")
        
        else:
            self.print_error(f"Unknown settings action: {action}")
            self.print_system_message("Available actions: get, set, list")

    def _cmd_history(self, action: str = "list", count: str = "10"):
        """
        View or manage command history.
        
        Args:
            action: Action to perform (list, clear, save, load)
            count: Number of history items to show for list action
        """
        if action == "list":
            # List command history
            try:
                count_int = int(count)
            except ValueError:
                count_int = 10
            
            history_length = readline.get_current_history_length()
            start_index = max(1, history_length - count_int + 1)
            
            if RICH_AVAILABLE:
                table = Table(title=f"Command History (last {count_int})")
                table.add_column("#", style="cyan")
                table.add_column("Command", style="green")
                
                for i in range(start_index, history_length + 1):
                    command = readline.get_history_item(i)
                    if command:
                        table.add_row(str(i), command)
                
                self.console.print(table)
            else:
                print(f"Command History (last {count_int}):")
                for i in range(start_index, history_length + 1):
                    command = readline.get_history_item(i)
                    if command:
                        print(f"{i}: {command}")
        
        elif action == "clear":
            # Clear command history
            readline.clear_history()
            self.print_system_message("Command history cleared")
        
        elif action == "save":
            # Save history to file
            try:
                readline.write_history_file(self.command_history_file)
                self.print_system_message(f"History saved to {self.command_history_file}")
            except Exception as e:
                self.print_error(f"Error saving history: {str(e)}")
        
        elif action == "load":
            # Load history from file
            if os.path.exists(self.command_history_file):
                try:
                    readline.read_history_file(self.command_history_file)
                    self.print_system_message(f"History loaded from {self.command_history_file}")
                except Exception as e:
                    self.print_error(f"Error loading history: {str(e)}")
            else:
                self.print_error(f"History file not found: {self.command_history_file}")
        
        else:
            self.print_error(f"Unknown history action: {action}")
            self.print_system_message("Available actions: list, clear, save, load")

    # ----- Utility Methods -----
    
    def print_system_message(self, message: str):
        """Print a system message with appropriate formatting."""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.settings['system_color']}]System: {message}[/{self.settings['system_color']}]")
        else:
            print(f"System: {message}")
    
    def print_error(self, message: str):
        """Print an error message with appropriate formatting."""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.settings['error_color']}]Error: {message}[/{self.settings['error_color']}]")
        else:
            print(f"Error: {message}")
    
    def print_user_message(self, message: str):
        """Print a user message with appropriate formatting."""
        if RICH_AVAILABLE:
            self.console.print(f"[{self.settings['user_color']}]User: {message}[/{self.settings['user_color']}]")
        else:
            print(f"User: {message}")
    
    def print_assistant_message(self, message: str):
        """Print an assistant message with appropriate formatting."""
        if RICH_AVAILABLE:
            if self.settings["use_markdown"]:
                self.console.print(Markdown(message))
            else:
                self.console.print(f"[{self.settings['assistant_color']}]Assistant: {message}[/{self.settings['assistant_color']}]")
        else:
            print(f"Assistant: {message}")
    
    async def _shutdown(self):
        """Shutdown the CLI and assistant."""
        if self.assistant and self._assistant_initialized:
            await self.assistant.shutdown()
        
        # Save history if enabled
        if self.settings["save_history"]:
            try:
                readline.write_history_file(self.command_history_file)
            except Exception as e:
                self.logger.warning(f"Could not save history: {str(e)}")


async def async_main(args=None):
    """Async entry point for the CLI."""
    cli = AssistantCLI()
    try:
        await cli.run(args)
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        await cli._shutdown()


def main(args=None):
    """Main entry point for the CLI."""
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nShutdown completed")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
