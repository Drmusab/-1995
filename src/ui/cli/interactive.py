"""
Interactive CLI Session
Author: Drmusab
Last Modified: 2025-07-05 11:03:07 UTC

This module provides an interactive command-line interface for the AI assistant,
allowing users to engage in a conversation with the assistant through a terminal
session with rich formatting, command handling, and context awareness.
"""

import json
import logging
import os
import readline
import shlex
import signal
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import asyncio

# Rich terminal UI components
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

# Assistant imports
from src.assistant.core import InteractionHandler
from src.assistant.core import EnhancedSessionManager as SessionManager
from src.core.config.loader import ConfigLoader

# Core imports
from src.core.dependency_injection import Container
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    AssistantMessageSent,
    ErrorOccurred,
    InteractionCompleted,
    UserMessageReceived,
)

# Memory imports
from src.memory.operations.context_manager import MemoryContextManager

# Observability imports
from src.observability.logging.config import get_logger

# Initialize rich console
console = Console()
error_console = Console(stderr=True)


class CommandType(Enum):
    """Types of special commands in the interactive session."""

    SYSTEM = "system"  # System commands like exit, help
    CONTEXT = "context"  # Context management commands
    SKILL = "skill"  # Skill-related commands
    FILE = "file"  # File operations
    SETTINGS = "settings"  # Session settings
    DEBUG = "debug"  # Debug commands


@dataclass
class Command:
    """A special command that can be executed in the interactive session."""

    name: str
    handler: Callable
    help_text: str
    command_type: CommandType = CommandType.SYSTEM
    aliases: List[str] = None


class MessageRole(Enum):
    """Message roles in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """A message in the conversation history."""

    role: MessageRole
    content: str
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


class InteractiveSession:
    """
    Interactive CLI session for the AI assistant.

    This class provides a command-line interface for interacting with the
    AI assistant, handling user inputs, displaying assistant responses,
    and managing special commands.
    """

    def __init__(
        self,
        container: Container,
        session_id: str = None,
        user_id: str = None,
        prompt_symbol: str = "🤔",
        response_symbol: str = "🤖",
        welcome_message: str = None,
    ):
        """
        Initialize the interactive session.

        Args:
            container: Dependency injection container
            session_id: Optional session ID (will create new if None)
            user_id: Optional user ID
            prompt_symbol: Symbol to show in the user prompt
            response_symbol: Symbol to show for assistant responses
            welcome_message: Optional custom welcome message
        """
        self.container = container
        self.logger = get_logger(__name__)

        # Set session parameters
        self.session_id = session_id
        self.user_id = user_id or os.environ.get("USER", "anonymous")
        self.prompt_symbol = prompt_symbol
        self.response_symbol = response_symbol
        self.welcome_message = welcome_message

        # Get required components
        self.interaction_handler = container.get(InteractionHandler)
        self.session_manager = container.get(SessionManager)
        self.event_bus = container.get(EventBus)
        self.config_loader = container.get(ConfigLoader)

        # Try to get optional components
        try:
            self.context_manager = container.get(MemoryContextManager)
        except Exception:
            self.context_manager = None
            self.logger.warning("MemoryContextManager not available")

        # Initialize state
        self.running = False
        self.messages: List[Message] = []
        self.stream_output = True
        self.save_history = True
        self.multiline_input = False
        self.command_prefix = "/"
        self.show_timestamps = False
        self.debug_mode = False

        # User preferences
        self.preferences = {
            "stream_output": self.stream_output,
            "multiline_input": self.multiline_input,
            "command_prefix": self.command_prefix,
            "show_timestamps": self.show_timestamps,
            "debug_mode": self.debug_mode,
        }

        # Command history
        self.history = []
        self.history_file = os.path.expanduser("~/.aiassistant_history")

        # Register special commands
        self.commands = self._register_commands()

        # Setup readline if available
        self._setup_readline()

        # Register signal handlers
        self._register_signal_handlers()

    def _register_commands(self) -> Dict[str, Command]:
        """
        Register special commands for the interactive session.

        Returns:
            Dictionary of commands by name
        """
        commands = {}

        # System commands
        commands["help"] = Command(
            name="help",
            handler=self._cmd_help,
            help_text="Show available commands",
            command_type=CommandType.SYSTEM,
            aliases=["?"],
        )

        commands["exit"] = Command(
            name="exit",
            handler=self._cmd_exit,
            help_text="Exit the interactive session",
            command_type=CommandType.SYSTEM,
            aliases=["quit", "q"],
        )

        commands["clear"] = Command(
            name="clear",
            handler=self._cmd_clear,
            help_text="Clear the terminal screen",
            command_type=CommandType.SYSTEM,
            aliases=["cls"],
        )

        commands["history"] = Command(
            name="history",
            handler=self._cmd_history,
            help_text="Show conversation history",
            command_type=CommandType.SYSTEM,
        )

        # Context commands
        commands["context"] = Command(
            name="context",
            handler=self._cmd_context,
            help_text="Show or modify conversation context",
            command_type=CommandType.CONTEXT,
        )

        commands["reset"] = Command(
            name="reset",
            handler=self._cmd_reset,
            help_text="Reset the conversation context",
            command_type=CommandType.CONTEXT,
        )

        # File commands
        commands["upload"] = Command(
            name="upload",
            handler=self._cmd_upload,
            help_text="Upload a file to the conversation",
            command_type=CommandType.FILE,
            aliases=["file"],
        )

        commands["save"] = Command(
            name="save",
            handler=self._cmd_save,
            help_text="Save conversation history to a file",
            command_type=CommandType.FILE,
        )

        # Settings commands
        commands["settings"] = Command(
            name="settings",
            handler=self._cmd_settings,
            help_text="Show or modify session settings",
            command_type=CommandType.SETTINGS,
            aliases=["config"],
        )

        commands["multiline"] = Command(
            name="multiline",
            handler=self._cmd_multiline,
            help_text="Toggle multiline input mode",
            command_type=CommandType.SETTINGS,
            aliases=["ml"],
        )

        # Debug commands
        commands["debug"] = Command(
            name="debug",
            handler=self._cmd_debug,
            help_text="Toggle debug mode",
            command_type=CommandType.DEBUG,
        )

        commands["info"] = Command(
            name="info",
            handler=self._cmd_info,
            help_text="Show session information",
            command_type=CommandType.DEBUG,
        )

        # Register aliases as separate command references
        aliases = {}
        for cmd_name, cmd in commands.items():
            if cmd.aliases:
                for alias in cmd.aliases:
                    aliases[alias] = cmd

        # Merge commands and aliases
        commands.update(aliases)

        return commands

    def _setup_readline(self) -> None:
        """Setup readline for command history and editing."""
        if "readline" in sys.modules:
            # Load history if exists
            if os.path.exists(self.history_file) and self.save_history:
                try:
                    readline.read_history_file(self.history_file)
                    self.logger.debug(f"Loaded history from {self.history_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load history: {str(e)}")

            # Set history length
            readline.set_history_length(1000)

            # Setup tab completion
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self._completer)

    def _completer(self, text: str, state: int) -> Optional[str]:
        """
        Tab completion for commands and arguments.

        Args:
            text: Current text to complete
            state: State of completion cycle

        Returns:
            Completed text or None if no match
        """
        # Start with command completion
        if text.startswith(self.command_prefix):
            cmd_text = text[len(self.command_prefix) :]
            matches = [
                f"{self.command_prefix}{cmd}"
                for cmd in self.commands.keys()
                if cmd.startswith(cmd_text)
            ]
            try:
                return matches[state]
            except IndexError:
                return None

        # No completion for regular text
        return None

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for clean shutdown."""
        # Handle Ctrl+C (SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)

        # Handle SIGTERM
        signal.signal(signal.SIGTERM, self._handle_terminate)

    def _handle_interrupt(self, _sig, _frame) -> None:
        """
        Handle interrupt signal (Ctrl+C).

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        # Ask for confirmation before exiting
        console.print("\n[bold yellow]Interrupted. Exit session? (y/N)[/]", end=" ")
        try:
            choice = input().lower()
            if choice in ("y", "yes"):
                self._cleanup_and_exit()
            else:
                console.print("[yellow]Continuing session[/]")
        except Exception:
            # If input fails, just exit
            self._cleanup_and_exit()

    def _handle_terminate(self, _sig, _frame) -> None:
        """
        Handle terminate signal.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        console.print("\n[bold red]Received termination signal. Exiting...[/]")
        self._cleanup_and_exit()

    def _cleanup_and_exit(self) -> None:
        """Clean up resources and exit the session."""
        # Save command history
        if "readline" in sys.modules and self.save_history:
            try:
                readline.write_history_file(self.history_file)
            except Exception as e:
                self.logger.warning(f"Failed to save history: {str(e)}")

        # Set running flag to False to exit the main loop
        self.running = False

        # Close session if we created it
        if self.session_id and self._session_owned:
            asyncio.create_task(self._close_session())

        # Exit with a newline for cleaner terminal
        console.print()
        sys.exit(0)

    async def _close_session(self) -> None:
        """Close the current session."""
        try:
            await self.session_manager.close_session(self.session_id)
            self.logger.info(f"Closed session {self.session_id}")
        except Exception as e:
            self.logger.error(f"Error closing session: {str(e)}")

    async def run(self) -> None:
        """
        Run the interactive session.

        This is the main entry point for the interactive session, starting
        the REPL loop and handling user inputs.
        """
        self.running = True
        self._session_owned = False

        # Initialize session if needed
        if not self.session_id:
            try:
                self.session_id = await self.session_manager.create_session(
                    user_id=self.user_id, metadata={"source": "cli", "interface": "interactive"}
                )
                self._session_owned = True
                self.logger.info(f"Created new session: {self.session_id}")
            except Exception as e:
                error_console.print(f"[bold red]Error creating session:[/] {str(e)}")
                return
        else:
            # Verify session exists
            session = await self.session_manager.get_session(self.session_id)
            if not session:
                error_console.print(f"[bold red]Error:[/] Session {self.session_id} not found")
                return

        # Show welcome message
        await self._show_welcome()

        # Main interaction loop
        while self.running:
            try:
                # Get user input
                user_input = await self._get_user_input()

                # Skip empty inputs
                if not user_input or user_input.isspace():
                    continue

                # Check for special commands
                if user_input.startswith(self.command_prefix):
                    await self._handle_command(user_input[len(self.command_prefix) :])
                    continue

                # Process regular message
                await self._process_user_message(user_input)

            except KeyboardInterrupt:
                # Handle Ctrl+C during input
                console.print("\n[yellow]Input canceled[/]")
                continue

            except EOFError:
                # Handle Ctrl+D (EOF)
                console.print("\n[yellow]Received EOF, exiting...[/]")
                break

            except Exception as e:
                error_console.print(f"\n[bold red]Error in interactive session:[/] {str(e)}")
                if self.debug_mode:
                    error_console.print(traceback.format_exc())

        # Cleanup when loop exits
        if self._session_owned:
            await self._close_session()

    async def _show_welcome(self) -> None:
        """Show welcome message and session information."""
        # Custom welcome message if provided
        if self.welcome_message:
            console.print(f"\n{self.welcome_message}\n")
        else:
            # Default welcome message
            console.print("\n[bold blue]Welcome to the AI Assistant Interactive Session[/]")
            console.print("Type your messages to chat with the assistant")
            console.print(f"Type [bold]{self.command_prefix}help[/] for available commands")

        # Show session info
        console.print(f"\n[dim]Session ID: {self.session_id}[/]")
        console.print(f"[dim]User: {self.user_id}[/]")
        console.print(
            f"[dim]Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/]"
        )
        console.print("\n" + "─" * console.width + "\n")

    async def _get_user_input(self) -> str:
        """
        Get input from the user.

        Returns:
            User input string
        """
        if self.multiline_input:
            # Show multiline input prompt
            console.print(
                f"\n{self.prompt_symbol} [bold blue]Enter your message (Ctrl+D or type '\\end' on a new line to finish):[/]"
            )
            lines = []

            while True:
                try:
                    line = input("... ")
                    if line.strip() == "\\end":
                        break
                    lines.append(line)
                except EOFError:
                    # End on Ctrl+D
                    console.print()
                    break

            return "\n".join(lines)
        else:
            # Show single-line input prompt
            prompt_text = f"{self.prompt_symbol} "
            user_input = Prompt.ask(prompt_text, console=console)
            return user_input

    async def _handle_command(self, cmd_text: str) -> None:
        """
        Handle a special command.

        Args:
            cmd_text: Command text without prefix
        """
        # Parse command and arguments
        parts = shlex.split(cmd_text)
        if not parts:
            return

        cmd_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        # Find command
        if cmd_name in self.commands:
            command = self.commands[cmd_name]
            try:
                # Execute command handler
                await command.handler(args)
            except Exception as e:
                error_console.print(f"[bold red]Error executing command:[/] {str(e)}")
                if self.debug_mode:
                    error_console.print(traceback.format_exc())
        else:
            error_console.print(f"[bold red]Unknown command:[/] {cmd_name}")
            error_console.print(f"Type [bold]{self.command_prefix}help[/] for available commands")

    async def _process_user_message(self, message: str) -> None:
        """
        Process a user message and get assistant response.

        Args:
            message: User message text
        """
        # Add to message history
        user_message = Message(
            role=MessageRole.USER,
            content=message,
            timestamp=datetime.now(timezone.utc),
            metadata={"session_id": self.session_id},
        )
        self.messages.append(user_message)

        # Emit event
        await self.event_bus.emit(
            UserMessageReceived(
                session_id=self.session_id,
                user_id=self.user_id,
                message=message,
                timestamp=user_message.timestamp.isoformat(),
            )
        )

        try:
            # Show typing indicator
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Assistant is thinking...[/]"),
                console=console,
                transient=True,
            ) as progress:
                progress_task = progress.add_task("thinking", total=None)

                # Get response from assistant
                if self.stream_output:
                    # For streaming, we need to handle the response differently
                    response_text = await self._handle_streaming_response(message)
                else:
                    # For non-streaming, get the complete response
                    response_text = await self.interaction_handler.handle_text_input(
                        text=message, session_id=self.session_id, user_id=self.user_id
                    )

            # Add assistant response to history
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_text,
                timestamp=datetime.now(timezone.utc),
                metadata={"session_id": self.session_id},
            )
            self.messages.append(assistant_message)

            # Emit event
            await self.event_bus.emit(
                AssistantMessageSent(
                    session_id=self.session_id,
                    message=response_text,
                    timestamp=assistant_message.timestamp.isoformat(),
                )
            )

            # Emit interaction completed event
            await self.event_bus.emit(
                InteractionCompleted(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    interaction_type="text",
                    success=True,
                )
            )

        except Exception as e:
            error_console.print(f"[bold red]Error getting assistant response:[/] {str(e)}")
            if self.debug_mode:
                error_console.print(traceback.format_exc())

            # Emit error event
            await self.event_bus.emit(
                ErrorOccurred(
                    component="interactive_session",
                    error_type="assistant_response",
                    error_message=str(e),
                )
            )

    async def _handle_streaming_response(self, message: str) -> str:
        """
        Handle streaming response from the assistant.

        Args:
            message: User message text

        Returns:
            Complete response text
        """
        # Start streaming response
        stream = await self.interaction_handler.handle_text_input_stream(
            text=message, session_id=self.session_id, user_id=self.user_id
        )

        # Display response header
        console.print(f"\n{self.response_symbol} [bold green]Assistant:[/]")

        # Accumulate the complete response
        complete_response = ""

        # Process the stream
        is_first_chunk = True
        try:
            async for chunk in stream:
                if is_first_chunk:
                    # Add a newline before the first chunk
                    console.print()
                    is_first_chunk = False

                # Print the chunk
                console.print(chunk, end="", highlight=False)

                # Accumulate the response
                complete_response += chunk

                # Ensure the output is displayed immediately
                console.file.flush()

                # Small delay to make the streaming effect more visible
                await asyncio.sleep(0.01)

            # Add a final newline
            console.print()

        except Exception as e:
            error_console.print(f"\n[bold red]Error during streaming response:[/] {str(e)}")
            if self.debug_mode:
                error_console.print(traceback.format_exc())

        return complete_response

    async def _cmd_help(self, args: List[str]) -> None:
        """
        Show help information.

        Args:
            args: Command arguments
        """
        if args:
            # Show help for specific command
            cmd_name = args[0].lower()
            if cmd_name in self.commands:
                command = self.commands[cmd_name]
                console.print(f"\n[bold blue]Help for command: {self.command_prefix}{cmd_name}[/]")
                console.print(f"{command.help_text}")

                # Show aliases if any
                if command.aliases:
                    aliases = ", ".join(
                        f"{self.command_prefix}{alias}" for alias in command.aliases
                    )
                    console.print(f"Aliases: {aliases}")
            else:
                error_console.print(f"[bold red]Unknown command:[/] {cmd_name}")
        else:
            # Show all commands grouped by type
            console.print("\n[bold blue]Available Commands:[/]")

            # Group commands by type
            by_type: Dict[CommandType, List[Command]] = {}
            for cmd_name, cmd in self.commands.items():
                # Skip aliases
                if cmd.aliases and cmd_name in cmd.aliases:
                    continue

                if cmd.command_type not in by_type:
                    by_type[cmd.command_type] = []

                by_type[cmd.command_type].append(cmd)

            # Display commands by type
            for cmd_type, cmds in by_type.items():
                console.print(f"\n[bold green]{cmd_type.value.capitalize()} Commands:[/]")

                table = Table(show_header=True, box=None)
                table.add_column("Command", style="cyan")
                table.add_column("Description")
                table.add_column("Aliases", style="dim")

                for cmd in sorted(cmds, key=lambda c: c.name):
                    aliases = ", ".join(cmd.aliases) if cmd.aliases else ""
                    table.add_row(f"{self.command_prefix}{cmd.name}", cmd.help_text, aliases)

                console.print(table)

    async def _cmd_exit(self, args: List[str]) -> None:
        """
        Exit the interactive session.

        Args:
            args: Command arguments
        """
        console.print("[yellow]Exiting interactive session...[/]")
        self.running = False

    async def _cmd_clear(self, args: List[str]) -> None:
        """
        Clear the terminal screen.

        Args:
            args: Command arguments
        """
        # Clear screen (cross-platform)
        os.system("cls" if os.name == "nt" else "clear")

    async def _cmd_history(self, args: List[str]) -> None:
        """
        Show conversation history.

        Args:
            args: Command arguments
        """
        # Parse limit argument
        limit = 10  # Default
        if args and args[0].isdigit():
            limit = int(args[0])

        # Get messages to display
        messages_to_show = self.messages[-limit:] if limit > 0 else self.messages

        if not messages_to_show:
            console.print("[yellow]No conversation history yet[/]")
            return

        console.print(
            f"\n[bold blue]Conversation History (showing {len(messages_to_show)} of {len(self.messages)} messages):[/]\n"
        )

        # Display messages
        for i, msg in enumerate(messages_to_show):
            # Format timestamp
            timestamp = msg.timestamp.strftime("%H:%M:%S")

            # Format based on role
            if msg.role == MessageRole.USER:
                role_text = f"{self.prompt_symbol} [bold blue]You"
                if self.show_timestamps:
                    role_text += f" ({timestamp})"
                role_text += ":[/]"
                console.print(role_text)
                console.print(msg.content)
            elif msg.role == MessageRole.ASSISTANT:
                role_text = f"{self.response_symbol} [bold green]Assistant"
                if self.show_timestamps:
                    role_text += f" ({timestamp})"
                role_text += ":[/]"
                console.print(role_text)

                # Format markdown for assistant responses
                console.print(Markdown(msg.content))
            elif msg.role == MessageRole.SYSTEM:
                role_text = f"[bold yellow]System"
                if self.show_timestamps:
                    role_text += f" ({timestamp})"
                role_text += ":[/]"
                console.print(role_text)
                console.print(f"[yellow]{msg.content}[/]")

            # Add separator between messages
            if i < len(messages_to_show) - 1:
                console.print()

    async def _cmd_context(self, args: List[str]) -> None:
        """
        Show or modify conversation context.

        Args:
            args: Command arguments
        """
        if not self.context_manager:
            error_console.print("[bold red]Error:[/] Context manager not available")
            return

        if not args:
            # Show current context
            context = await self.context_manager.get_context_dict(self.session_id)

            if not context:
                console.print("[yellow]No context available for this session[/]")
                return

            console.print("\n[bold blue]Current Conversation Context:[/]")

            # Format and display context
            for context_type, content in context.items():
                console.print(f"\n[bold green]{context_type.capitalize()}:[/]")

                if isinstance(content, dict):
                    # Format dict as YAML for readability
                    import yaml

                    yaml_str = yaml.dump(content, default_flow_style=False)
                    console.print(Syntax(yaml_str, "yaml"))
                elif isinstance(content, list):
                    # Show list items
                    for item in content:
                        if isinstance(item, dict):
                            # Summarize dict items
                            summary = ", ".join(
                                f"{k}: {v}"
                                for k, v in item.items()
                                if not isinstance(v, (dict, list))
                            )
                            console.print(f"- {summary}")
                        else:
                            console.print(f"- {item}")
                else:
                    console.print(str(content))
        else:
            # Context subcommands
            subcommand = args[0].lower()

            if subcommand == "clear":
                # Clear context
                success = await self.context_manager.clear_context(self.session_id)
                if success:
                    console.print("[bold green]Context cleared successfully[/]")
                else:
                    error_console.print("[bold red]Error:[/] Failed to clear context")

            elif subcommand == "add" and len(args) >= 3:
                # Add context element
                context_type = args[1]
                content = " ".join(args[2:])

                try:
                    # Try to parse as JSON
                    import json

                    content_obj = json.loads(content)
                except json.JSONDecodeError:
                    # Use as string if not valid JSON
                    content_obj = content

                # Add context element
                from src.memory.operations.context_manager import ContextPriority, ContextType

                try:
                    context_type_enum = ContextType(context_type.upper())
                except ValueError:
                    error_console.print(f"[bold red]Error:[/] Invalid context type: {context_type}")
                    return

                element_id = await self.context_manager.add_context_element(
                    session_id=self.session_id,
                    content=content_obj,
                    context_type=context_type_enum,
                    priority=ContextPriority.MEDIUM,
                )

                if element_id:
                    console.print(f"[bold green]Context element added with ID: {element_id}[/]")
                else:
                    error_console.print("[bold red]Error:[/] Failed to add context element")

            else:
                error_console.print(f"[bold red]Error:[/] Unknown context subcommand: {subcommand}")
                console.print("Available subcommands: clear, add <type> <content>")

    async def _cmd_reset(self, args: List[str]) -> None:
        """
        Reset the conversation context.

        Args:
            args: Command arguments
        """
        # Confirm reset
        confirm = Confirm.ask("Are you sure you want to reset the conversation?")
        if not confirm:
            console.print("[yellow]Reset cancelled[/]")
            return

        # Clear context if context manager available
        if self.context_manager:
            await self.context_manager.clear_context(self.session_id)

        # Clear message history
        self.messages.clear()

        # Add system message
        system_message = Message(
            role=MessageRole.SYSTEM,
            content="Conversation has been reset.",
            timestamp=datetime.now(timezone.utc),
        )
        self.messages.append(system_message)

        console.print("[bold green]Conversation has been reset[/]")

    async def _cmd_upload(self, args: List[str]) -> None:
        """
        Upload a file to the conversation.

        Args:
            args: Command arguments
        """
        if not args:
            error_console.print("[bold red]Error:[/] File path required")
            console.print(f"Usage: {self.command_prefix}upload <file_path>")
            return

        file_path = args[0]

        # Check if file exists
        if not os.path.exists(file_path):
            error_console.print(f"[bold red]Error:[/] File not found: {file_path}")
            return

        # Get file info
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        # Check file size (10MB limit)
        if file_size > 10 * 1024 * 1024:
            error_console.print("[bold red]Error:[/] File too large (max 10MB)")
            return

        # Handle different file types
        try:
            file_content = None
            file_type = "unknown"

            # Determine file type and read content
            if file_ext in (".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml"):
                # Text file
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                file_type = "text"
            elif file_ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
                # Image file
                file_type = "image"
                error_console.print(
                    "[bold yellow]Note:[/] Image processing not implemented in this version"
                )
                return
            elif file_ext in (".pdf", ".doc", ".docx"):
                # Document file
                file_type = "document"
                error_console.print(
                    "[bold yellow]Note:[/] Document processing not implemented in this version"
                )
                return
            else:
                # Binary file - not supported
                error_console.print(f"[bold red]Error:[/] Unsupported file type: {file_ext}")
                return

            if file_content is not None:
                # Add file content to the conversation
                file_info = f"I'm sending you the contents of the file: {os.path.basename(file_path)}\n\n```{file_ext[1:]}\n{file_content}\n```"

                # Process as a user message
                await self._process_user_message(file_info)

        except Exception as e:
            error_console.print(f"[bold red]Error uploading file:[/] {str(e)}")

    async def _cmd_save(self, args: List[str]) -> None:
        """
        Save conversation history to a file.

        Args:
            args: Command arguments
        """
        if not self.messages:
            console.print("[yellow]No conversation to save[/]")
            return

        # Default filename
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        # Use provided filename if available
        if args:
            filename = args[0]

        try:
            # Create markdown export
            content = [f"# Conversation Export\n"]
            content.append(f"Session ID: {self.session_id}")
            content.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            content.append(f"User: {self.user_id}\n")

            # Add messages
            for msg in self.messages:
                # Format timestamp
                timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")

                # Format based on role
                if msg.role == MessageRole.USER:
                    content.append(f"## User ({timestamp})\n")
                    content.append(msg.content)
                elif msg.role == MessageRole.ASSISTANT:
                    content.append(f"## Assistant ({timestamp})\n")
                    content.append(msg.content)
                elif msg.role == MessageRole.SYSTEM:
                    content.append(f"## System ({timestamp})\n")
                    content.append(msg.content)

                # Add separator
                content.append("\n---\n")

            # Write to file
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(content))

            console.print(f"[bold green]Conversation saved to {filename}[/]")

        except Exception as e:
            error_console.print(f"[bold red]Error saving conversation:[/] {str(e)}")

    async def _cmd_settings(self, args: List[str]) -> None:
        """
        Show or modify session settings.

        Args:
            args: Command arguments
        """
        if not args:
            # Show current settings
            console.print("\n[bold blue]Current Settings:[/]")

            table = Table(show_header=True)
            table.add_column("Setting")
            table.add_column("Value")
            table.add_column("Description")

            table.add_row("stream_output", str(self.stream_output), "Stream assistant responses")
            table.add_row(
                "multiline_input", str(self.multiline_input), "Enable multiline input mode"
            )
            table.add_row("command_prefix", self.command_prefix, "Command prefix character")
            table.add_row("show_timestamps", str(self.show_timestamps), "Show message timestamps")
            table.add_row("debug_mode", str(self.debug_mode), "Enable debug mode")

            console.print(table)
            console.print(
                f"\nTo change a setting: {self.command_prefix}settings set <name> <value>"
            )

        elif len(args) >= 1:
            subcommand = args[0].lower()

            if subcommand == "set" and len(args) >= 3:
                # Set a setting
                setting_name = args[1].lower()
                setting_value = args[2].lower()

                if setting_name == "stream_output":
                    self.stream_output = setting_value in ("true", "yes", "1", "on")
                    console.print(f"[bold green]Set stream_output to {self.stream_output}[/]")

                elif setting_name == "multiline_input":
                    self.multiline_input = setting_value in ("true", "yes", "1", "on")
                    console.print(f"[bold green]Set multiline_input to {self.multiline_input}[/]")

                elif setting_name == "command_prefix":
                    self.command_prefix = setting_value
                    console.print(f"[bold green]Set command_prefix to {self.command_prefix}[/]")

                elif setting_name == "show_timestamps":
                    self.show_timestamps = setting_value in ("true", "yes", "1", "on")
                    console.print(f"[bold green]Set show_timestamps to {self.show_timestamps}[/]")

                elif setting_name == "debug_mode":
                    self.debug_mode = setting_value in ("true", "yes", "1", "on")
                    console.print(f"[bold green]Set debug_mode to {self.debug_mode}[/]")

                else:
                    error_console.print(f"[bold red]Error:[/] Unknown setting: {setting_name}")

                # Update preferences dict
                self.preferences[setting_name] = getattr(self, setting_name, None)

            elif subcommand == "reset":
                # Reset settings to defaults
                self.stream_output = True
                self.multiline_input = False
                self.command_prefix = "/"
                self.show_timestamps = False
                self.debug_mode = False

                # Update preferences dict
                self.preferences = {
                    "stream_output": self.stream_output,
                    "multiline_input": self.multiline_input,
                    "command_prefix": self.command_prefix,
                    "show_timestamps": self.show_timestamps,
                    "debug_mode": self.debug_mode,
                }

                console.print("[bold green]Settings reset to defaults[/]")

            else:
                error_console.print(
                    f"[bold red]Error:[/] Unknown settings subcommand: {subcommand}"
                )
                console.print("Available subcommands: set <name> <value>, reset")

    async def _cmd_multiline(self, args: List[str]) -> None:
        """
        Toggle multiline input mode.

        Args:
            args: Command arguments
        """
        self.multiline_input = not self.multiline_input
        console.print(
            f"[bold green]Multiline input mode: {'enabled' if self.multiline_input else 'disabled'}[/]"
        )
        self.preferences["multiline_input"] = self.multiline_input

    async def _cmd_debug(self, args: List[str]) -> None:
        """
        Toggle debug mode.

        Args:
            args: Command arguments
        """
        self.debug_mode = not self.debug_mode
        console.print(f"[bold green]Debug mode: {'enabled' if self.debug_mode else 'disabled'}[/]")
        self.preferences["debug_mode"] = self.debug_mode

    async def _cmd_info(self, args: List[str]) -> None:
        """
        Show session information.

        Args:
            args: Command arguments
        """
        console.print("\n[bold blue]Session Information:[/]")

        # Get session details
        session = await self.session_manager.get_session(self.session_id)

        if not session:
            error_console.print(f"[bold red]Error:[/] Session {self.session_id} not found")
            return

        # Basic session info
        info_table = Table(show_header=True)
        info_table.add_column("Property")
        info_table.add_column("Value")

        info_table.add_row("Session ID", session.session_id)
        info_table.add_row("User ID", session.user_id or "anonymous")
        info_table.add_row("Created At", session.created_at)
        info_table.add_row("Last Activity", session.last_activity)
        info_table.add_row("Status", session.status)
        info_table.add_row("Message Count", str(len(self.messages)))

        console.print(info_table)

        # Show metadata if available
        if session.metadata:
            console.print("\n[bold blue]Session Metadata:[/]")

            metadata_table = Table(show_header=True)
            metadata_table.add_column("Key")
            metadata_table.add_column("Value")

            for key, value in session.metadata.items():
                metadata_table.add_row(key, str(value))

            console.print(metadata_table)

        # Show system info
        console.print("\n[bold blue]System Information:[/]")

        system_table = Table(show_header=True)
        system_table.add_column("Property")
        system_table.add_column("Value")

        system_table.add_row("Python Version", sys.version.split()[0])
        system_table.add_row("Platform", sys.platform)
        system_table.add_row("Terminal Size", f"{console.width}x{console.height}")
        system_table.add_row(
            "Current Time (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        )

        console.print(system_table)


# Command execution helper function
async def run_interactive(
    container: Container, session_id: Optional[str] = None, user_id: Optional[str] = None
) -> None:
    """
    Run an interactive session.

    Args:
        container: Dependency injection container
        session_id: Optional session ID
        user_id: Optional user ID
    """
    # Create interactive session
    session = InteractiveSession(container=container, session_id=session_id, user_id=user_id)

    # Run the session
    await session.run()
