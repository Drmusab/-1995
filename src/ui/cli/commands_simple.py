"""
Simple command registry for CLI
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Command:
    """Simple command data class."""
    name: str
    description: str
    callback: Callable
    usage: Optional[str] = None
    aliases: Optional[List[str]] = None


class CommandRegistry:
    """Registry for CLI commands."""

    def __init__(self):
        """Initialize the command registry."""
        self.commands = {}
        self.aliases = {}

    def register(
        self,
        command_or_name,
        handler: Optional[Callable] = None,
        help_text: Optional[str] = None,
        group: str = "general",
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a command."""
        if isinstance(command_or_name, Command):
            # New style: Command object
            command = command_or_name
            name = command.name
            handler = command.callback
            help_text = command.description
            aliases = command.aliases or []
        else:
            # Old style: individual parameters
            name = command_or_name
            if handler is None or help_text is None:
                raise ValueError("handler and help_text are required when providing name as string")
            aliases = aliases or []
        
        self.commands[name] = {"handler": handler, "help": help_text, "group": group}

        # Register aliases
        if aliases:
            for alias in aliases:
                self.aliases[alias] = name

    def get_commands(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered commands."""
        return self.commands.copy()

    def get_command_by_name_or_alias(self, name: str) -> Optional[Dict[str, Any]]:
        """Get command info by name or alias."""
        # Check direct command name first
        if name in self.commands:
            cmd_info = self.commands[name].copy()
            cmd_info['name'] = name
            cmd_info['callback'] = cmd_info['handler']  # Add callback alias
            return cmd_info
        
        # Check aliases
        if name in self.aliases:
            command_name = self.aliases[name]
            cmd_info = self.commands[command_name].copy()
            cmd_info['name'] = command_name
            cmd_info['callback'] = cmd_info['handler']  # Add callback alias
            return cmd_info
        
        return None
