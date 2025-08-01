"""
Security interfaces for loose coupling.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional


class ISessionProvider(ABC):
    """Interface for session providers to avoid direct dependency on session manager."""

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID."""
        pass

    @abstractmethod
    async def validate_session(self, session_id: str) -> bool:
        """Validate if session is active and valid."""
        pass

    @abstractmethod
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        pass


class IUserProvider(ABC):
    """Interface for user providers."""

    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data by ID."""
        pass

    @abstractmethod
    async def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return user ID if successful."""
        pass
