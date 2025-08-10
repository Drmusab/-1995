"""
Advanced Calendar API Integration for AI Assistant
Author: Drmusab
Last Modified: 2025-01-20 02:20:24 UTC

This module provides comprehensive calendar integration for the AI assistant,
supporting Google Calendar and other calendar services with full CRUD operations,
recurring events, timezone handling, and seamless integration with the core system.
"""

import base64
import hashlib
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Union
from urllib.parse import quote, urlencode

import aiohttp
import asyncio
from dateutil.parser import parse as parse_date
from dateutil.rrule import DAILY, MONTHLY, WEEKLY, YEARLY, rrule

# Assistant components
from src.assistant.core import ComponentManager
from src.assistant.core import SessionManager
from src.assistant.core import WorkflowOrchestrator

# Core imports
from src.core.config.loader import ConfigLoader
from src.core.dependency_injection import Container
from src.core.error_handling import ErrorHandler, handle_exceptions
from src.core.events.event_bus import EventBus
from src.core.events.event_types import (
    CalendarAuthRequired,
    CalendarConnected,
    CalendarError,
    CalendarEventCreated,
    ComponentHealthChanged,
)
from src.core.health_check import HealthCheck
from src.core.security.authentication import AuthenticationManager
from src.core.security.authorization import AuthorizationManager
from src.core.security.encryption import EncryptionManager
from src.integrations.cache.cache_strategy import CacheStrategy
from src.integrations.cache.redis_cache import RedisCache

# Memory and caching
from src.memory.core_memory.memory_manager import MemoryManager
from src.memory.operations.context_manager import ContextManager
from src.observability.logging.config import get_logger

# Observability
from src.observability.monitoring.metrics import MetricsCollector
from src.observability.monitoring.tracing import TraceManager

# Skills integration
from src.skills.skill_registry import SkillRegistry


class CalendarProvider(Enum):
    """Supported calendar providers."""

    GOOGLE = "google"
    OUTLOOK = "outlook"
    APPLE = "apple"
    CALDAV = "caldav"
    EXCHANGE = "exchange"


class EventStatus(Enum):
    """Calendar event status."""

    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventVisibility(Enum):
    """Calendar event visibility."""

    DEFAULT = "default"
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class RecurrenceFrequency(Enum):
    """Recurrence frequency types."""

    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class CalendarPermission(Enum):
    """Calendar access permissions."""

    READ = "read"
    WRITE = "write"
    OWNER = "owner"
    FREE_BUSY = "freeBusy"


@dataclass
class CalendarRecurrence:
    """Calendar event recurrence definition."""

    frequency: RecurrenceFrequency
    interval: int = 1
    count: Optional[int] = None
    until: Optional[datetime] = None
    by_day: List[str] = field(default_factory=list)  # MO, TU, WE, etc.
    by_month_day: List[int] = field(default_factory=list)
    by_month: List[int] = field(default_factory=list)
    timezone: str = "UTC"


@dataclass
class CalendarAttendee:
    """Calendar event attendee."""

    email: str
    name: Optional[str] = None
    status: str = "needsAction"  # needsAction, declined, tentative, accepted
    is_organizer: bool = False
    is_optional: bool = False
    comment: Optional[str] = None


@dataclass
class CalendarReminder:
    """Calendar event reminder."""

    method: str = "email"  # email, popup, sms
    minutes_before: int = 15


@dataclass
class CalendarEvent:
    """Comprehensive calendar event representation."""

    id: Optional[str] = None
    calendar_id: str = ""
    title: str = ""
    description: Optional[str] = None
    location: Optional[str] = None

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    timezone: str = "UTC"
    all_day: bool = False

    # Recurrence
    recurrence: Optional[CalendarRecurrence] = None
    recurring_event_id: Optional[str] = None

    # Attendees and organization
    organizer: Optional[CalendarAttendee] = None
    attendees: List[CalendarAttendee] = field(default_factory=list)

    # Properties
    status: EventStatus = EventStatus.CONFIRMED
    visibility: EventVisibility = EventVisibility.DEFAULT

    # Reminders and notifications
    reminders: List[CalendarReminder] = field(default_factory=list)

    # Meeting details
    meeting_url: Optional[str] = None
    meeting_id: Optional[str] = None
    conference_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    creator_email: Optional[str] = None

    # Custom fields
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    # Provider specific
    provider: CalendarProvider = CalendarProvider.GOOGLE
    provider_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalendarInfo:
    """Calendar metadata."""

    id: str
    name: str
    description: Optional[str] = None
    timezone: str = "UTC"
    color: Optional[str] = None
    is_primary: bool = False
    access_role: CalendarPermission = CalendarPermission.READ
    provider: CalendarProvider = CalendarProvider.GOOGLE
    owner_email: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class FreeBusyInfo:
    """Free/busy time information."""

    calendar_id: str
    busy_times: List[Dict[str, datetime]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timezone: str = "UTC"


@dataclass
class CalendarAuthConfig:
    """Calendar authentication configuration."""

    provider: CalendarProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: List[str] = field(default_factory=list)
    credentials_file: Optional[str] = None
    token_file: Optional[str] = None


class CalendarError(Exception):
    """Custom exception for calendar operations."""

    def __init__(
        self,
        message: str,
        provider: Optional[CalendarProvider] = None,
        calendar_id: Optional[str] = None,
        event_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.calendar_id = calendar_id
        self.event_id = event_id
        self.error_code = error_code
        self.timestamp = datetime.now(timezone.utc)


class CalendarProviderInterface(ABC):
    """Abstract interface for calendar providers."""

    @abstractmethod
    async def authenticate(
        self, auth_config: CalendarAuthConfig, user_credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Authenticate with the calendar provider."""
        pass

    @abstractmethod
    async def get_calendars(self, auth_token: str) -> List[CalendarInfo]:
        """Get list of user's calendars."""
        pass

    @abstractmethod
    async def get_events(
        self,
        auth_token: str,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        max_results: Optional[int] = None,
    ) -> List[CalendarEvent]:
        """Get events from a calendar."""
        pass

    @abstractmethod
    async def create_event(
        self, auth_token: str, calendar_id: str, event: CalendarEvent
    ) -> CalendarEvent:
        """Create a new calendar event."""
        pass

    @abstractmethod
    async def update_event(
        self, auth_token: str, calendar_id: str, event: CalendarEvent
    ) -> CalendarEvent:
        """Update an existing calendar event."""
        pass

    @abstractmethod
    async def delete_event(self, auth_token: str, calendar_id: str, event_id: str) -> bool:
        """Delete a calendar event."""
        pass

    @abstractmethod
    async def get_free_busy(
        self, auth_token: str, calendar_ids: List[str], start_time: datetime, end_time: datetime
    ) -> List[FreeBusyInfo]:
        """Get free/busy information for calendars."""
        pass


class GoogleCalendarProvider(CalendarProviderInterface):
    """Google Calendar API provider implementation."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.base_url = "https://www.googleapis.com/calendar/v3"
        self.auth_url = "https://accounts.google.com/o/oauth2/v2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"

        # Default scopes for Google Calendar
        self.default_scopes = [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/calendar.events",
        ]

    async def authenticate(
        self, auth_config: CalendarAuthConfig, user_credentials: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Authenticate with Google Calendar API."""
        try:
            if user_credentials and "access_token" in user_credentials:
                # Validate existing token
                is_valid = await self._validate_token(user_credentials["access_token"])
                if is_valid:
                    return user_credentials
                elif "refresh_token" in user_credentials:
                    # Refresh the token
                    return await self._refresh_token(auth_config, user_credentials["refresh_token"])

            # Generate new authorization URL
            auth_params = {
                "client_id": auth_config.client_id,
                "redirect_uri": auth_config.redirect_uri,
                "scope": " ".join(auth_config.scopes or self.default_scopes),
                "response_type": "code",
                "access_type": "offline",
                "prompt": "consent",
            }

            auth_url = f"{self.auth_url}?{urlencode(auth_params)}"

            return {"auth_url": auth_url, "requires_user_consent": True}

        except Exception as e:
            raise CalendarError(
                f"Google Calendar authentication failed: {str(e)}", CalendarProvider.GOOGLE
            )

    async def exchange_code_for_token(
        self, auth_config: CalendarAuthConfig, authorization_code: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            token_data = {
                "client_id": auth_config.client_id,
                "client_secret": auth_config.client_secret,
                "code": authorization_code,
                "grant_type": "authorization_code",
                "redirect_uri": auth_config.redirect_uri,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_url, data=token_data) as response:
                    if response.status == 200:
                        token_response = await response.json()
                        return {
                            "access_token": token_response["access_token"],
                            "refresh_token": token_response.get("refresh_token"),
                            "expires_in": token_response.get("expires_in", 3600),
                            "token_type": token_response.get("token_type", "Bearer"),
                        }
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Token exchange failed: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to exchange code for token: {str(e)}", CalendarProvider.GOOGLE
            )

    async def _validate_token(self, access_token: str) -> bool:
        """Validate Google access token."""
        try:
            validation_url = (
                f"https://www.googleapis.com/oauth2/v1/tokeninfo?access_token={access_token}"
            )

            async with aiohttp.ClientSession() as session:
                async with session.get(validation_url) as response:
                    return response.status == 200

        except Exception:
            return False

    async def _refresh_token(
        self, auth_config: CalendarAuthConfig, refresh_token: str
    ) -> Dict[str, Any]:
        """Refresh Google access token."""
        try:
            refresh_data = {
                "client_id": auth_config.client_id,
                "client_secret": auth_config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_url, data=refresh_data) as response:
                    if response.status == 200:
                        token_response = await response.json()
                        return {
                            "access_token": token_response["access_token"],
                            "refresh_token": refresh_token,  # Keep the original refresh token
                            "expires_in": token_response.get("expires_in", 3600),
                            "token_type": token_response.get("token_type", "Bearer"),
                        }
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Token refresh failed: {error_data}")

        except Exception as e:
            raise CalendarError(f"Failed to refresh token: {str(e)}", CalendarProvider.GOOGLE)

    async def get_calendars(self, auth_token: str) -> List[CalendarInfo]:
        """Get Google calendars."""
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            url = f"{self.base_url}/users/me/calendarList"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [self._parse_google_calendar(cal) for cal in data.get("items", [])]
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Failed to get calendars: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to get Google calendars: {str(e)}", CalendarProvider.GOOGLE
            )

    def _parse_google_calendar(self, cal_data: Dict[str, Any]) -> CalendarInfo:
        """Parse Google calendar data to CalendarInfo."""
        return CalendarInfo(
            id=cal_data["id"],
            name=cal_data.get("summary", ""),
            description=cal_data.get("description"),
            timezone=cal_data.get("timeZone", "UTC"),
            color=cal_data.get("backgroundColor"),
            is_primary=cal_data.get("primary", False),
            access_role=CalendarPermission(cal_data.get("accessRole", "read")),
            provider=CalendarProvider.GOOGLE,
        )

    async def get_events(
        self,
        auth_token: str,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        max_results: Optional[int] = None,
    ) -> List[CalendarEvent]:
        """Get Google Calendar events."""
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}

            params = {
                "timeMin": start_time.isoformat(),
                "timeMax": end_time.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
            }

            if max_results:
                params["maxResults"] = str(max_results)

            url = f"{self.base_url}/calendars/{quote(calendar_id)}/events"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            self._parse_google_event(event, calendar_id)
                            for event in data.get("items", [])
                        ]
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Failed to get events: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to get Google Calendar events: {str(e)}",
                CalendarProvider.GOOGLE,
                calendar_id,
            )

    def _parse_google_event(self, event_data: Dict[str, Any], calendar_id: str) -> CalendarEvent:
        """Parse Google event data to CalendarEvent."""
        # Parse start/end times
        start_data = event_data.get("start", {})
        end_data = event_data.get("end", {})

        all_day = "date" in start_data

        if all_day:
            start_time = parse_date(start_data["date"]).replace(tzinfo=timezone.utc)
            end_time = parse_date(end_data["date"]).replace(tzinfo=timezone.utc)
            event_timezone = "UTC"
        else:
            start_time = parse_date(start_data["dateTime"])
            end_time = parse_date(end_data["dateTime"])
            event_timezone = start_data.get("timeZone", "UTC")

        # Parse attendees
        attendees = []
        for attendee_data in event_data.get("attendees", []):
            attendees.append(
                CalendarAttendee(
                    email=attendee_data["email"],
                    name=attendee_data.get("displayName"),
                    status=attendee_data.get("responseStatus", "needsAction"),
                    is_organizer=attendee_data.get("organizer", False),
                    is_optional=attendee_data.get("optional", False),
                )
            )

        # Parse organizer
        organizer_data = event_data.get("organizer", {})
        organizer = None
        if organizer_data:
            organizer = CalendarAttendee(
                email=organizer_data["email"],
                name=organizer_data.get("displayName"),
                is_organizer=True,
            )

        # Parse reminders
        reminders = []
        reminder_data = event_data.get("reminders", {})
        if reminder_data.get("useDefault"):
            reminders.append(CalendarReminder())
        else:
            for override in reminder_data.get("overrides", []):
                reminders.append(
                    CalendarReminder(method=override["method"], minutes_before=override["minutes"])
                )

        # Parse recurrence
        recurrence = None
        if "recurrence" in event_data:
            recurrence = self._parse_google_recurrence(event_data["recurrence"])

        return CalendarEvent(
            id=event_data["id"],
            calendar_id=calendar_id,
            title=event_data.get("summary", ""),
            description=event_data.get("description"),
            location=event_data.get("location"),
            start_time=start_time,
            end_time=end_time,
            timezone=event_timezone,
            all_day=all_day,
            recurrence=recurrence,
            organizer=organizer,
            attendees=attendees,
            status=EventStatus(event_data.get("status", "confirmed")),
            visibility=EventVisibility(event_data.get("visibility", "default")),
            reminders=reminders,
            meeting_url=event_data.get("hangoutLink"),
            created_at=parse_date(event_data["created"]) if "created" in event_data else None,
            updated_at=parse_date(event_data["updated"]) if "updated" in event_data else None,
            creator_email=event_data.get("creator", {}).get("email"),
            provider=CalendarProvider.GOOGLE,
            provider_data=event_data,
        )

    def _parse_google_recurrence(self, recurrence_rules: List[str]) -> Optional[CalendarRecurrence]:
        """Parse Google Calendar recurrence rules."""
        try:
            # Google uses RRULE format
            for rule in recurrence_rules:
                if rule.startswith("RRULE:"):
                    rrule_str = rule[6:]  # Remove "RRULE:" prefix

                    # Parse basic frequency
                    freq_map = {
                        "DAILY": RecurrenceFrequency.DAILY,
                        "WEEKLY": RecurrenceFrequency.WEEKLY,
                        "MONTHLY": RecurrenceFrequency.MONTHLY,
                        "YEARLY": RecurrenceFrequency.YEARLY,
                    }

                    frequency = RecurrenceFrequency.NONE
                    interval = 1
                    count = None
                    until = None
                    by_day = []

                    # Parse RRULE components
                    for component in rrule_str.split(";"):
                        if "=" in component:
                            key, value = component.split("=", 1)

                            if key == "FREQ":
                                frequency = freq_map.get(value, RecurrenceFrequency.NONE)
                            elif key == "INTERVAL":
                                interval = int(value)
                            elif key == "COUNT":
                                count = int(value)
                            elif key == "UNTIL":
                                until = parse_date(value)
                            elif key == "BYDAY":
                                by_day = value.split(",")

                    return CalendarRecurrence(
                        frequency=frequency,
                        interval=interval,
                        count=count,
                        until=until,
                        by_day=by_day,
                    )

            return None

        except Exception as e:
            self.logger.warning(f"Failed to parse Google recurrence: {str(e)}")
            return None

    async def create_event(
        self, auth_token: str, calendar_id: str, event: CalendarEvent
    ) -> CalendarEvent:
        """Create Google Calendar event."""
        try:
            headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

            event_data = self._format_google_event(event)
            url = f"{self.base_url}/calendars/{quote(calendar_id)}/events"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=event_data) as response:
                    if response.status == 200:
                        created_event = await response.json()
                        return self._parse_google_event(created_event, calendar_id)
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Failed to create event: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to create Google Calendar event: {str(e)}",
                CalendarProvider.GOOGLE,
                calendar_id,
            )

    def _format_google_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Format CalendarEvent for Google Calendar API."""
        event_data = {
            "summary": event.title,
            "description": event.description,
            "location": event.location,
            "status": event.status.value,
            "visibility": event.visibility.value,
        }

        # Format start/end times
        if event.all_day:
            event_data["start"] = {"date": event.start_time.strftime("%Y-%m-%d")}
            event_data["end"] = {"date": event.end_time.strftime("%Y-%m-%d")}
        else:
            event_data["start"] = {
                "dateTime": event.start_time.isoformat(),
                "timeZone": event.timezone,
            }
            event_data["end"] = {"dateTime": event.end_time.isoformat(), "timeZone": event.timezone}

        # Format attendees
        if event.attendees:
            event_data["attendees"] = [
                {
                    "email": attendee.email,
                    "displayName": attendee.name,
                    "optional": attendee.is_optional,
                }
                for attendee in event.attendees
            ]

        # Format reminders
        if event.reminders:
            event_data["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {"method": reminder.method, "minutes": reminder.minutes_before}
                    for reminder in event.reminders
                ],
            }

        # Format recurrence
        if event.recurrence and event.recurrence.frequency != RecurrenceFrequency.NONE:
            rrule = self._format_google_recurrence(event.recurrence)
            if rrule:
                event_data["recurrence"] = [rrule]

        return event_data

    def _format_google_recurrence(self, recurrence: CalendarRecurrence) -> str:
        """Format CalendarRecurrence for Google Calendar RRULE."""
        freq_map = {
            RecurrenceFrequency.DAILY: "DAILY",
            RecurrenceFrequency.WEEKLY: "WEEKLY",
            RecurrenceFrequency.MONTHLY: "MONTHLY",
            RecurrenceFrequency.YEARLY: "YEARLY",
        }

        rrule_parts = [f"FREQ={freq_map[recurrence.frequency]}"]

        if recurrence.interval > 1:
            rrule_parts.append(f"INTERVAL={recurrence.interval}")

        if recurrence.count:
            rrule_parts.append(f"COUNT={recurrence.count}")
        elif recurrence.until:
            rrule_parts.append(f"UNTIL={recurrence.until.strftime('%Y%m%dT%H%M%SZ')}")

        if recurrence.by_day:
            rrule_parts.append(f"BYDAY={','.join(recurrence.by_day)}")

        return f"RRULE:{';'.join(rrule_parts)}"

    async def update_event(
        self, auth_token: str, calendar_id: str, event: CalendarEvent
    ) -> CalendarEvent:
        """Update Google Calendar event."""
        try:
            headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

            event_data = self._format_google_event(event)
            url = f"{self.base_url}/calendars/{quote(calendar_id)}/events/{event.id}"

            async with aiohttp.ClientSession() as session:
                async with session.put(url, headers=headers, json=event_data) as response:
                    if response.status == 200:
                        updated_event = await response.json()
                        return self._parse_google_event(updated_event, calendar_id)
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Failed to update event: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to update Google Calendar event: {str(e)}",
                CalendarProvider.GOOGLE,
                calendar_id,
                event.id,
            )

    async def delete_event(self, auth_token: str, calendar_id: str, event_id: str) -> bool:
        """Delete Google Calendar event."""
        try:
            headers = {"Authorization": f"Bearer {auth_token}"}
            url = f"{self.base_url}/calendars/{quote(calendar_id)}/events/{event_id}"

            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status == 204:
                        return True
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Failed to delete event: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to delete Google Calendar event: {str(e)}",
                CalendarProvider.GOOGLE,
                calendar_id,
                event_id,
            )

    async def get_free_busy(
        self, auth_token: str, calendar_ids: List[str], start_time: datetime, end_time: datetime
    ) -> List[FreeBusyInfo]:
        """Get Google Calendar free/busy information."""
        try:
            headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

            request_data = {
                "timeMin": start_time.isoformat(),
                "timeMax": end_time.isoformat(),
                "items": [{"id": cal_id} for cal_id in calendar_ids],
            }

            url = f"{self.base_url}/freeBusy"

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=request_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_free_busy(data)
                    else:
                        error_data = await response.json()
                        raise CalendarError(f"Failed to get free/busy: {error_data}")

        except Exception as e:
            raise CalendarError(
                f"Failed to get Google Calendar free/busy: {str(e)}", CalendarProvider.GOOGLE
            )

    def _parse_google_free_busy(self, data: Dict[str, Any]) -> List[FreeBusyInfo]:
        """Parse Google free/busy response."""
        result = []

        for calendar_id, cal_data in data.get("calendars", {}).items():
            busy_times = []

            for busy_period in cal_data.get("busy", []):
                busy_times.append(
                    {
                        "start": parse_date(busy_period["start"]),
                        "end": parse_date(busy_period["end"]),
                    }
                )

            errors = cal_data.get("errors", [])
            error_messages = [error.get("reason", str(error)) for error in errors]

            result.append(
                FreeBusyInfo(calendar_id=calendar_id, busy_times=busy_times, errors=error_messages)
            )

        return result


class CalendarCache:
    """Caching layer for calendar data."""

    def __init__(self, cache_strategy: CacheStrategy):
        self.cache = cache_strategy
        self.logger = get_logger(__name__)

        # Cache TTL settings
        self.calendar_list_ttl = 3600  # 1 hour
        self.event_ttl = 300  # 5 minutes
        self.free_busy_ttl = 60  # 1 minute

    async def get_calendars(
        self, user_id: str, provider: CalendarProvider
    ) -> Optional[List[CalendarInfo]]:
        """Get cached calendar list."""
        cache_key = f"calendars:{provider.value}:{user_id}"
        return await self.cache.get(cache_key)

    async def set_calendars(
        self, user_id: str, provider: CalendarProvider, calendars: List[CalendarInfo]
    ) -> None:
        """Cache calendar list."""
        cache_key = f"calendars:{provider.value}:{user_id}"
        await self.cache.set(cache_key, calendars, ttl=self.calendar_list_ttl)

    async def get_events(
        self, user_id: str, calendar_id: str, start_time: datetime, end_time: datetime
    ) -> Optional[List[CalendarEvent]]:
        """Get cached events."""
        time_key = f"{start_time.isoformat()}:{end_time.isoformat()}"
        cache_key = f"events:{user_id}:{calendar_id}:{time_key}"
        return await self.cache.get(cache_key)

    async def set_events(
        self,
        user_id: str,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        events: List[CalendarEvent],
    ) -> None:
        """Cache events."""
        time_key = f"{start_time.isoformat()}:{end_time.isoformat()}"
        cache_key = f"events:{user_id}:{calendar_id}:{time_key}"
        await self.cache.set(cache_key, events, ttl=self.event_ttl)

    async def invalidate_calendar(self, user_id: str, calendar_id: str) -> None:
        """Invalidate all cache entries for a calendar."""
        pattern = f"events:{user_id}:{calendar_id}:*"
        await self.cache.delete_pattern(pattern)

    async def invalidate_user(self, user_id: str) -> None:
        """Invalidate all cache entries for a user."""
        patterns = [f"calendars:*:{user_id}", f"events:{user_id}:*", f"free_busy:{user_id}:*"]

        for pattern in patterns:
            await self.cache.delete_pattern(pattern)


class EnhancedCalendarAPI:
    """
    Advanced Calendar API Integration for the AI Assistant.

    This component provides comprehensive calendar functionality including:
    - Multi-provider support (Google, Outlook, Apple, CalDAV)
    - Full CRUD operations for events and calendars
    - Advanced recurrence pattern support
    - Free/busy time queries and scheduling assistance
    - Timezone handling and conversion
    - Caching and performance optimization
    - Integration with session management
    - Event-driven notifications
    - Security and authentication
    - Memory system integration for context
    - Workflow orchestrator integration
    - Skills system integration for calendar actions
    """

    def __init__(self, container: Container):
        """
        Initialize the enhanced calendar API.

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

        # Security components
        try:
            self.auth_manager = container.get(AuthenticationManager)
            self.authz_manager = container.get(AuthorizationManager)
            self.encryption_manager = container.get(EncryptionManager)
        except Exception:
            self.auth_manager = None
            self.authz_manager = None
            self.encryption_manager = None

        # Assistant components
        self.session_manager = container.get(SessionManager)
        self.component_manager = container.get(ComponentManager)
        self.workflow_orchestrator = container.get(WorkflowOrchestrator)

        # Memory and context
        self.memory_manager = container.get(MemoryManager)
        self.context_manager = container.get(ContextManager)

        # Caching
        try:
            cache_strategy = container.get(CacheStrategy)
            self.cache = CalendarCache(cache_strategy)
        except Exception:
            self.cache = None

        # Skills integration
        self.skill_registry = container.get(SkillRegistry)

        # Observability
        self.metrics = container.get(MetricsCollector)
        self.tracer = container.get(TraceManager)

        # Calendar providers
        self.providers: Dict[CalendarProvider, CalendarProviderInterface] = {
            CalendarProvider.GOOGLE: GoogleCalendarProvider()
        }

        # User credentials storage
        self.user_credentials: Dict[str, Dict[CalendarProvider, Dict[str, Any]]] = {}
        self.credential_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Configuration
        self.auth_configs = self._load_auth_configs()
        self.default_timezone = self.config.get("calendar.default_timezone", "UTC")
        self.max_events_per_query = self.config.get("calendar.max_events_per_query", 2500)
        self.enable_caching = self.config.get("calendar.enable_caching", True)
        self.cache_ttl = self.config.get("calendar.cache_ttl", 300)

        # Performance tracking
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Setup monitoring
        self._setup_monitoring()

        # Register health check
        self.health_check.register_component("calendar_api", self._health_check_callback)

        self.logger.info("EnhancedCalendarAPI initialized successfully")

    def _load_auth_configs(self) -> Dict[CalendarProvider, CalendarAuthConfig]:
        """Load authentication configurations for calendar providers."""
        auth_configs = {}

        # Google Calendar configuration
        google_config = self.config.get("calendar.providers.google", {})
        if google_config:
            auth_configs[CalendarProvider.GOOGLE] = CalendarAuthConfig(
                provider=CalendarProvider.GOOGLE,
                client_id=google_config.get("client_id", ""),
                client_secret=google_config.get("client_secret", ""),
                redirect_uri=google_config.get("redirect_uri", ""),
                scopes=google_config.get(
                    "scopes",
                    [
                        "https://www.googleapis.com/auth/calendar",
                        "https://www.googleapis.com/auth/calendar.events",
                    ],
                ),
            )

        return auth_configs

    def _setup_monitoring(self) -> None:
        """Setup monitoring and metrics."""
        try:
            # Register calendar metrics
            self.metrics.register_counter("calendar_operations_total")
            self.metrics.register_counter("calendar_operations_successful")
            self.metrics.register_counter("calendar_operations_failed")
            self.metrics.register_histogram("calendar_operation_duration_seconds")
            self.metrics.register_counter("calendar_events_created")
            self.metrics.register_counter("calendar_events_updated")
            self.metrics.register_counter("calendar_events_deleted")
            self.metrics.register_gauge("calendar_active_connections")
            self.metrics.register_counter("calendar_auth_requests")
            self.metrics.register_counter("calendar_cache_hits")
            self.metrics.register_counter("calendar_cache_misses")

        except Exception as e:
            self.logger.warning(f"Failed to setup monitoring: {str(e)}")

    async def initialize(self) -> None:
        """Initialize the calendar API."""
        try:
            # Register calendar skills
            await self._register_calendar_skills()

            # Register event handlers
            await self._register_event_handlers()

            # Register with component manager
            await self._register_as_component()

            self.logger.info("CalendarAPI initialization completed")

        except Exception as e:
            self.logger.error(f"Failed to initialize CalendarAPI: {str(e)}")
            raise CalendarError(f"Initialization failed: {str(e)}")

    async def _register_calendar_skills(self) -> None:
        """Register calendar-related skills."""
        try:
            # Calendar skills would be registered here
            # This integrates with the skills system
            calendar_skills = [
                "create_event",
                "update_event",
                "delete_event",
                "get_events",
                "check_availability",
                "schedule_meeting",
                "find_free_time",
            ]

            for skill_name in calendar_skills:
                # Register skills with the skill registry
                # Implementation would depend on skill system structure
                pass

        except Exception as e:
            self.logger.warning(f"Failed to register calendar skills: {str(e)}")

    async def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        # User session events
        self.event_bus.subscribe("user_authenticated", self._handle_user_authentication)
        self.event_bus.subscribe("user_logged_out", self._handle_user_logout)
        self.event_bus.subscribe("session_ended", self._handle_session_ended)

    async def _register_as_component(self) -> None:
        """Register calendar API as a system component."""
        try:
            # Register with component manager for health monitoring
            # Implementation depends on component manager structure
            pass
        except Exception as e:
            self.logger.warning(f"Failed to register as component: {str(e)}")

    @handle_exceptions
    async def authenticate_user(
        self,
        user_id: str,
        provider: CalendarProvider,
        authorization_code: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate user with calendar provider.

        Args:
            user_id: User identifier
            provider: Calendar provider
            authorization_code: OAuth authorization code (if completing OAuth flow)
            session_id: Optional session ID

        Returns:
            Authentication result
        """
        start_time = time.time()

        try:
            with self.tracer.trace("calendar_authentication") as span:
                span.set_attributes(
                    {
                        "user_id": user_id,
                        "provider": provider.value,
                        "has_auth_code": authorization_code is not None,
                    }
                )

                if provider not in self.auth_configs:
                    raise CalendarError(f"Provider {provider.value} not configured")

                auth_config = self.auth_configs[provider]
                provider_impl = self.providers[provider]

                async with self.credential_locks[user_id]:
                    # Check for existing credentials
                    existing_credentials = self.user_credentials.get(user_id, {}).get(provider)

                    if authorization_code:
                        # Complete OAuth flow
                        if provider == CalendarProvider.GOOGLE:
                            credentials = await provider_impl.exchange_code_for_token(
                                auth_config, authorization_code
                            )
                        else:
                            raise CalendarError(f"OAuth flow not implemented for {provider.value}")

                        # Store credentials securely
                        await self._store_user_credentials(user_id, provider, credentials)

                        # Emit authentication event
                        await self.event_bus.emit(
                            CalendarConnected(
                                user_id=user_id, provider=provider.value, session_id=session_id
                            )
                        )

                        self.metrics.increment("calendar_auth_requests")

                        return {"success": True, "provider": provider.value, "authenticated": True}

                    else:
                        # Start authentication flow
                        auth_result = await provider_impl.authenticate(
                            auth_config, existing_credentials
                        )

                        if auth_result.get("requires_user_consent"):
                            # Emit auth required event
                            await self.event_bus.emit(
                                CalendarAuthRequired(
                                    user_id=user_id,
                                    provider=provider.value,
                                    auth_url=auth_result["auth_url"],
                                    session_id=session_id,
                                )
                            )

                        return auth_result

        except Exception as e:
            self.error_counts["authentication"] += 1
            self.metrics.increment("calendar_operations_failed")

            await self.event_bus.emit(
                CalendarError(
                    user_id=user_id,
                    provider=provider.value,
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            )

            self.logger.error(f"Calendar authentication failed for user {user_id}: {str(e)}")
            raise

        finally:
            operation_time = time.time() - start_time
            self.metrics.record("calendar_operation_duration_seconds", operation_time)

    async def _store_user_credentials(
        self, user_id: str, provider: CalendarProvider, credentials: Dict[str, Any]
    ) -> None:
        """Store user credentials securely."""
        try:
            # Encrypt credentials if encryption is available
            if self.encryption_manager:
                encrypted_credentials = await self.encryption_manager.encrypt(
                    json.dumps(credentials)
                )
                credentials_to_store = {"encrypted": encrypted_credentials}
            else:
                credentials_to_store = credentials

            # Store in memory (in production, would use secure storage)
            if user_id not in self.user_credentials:
                self.user_credentials[user_id] = {}

            self.user_credentials[user_id][provider] = credentials_to_store

            # Store in session context if session is available
            if hasattr(self.context_manager, "update_user_context"):
                await self.context_manager.update_user_context(
                    user_id, {f"calendar_{provider.value}_authenticated": True}
                )

        except Exception as e:
            self.logger.error(f"Failed to store credentials for user {user_id}: {str(e)}")

    async def _get_user_credentials(
        self, user_id: str, provider: CalendarProvider
    ) -> Optional[Dict[str, Any]]:
        """Get user credentials securely."""
        try:
            credentials = self.user_credentials.get(user_id, {}).get(provider)

            if not credentials:
                return None

            # Decrypt if encrypted
            if "encrypted" in credentials and self.encryption_manager:
                decrypted_data = await self.encryption_manager.decrypt(credentials["encrypted"])
                return json.loads(decrypted_data)

            return credentials

        except Exception as e:
            self.logger.error(f"Failed to get credentials for user {user_id}: {str(e)}")
            return None

    @handle_exceptions
    async def get_calendars(
        self,
        user_id: str,
        provider: CalendarProvider = CalendarProvider.GOOGLE,
        session_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[CalendarInfo]:
        """
        Get user's calendars.

        Args:
            user_id: User identifier
            provider: Calendar provider
            session_id: Optional session ID
            use_cache: Whether to use cache

        Returns:
            List of calendar information
        """
        start_time = time.time()

        try:
            with self.tracer.trace("get_calendars") as span:
                span.set_attributes(
                    {"user_id": user_id, "provider": provider.value, "use_cache": use_cache}
                )

                # Check cache first
                if use_cache and self.cache:
                    cached_calendars = await self.cache.get_calendars(user_id, provider)
                    if cached_calendars:
                        self.metrics.increment("calendar_cache_hits")
                        return cached_calendars
                    self.metrics.increment("calendar_cache_misses")

                # Get credentials
                credentials = await self._get_user_credentials(user_id, provider)
                if not credentials:
                    raise CalendarError(
                        f"No credentials found for user {user_id} and provider {provider.value}"
                    )

                # Get calendars from provider
                provider_impl = self.providers[provider]
                calendars = await provider_impl.get_calendars(credentials["access_token"])

                # Cache results
                if self.cache and use_cache:
                    await self.cache.set_calendars(user_id, provider, calendars)

                # Store in memory for context
                await self._store_calendar_context(user_id, calendars, session_id)

                self.metrics.increment("calendar_operations_successful")

                return calendars

        except Exception as e:
            self.error_counts["get_calendars"] += 1
            self.metrics.increment("calendar_operations_failed")

            self.logger.error(f"Failed to get calendars for user {user_id}: {str(e)}")
            raise

        finally:
            operation_time = time.time() - start_time
            self.metrics.record("calendar_operation_duration_seconds", operation_time)

    async def _store_calendar_context(
        self, user_id: str, calendars: List[CalendarInfo], session_id: Optional[str]
    ) -> None:
        """Store calendar information in memory for context."""
        try:
            calendar_data = {
                "calendars": [
                    {
                        "id": cal.id,
                        "name": cal.name,
                        "is_primary": cal.is_primary,
                        "timezone": cal.timezone,
                    }
                    for cal in calendars
                ],
                "primary_calendar": next((cal.id for cal in calendars if cal.is_primary), None),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Store in memory manager
            await self.memory_manager.store_user_data(
                user_id=user_id,
                data_type="calendar_info",
                data=calendar_data,
                session_id=session_id,
            )

        except Exception as e:
            self.logger.warning(f"Failed to store calendar context: {str(e)}")

    @handle_exceptions
    async def get_events(
        self,
        user_id: str,
        calendar_id: str,
        start_time: datetime,
        end_time: datetime,
        provider: CalendarProvider = CalendarProvider.GOOGLE,
        max_results: Optional[int] = None,
        session_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[CalendarEvent]:
        """
        Get calendar events.

        Args:
            user_id: User identifier
            calendar_id: Calendar identifier
            start_time: Start time for query
            end_time: End time for query
            provider: Calendar provider
            max_results: Maximum number of events to return
            session_id: Optional session ID
            use_cache: Whether to use cache

        Returns:
            List of calendar events
        """
        start_query_time = time.time()

        try:
            with self.tracer.trace("get_events") as span:
                span.set_attributes(
                    {
                        "user_id": user_id,
                        "calendar_id": calendar_id,
                        "provider": provider.value,
                        "date_range_days": (end_time - start_time).days,
                    }
                )

                # Validate time range
                if end_time <= start_time:
                    raise CalendarError("End time must be after start time")

                # Limit query size
                max_results = min(
                    max_results or self.max_events_per_query, self.max_events_per_query
                )

                # Check cache first
                if use_cache and self.cache:
                    cached_events = await self.cache.get_events(
                        user_id, calendar_id, start_time, end_time
                    )
                    if cached_events:
                        self.metrics.increment("calendar_cache_hits")
                        return cached_events[:max_results] if max_results else cached_events
                    self.metrics.increment("calendar_cache_misses")

                # Get credentials
                credentials = await self._get_user_credentials(user_id, provider)
                if not credentials:
                    raise CalendarError(f"No credentials found for user {user_id}")

                # Get events from provider
                provider_impl = self.providers[provider]
                events = await provider_impl.get_events(
                    credentials["access_token"], calendar_id, start_time, end_time, max_results
                )

                # Cache results
                if self.cache and use_cache:
                    await self.cache.set_events(user_id, calendar_id, start_time, end_time, events)

                # Store in memory for context
                await self._store_events_context(user_id, calendar_id, events, session_id)

                self.metrics.increment("calendar_operations_successful")

                return events

        except Exception as e:
            self.error_counts["get_events"] += 1
            self.metrics.increment("calendar_operations_failed")

            self.logger.error(f"Failed to get events for user {user_id}: {str(e)}")
            raise

        finally:
            operation_time = time.time() - start_query_time
            self.metrics.record("calendar_operation_duration_seconds", operation_time)

    async def _store_events_context(
        self, user_id: str, calendar_id: str, events: List[CalendarEvent], session_id: Optional[str]
    ) -> None:
        """Store events in memory for context."""
        try:
            # Create summary for context
            events_summary = {
                "calendar_id": calendar_id,
                "event_count": len(events),
                "upcoming_events": [
                    {
                        "id": event.id,
                        "title": event.title,
                        "start_time": event.start_time.isoformat() if event.start_time else None,
                        "end_time": event.end_time.isoformat() if event.end_time else None,
                        "all_day": event.all_day,
                    }
                    for event in events[:5]  # Store only first 5 for context
                ],
                "query_time": datetime.now(timezone.utc).isoformat(),
            }

            # Store in working memory for session
            if session_id:
                await self.memory_manager.store_session_data(
                    session_id=session_id, data_type="recent_calendar_query", data=events_summary
                )

        except Exception as e:
            self.logger.warning(f"Failed to store events context: {str(e)}")

    @handle_exceptions
    async def create_event(
        self,
        user_id: str,
        calendar_id: str,
        event: CalendarEvent,
        provider: CalendarProvider = CalendarProvider.GOOGLE,
        session_id: Optional[str] = None,
    ) -> CalendarEvent:
        """
        Create a calendar event.

        Args:
            user_id: User identifier
            calendar_id: Calendar identifier
            event: Event to create
            provider: Calendar provider
            session_id: Optional session ID

        Returns:
            Created event
        """
        start_time = time.time()

        try:
            with self.tracer.trace("create_event") as span:
                span.set_attributes(
                    {
                        "user_id": user_id,
                        "calendar_id": calendar_id,
                        "provider": provider.value,
                        "event_title": event.title,
                        "has_attendees": len(event.attendees) > 0,
                    }
                )

                # Validate event
                self._validate_event(event)

                # Get credentials
                credentials = await self._get_user_credentials(user_id, provider)
                if not credentials:
                    raise CalendarError(f"No credentials found for user {user_id}")

                # Set calendar ID
                event.calendar_id = calendar_id

                # Create event with provider
                provider_impl = self.providers[provider]
                created_event = await provider_impl.create_event(
                    credentials["access_token"], calendar_id, event
                )

                # Invalidate cache
                if self.cache:
                    await self.cache.invalidate_calendar(user_id, calendar_id)

                # Store in memory
                await self._store_event_operation(user_id, "created", created_event, session_id)

                # Emit event
                await self.event_bus.emit(
                    CalendarEventCreated(
                        user_id=user_id,
                        calendar_id=calendar_id,
                        event_id=created_event.id,
                        event_title=created_event.title,
                        session_id=session_id,
                    )
                )

                self.metrics.increment("calendar_events_created")
                self.metrics.increment("calendar_operations_successful")

                self.logger.info(f"Created event '{created_event.title}' for user {user_id}")

                return created_event

        except Exception as e:
            self.error_counts["create_event"] += 1
            self.metrics.increment("calendar_operations_failed")

            await self.event_bus.emit(
                CalendarError(
                    user_id=user_id,
                    calendar_id=calendar_id,
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            )

            self.logger.error(f"Failed to create event for user {user_id}: {str(e)}")
            raise

        finally:
            operation_time = time.time() - start_time
            self.metrics.record("calendar_operation_duration_seconds", operation_time)

    def _validate_event(self, event: CalendarEvent) -> None:
        """Validate calendar event data."""
        if not event.title:
            raise CalendarError("Event title is required")

        if not event.start_time:
            raise CalendarError("Event start time is required")

        if not event.end_time:
            raise CalendarError("Event end time is required")

        if event.end_time <= event.start_time:
            raise CalendarError("Event end time must be after start time")

        # Validate attendee emails
        for attendee in event.attendees:
            if not attendee.email or "@" not in attendee.email:
                raise CalendarError(f"Invalid attendee email: {attendee.email}")

    async def _store_event_operation(
        self, user_id: str, operation: str, event: CalendarEvent, session_id: Optional[str]
    ) -> None:
        """Store event operation in memory."""
        try:
            operation_data = {
                "operation": operation,
                "event_id": event.id,
                "event_title": event.title,
                "calendar_id": event.calendar_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Store in episodic memory
            await self.memory_manager.store_episodic_memory(
                user_id=user_id,
                event_type=f"calendar_event_{operation}",
                data=operation_data,
                session_id=session_id,
            )

        except Exception as e:
            self.logger.warning(f"Failed to store event operation: {str(e)}")

    @handle_exceptions
    async def update_event(
        self,
        user_id: str,
        calendar_id: str,
        event_id: str,
        event_data: Dict[str, Any],
        provider: CalendarProvider = CalendarProvider.GOOGLE,
    ) -> CalendarEvent:
        """Update an existing calendar event."""
        try:
            # Get provider instance
            provider_instance = self.providers.get(provider)
            if not provider_instance:
                raise CalendarAPIError(f"Provider {provider} not available")

            # Update event through provider
            event = await provider_instance.update_event(user_id, calendar_id, event_id, event_data)

            # Cache updated event
            cache_key = f"event:{provider.value}:{user_id}:{calendar_id}:{event_id}"
            await self._cache_event(cache_key, event)

            return event

        except Exception as e:
            self.logger.error(f"Failed to update event {event_id}: {str(e)}")
            raise CalendarAPIError(f"Update event failed: {str(e)}")
