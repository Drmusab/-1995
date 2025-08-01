"""WebSocket API Module"""

# Import the new broadcast module which should work
from .broadcast import BroadcastManager, BroadcastScope

# Try to import setup function but handle missing dependencies gracefully
try:
    from .websocket_setup import setup_websocket_api

    __all__ = ["setup_websocket_api", "BroadcastManager", "BroadcastScope"]
except ImportError:
    # websocket_setup has missing dependencies, only expose broadcast functionality
    __all__ = ["BroadcastManager", "BroadcastScope"]
