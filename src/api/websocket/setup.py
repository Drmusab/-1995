"""WebSocket API Setup Module"""

from typing import Any, Optional, Dict
import asyncio
import json
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

from src.core.config.loader import ConfigLoader
from src.observability.logging.config import get_logger


class WebSocketServer:
    """WebSocket server for real-time communication."""
    
    def __init__(self, assistant: Any):
        """Initialize WebSocket server."""
        self.assistant = assistant
        self.logger = get_logger(__name__)
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.server = None
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connection."""
        connection_id = f"ws_{len(self.connections)}_{datetime.now().timestamp()}"
        self.connections[connection_id] = websocket
        
        self.logger.info(f"WebSocket connection established: {connection_id}")
        
        try:
            await websocket.send(json.dumps({
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat()
            }))
            
            async for message in websocket:
                await self.handle_message(connection_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
        finally:
            if connection_id in self.connections:
                del self.connections[connection_id]
    
    async def handle_message(self, connection_id: str, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            if message_type == "ping":
                await self.send_to_connection(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            elif message_type == "status_request":
                status = self.assistant.get_status() if self.assistant else {"status": "unavailable"}
                await self.send_to_connection(connection_id, {
                    "type": "status_response",
                    "data": status
                })
            elif message_type == "text_input":
                # Process text input through the assistant
                text = data.get("text", "")
                if text and self.assistant:
                    response = await self.assistant.process_text_input(text)
                    await self.send_to_connection(connection_id, {
                        "type": "text_response",
                        "data": response
                    })
            else:
                await self.send_to_connection(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
        except json.JSONDecodeError:
            await self.send_to_connection(connection_id, {
                "type": "error",
                "message": "Invalid JSON message"
            })
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            await self.send_to_connection(connection_id, {
                "type": "error",
                "message": "Internal server error"
            })
    
    async def send_to_connection(self, connection_id: str, data: Dict):
        """Send data to a specific connection."""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send(json.dumps(data))
            except Exception as e:
                self.logger.error(f"Error sending to connection {connection_id}: {str(e)}")
    
    async def broadcast(self, data: Dict):
        """Broadcast data to all connections."""
        if self.connections:
            message = json.dumps(data)
            await asyncio.gather(
                *[conn.send(message) for conn in self.connections.values()],
                return_exceptions=True
            )
    
    async def start(self, host: str = "localhost", port: int = 8001):
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self.handle_connection,
            host,
            port
        )
        self.logger.info(f"WebSocket server started on {host}:{port}")
        return self.server
    
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.info("WebSocket server stopped")


async def setup_websocket_api(assistant: Any) -> WebSocketServer:
    """
    Setup and configure the WebSocket API.
    
    Args:
        assistant: AI Assistant instance
        
    Returns:
        WebSocket server instance
    """
    logger = get_logger(__name__)
    
    server = WebSocketServer(assistant)
    
    # Start the server (you might want to configure host/port from config)
    await server.start()
    
    logger.info("WebSocket API setup completed")
    return server