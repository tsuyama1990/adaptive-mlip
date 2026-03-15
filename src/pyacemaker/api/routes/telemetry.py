import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from pyacemaker.domain_models.telemetry import StateChangePayload, SystemTopology, TelemetryFrame
from pyacemaker.logger import telemetry_broker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/telemetry", tags=["telemetry"])


class ConnectionManager:
    """Manages active WebSocket connections for telemetry streams."""

    def __init__(self) -> None:
        # Maps workflow_id to a set of active connections
        self.active_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, workflow_id: str) -> None:
        await websocket.accept()
        if workflow_id not in self.active_connections:
            self.active_connections[workflow_id] = set()
        self.active_connections[workflow_id].add(websocket)

    def disconnect(self, websocket: WebSocket, workflow_id: str) -> None:
        if workflow_id in self.active_connections:
            self.active_connections[workflow_id].discard(websocket)
            if not self.active_connections[workflow_id]:
                del self.active_connections[workflow_id]

    async def _safe_send(self, connection: WebSocket, message: str, workflow_id: str) -> None:
        try:
            # Apply explicit backpressure/timeout to prevent hanging on slow clients
            await asyncio.wait_for(connection.send_text(message), timeout=2.0)
        except Exception:
            # Drop the disconnected or unresponsive client
            self.disconnect(connection, workflow_id)

    async def broadcast(self, message: str, workflow_id: str) -> None:
        if workflow_id in self.active_connections:
            # Capture the current connections in a tuple to avoid mutation errors during iteration
            current_connections = tuple(self.active_connections[workflow_id])

            # Execute all sends concurrently without blocking the main broadcast loop
            send_tasks = [
                self._safe_send(conn, message, workflow_id) for conn in current_connections
            ]

            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)


manager = ConnectionManager()


@router.websocket("/stream/{workflow_id}")
async def telemetry_stream(websocket: WebSocket, workflow_id: str) -> None:
    """
    WebSocket endpoint for real-time telemetry streaming.
    Receives messages continuously from the background broadcaster task.
    """
    await manager.connect(websocket, workflow_id)
    try:
        # The loop keeps the connection open and monitors for client disconnects
        while True:
            # Wait for client to send a message (e.g. ping), or just keep socket open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, workflow_id)


async def broadcast_loop() -> None:
    """
    Continuously pulls from the telemetry_broker queue and broadcasts
    the serialized payload to all active WebSocket clients.
    Designed to run as an asyncio background task.
    """
    while True:
        try:
            # This is an async queue get
            payload = await telemetry_broker.queue.get()

            if isinstance(payload, (SystemTopology, TelemetryFrame, StateChangePayload)):
                serialized = payload.model_dump_json()
                workflow_id = getattr(payload, "workflow_id", "default_workflow")
            else:
                continue

            await manager.broadcast(serialized, workflow_id)

        except asyncio.CancelledError:
            break
        except Exception:
            # Prevent the broadcasting loop from crashing completely on an unexpected error
            logger.exception("Error in telemetry broadcast loop")
