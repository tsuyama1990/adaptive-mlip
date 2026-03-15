import asyncio
import contextlib
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from pyacemaker.domain_models.logging import LoggingConfig


def setup_logger(config: LoggingConfig, project_name: str) -> logging.Logger:
    """
    Sets up the logger based on the configuration.
    Ensures idempotency to avoid duplicate handlers.

    Args:
        config: LoggingConfig object.
        project_name: Name of the project (logger name).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(project_name)

    # If handlers already exist, assume it's already configured and return.
    # This prevents duplicate logs if setup_logger is called multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(config.level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if config.log_file:
        log_path = Path(config.log_file)
        # Ensure directory exists
        if log_path.parent != Path():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            config.log_file, maxBytes=config.max_bytes, backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TelemetryBroker:
    """
    Singleton class managing the Pub/Sub architecture for the WebSockets.
    Safely bridges synchronous producer threads (LAMMPS) to the async event loop.
    """

    _instance: Optional["TelemetryBroker"] = None

    def __new__(cls) -> "TelemetryBroker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        from pyacemaker.domain_models.constants import TELEMETRY_QUEUE_MAXSIZE

        if getattr(self, "_initialized", False):
            return
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=TELEMETRY_QUEUE_MAXSIZE)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._initialized = True

    def initialize_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called upon FastAPI application startup to bind the running loop."""
        self.loop = loop

    def _validate_payload_size(self, payload: dict[str, Any]) -> str:
        # Pre-serialization strict length checks to avoid MemoryError during json dumping
        # A Float64 generally serializes to ~24 bytes max, but we assume 32 for strict bounds
        from pyacemaker.domain_models.constants import (
            BYTES_PER_ATOMIC_NUMBER,
            BYTES_PER_FLOAT64,
            MAX_PAYLOAD_SIZE_BYTES,
            PAYLOAD_BASE_OVERHEAD_BYTES,
        )

        estimated_bytes = PAYLOAD_BASE_OVERHEAD_BYTES  # Base JSON structural overhead
        if payload.get("positions") is not None:
            estimated_bytes += len(payload.get("positions", [])) * BYTES_PER_FLOAT64
            if payload.get("forces", []):
                estimated_bytes += len(payload.get("forces", [])) * BYTES_PER_FLOAT64
            if payload.get("variances", []):
                estimated_bytes += len(payload.get("variances", [])) * BYTES_PER_FLOAT64
        elif payload.get("atomic_numbers") is not None:
            estimated_bytes += len(payload.get("atomic_numbers", [])) * BYTES_PER_ATOMIC_NUMBER
            if payload.get("cell_dimensions", []):
                estimated_bytes += len(payload.get("cell_dimensions", [])) * BYTES_PER_FLOAT64

        if estimated_bytes > MAX_PAYLOAD_SIZE_BYTES:
            msg = f"Payload too large (estimated {estimated_bytes} bytes). Limit is {MAX_PAYLOAD_SIZE_BYTES} bytes."
            raise ValueError(msg)

        try:
            serialized_payload = (
                payload.model_dump_json()
                if hasattr(payload, "model_dump_json")
                else json.dumps(payload)
            )
            payload_size = len(serialized_payload.encode("utf-8"))
        except MemoryError as e:
            msg = "Payload serialization caused MemoryError. Payload rejected."
            raise ValueError(msg) from e

        if payload_size > MAX_PAYLOAD_SIZE_BYTES:
            msg = f"Payload too large ({payload_size} bytes). Limit is {MAX_PAYLOAD_SIZE_BYTES} bytes."
            raise ValueError(msg)

        return serialized_payload

    def publish(self, payload: dict[str, Any]) -> None:
        """
        Thread-safe method called by synchronous worker threads.
        Implements drop-oldest backpressure to prevent memory leaks if frontend is slow.
        """
        self._validate_payload_size(payload)

        if self.loop is None or not self.loop.is_running():
            return

        def _sync_put() -> None:
            try:
                self.queue.put_nowait(payload)
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    # Drop-oldest policy: Pop the oldest item
                    self.queue.get_nowait()

                # Retry putting the new payload
                with contextlib.suppress(asyncio.QueueFull):
                    self.queue.put_nowait(payload)

        self.loop.call_soon_threadsafe(_sync_put)


telemetry_broker = TelemetryBroker()
