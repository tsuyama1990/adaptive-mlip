import argparse
import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from pyacemaker.api.routes import telemetry
from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.defaults import (
    LOG_CONFIG_LOADED,
    LOG_DRY_RUN_COMPLETE,
    LOG_PROJECT_INIT,
)
from pyacemaker.domain_models.scenario import IntentRequest
from pyacemaker.logger import setup_logger, telemetry_broker
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.scenarios.base_scenario import BaseScenario
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario
from pyacemaker.utils.io import load_config

SCENARIO_REGISTRY: dict[str, type[BaseScenario]] = {
    "fept_mgo": FePtMgoScenario,
}


def get_scenario_runner(name: str, config: PyAceConfig) -> BaseScenario:
    """Factory method to get the appropriate scenario runner."""
    if name in SCENARIO_REGISTRY:
        return SCENARIO_REGISTRY[name](config)
    msg = f"Unknown scenario: {name}"
    raise ValueError(msg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Setup before app start
    loop = asyncio.get_running_loop()
    telemetry_broker.initialize_loop(loop)
    broadcast_task = asyncio.create_task(telemetry.broadcast_loop())
    yield
    # Cleanup after app shutdown
    broadcast_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await broadcast_task


app = FastAPI(title="PyAceMaker Intent-Driven API", lifespan=lifespan)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# Payload Size Limiting Middleware
class LimitUploadSize(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_upload_size: int) -> None:
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length is not None and int(content_length) > self.max_upload_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Payload too large. Maximum size is 5MB."},
                )
        res: Response = await call_next(request)
        return res


app.add_middleware(LimitUploadSize, max_upload_size=5_000_000)
app.include_router(telemetry.router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = exc.errors()
    formatted_errors = []
    for error in errors:
        loc_str = ".".join([str(loc) for loc in error.get("loc", []) if loc != "body"])
        formatted_errors.append(
            {
                "loc": loc_str,
                "msg": error.get("msg"),
                "type": error.get("type"),
            }
        )
    return JSONResponse(status_code=422, content={"detail": formatted_errors})


@app.post("/api/v1/intent/compile")
@limiter.limit("60/minute")
async def compile_intent(request: Request, payload: IntentRequest) -> dict[str, Any]:
    return {
        "status": "success",
        "message": "Payload validated successfully",
        "node_count": len(payload.nodes),
    }


def _run_gui_server(args: argparse.Namespace) -> None:

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            timeout_keep_alive=65,
            log_level="info",
        )
    except KeyboardInterrupt:
        pass
    finally:
        sys.exit(0)


def _run_legacy_simulation(args: argparse.Namespace) -> None:
    config_path = Path(args.config)

    try:
        config_dict = load_config(config_path)
        config = PyAceConfig(**config_dict)

        logger = setup_logger(config.logging, config.project_name)
        logger.info(LOG_CONFIG_LOADED)

        if args.dry_run:
            if args.scenario:
                get_scenario_runner(args.scenario, config)
                logger.info("Scenario '%s' selected for dry-run.", args.scenario)
            else:
                Orchestrator(config)

            logger.info(LOG_PROJECT_INIT.format(project_name=config.project_name))
            logger.info(LOG_DRY_RUN_COMPLETE)
            sys.exit(0)

        if args.scenario:
            runner = get_scenario_runner(args.scenario, config)
            runner.run()
        else:
            orchestrator = Orchestrator(config)
            orchestrator.run()

    except Exception:
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Fatal error during execution")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive MLIP construction orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # legacy run
    run_parser = subparsers.add_parser("run", help="Run simulation from config")
    run_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and exit without running"
    )
    run_parser.add_argument("--scenario", type=str, help="Run a specific scenario (e.g., fept_mgo)")

    # new gui command
    gui_parser = subparsers.add_parser("gui", help="Start the Intent-Driven GUI server")
    gui_parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    gui_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    gui_parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")

    args = parser.parse_args()

    # Legacy fallback logic for `pyacemaker --config config.yaml` directly
    if args.command is None:
        if "--config" in sys.argv:
            legacy_parser = argparse.ArgumentParser(
                description="Adaptive MLIP construction orchestrator"
            )
            legacy_parser.add_argument("--config", type=str, required=True)
            legacy_parser.add_argument("--dry-run", action="store_true")
            legacy_parser.add_argument("--scenario", type=str)
            args = legacy_parser.parse_args()
            args.command = "run"
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "gui":
        _run_gui_server(args)
    else:
        _run_legacy_simulation(args)


if __name__ == "__main__":
    main()
