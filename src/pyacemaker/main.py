import argparse
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

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.defaults import (
    LOG_CONFIG_LOADED,
    LOG_DRY_RUN_COMPLETE,
    LOG_PROJECT_INIT,
)
from pyacemaker.domain_models.scenario import IntentRequest
from pyacemaker.logger import setup_logger
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.scenarios.base_scenario import BaseScenario
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario
from pyacemaker.utils.io import load_config


def get_scenario_runner(name: str, config: PyAceConfig) -> BaseScenario:
    """Factory method to get the appropriate scenario runner."""
    if name == "fept_mgo":
        return FePtMgoScenario(config)
    msg = f"Unknown scenario: {name}"
    raise ValueError(msg)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Setup before app start
    yield
    # Cleanup after app shutdown


app = FastAPI(title="PyAceMaker Intent-Driven API", lifespan=lifespan)

# Restrict CORS specifically as required
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def compile_intent(payload: IntentRequest) -> dict[str, Any]:
    return {
        "status": "success",
        "message": "Payload validated successfully",
        "node_count": len(payload.nodes),
    }


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
            # We try parsing again assuming default 'run' structure if no subparser was provided
            # but arguments were passed directly.
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
        uvicorn.run("pyacemaker.main:app", host=args.host, port=args.port, workers=args.workers)
        sys.exit(0)

    config_path = Path(args.config)

    try:
        config_dict = load_config(config_path)
        # Validate config using Pydantic model
        config = PyAceConfig(**config_dict)

        # Initialize Logger
        logger = setup_logger(config.logging, config.project_name)
        logger.info(LOG_CONFIG_LOADED)

        if args.dry_run:
            if args.scenario:
                get_scenario_runner(args.scenario, config)  # Validate scenario name
                logger.info("Scenario '%s' selected for dry-run.", args.scenario)
            else:
                Orchestrator(config)  # Verify orchestrator init

            logger.info(LOG_PROJECT_INIT.format(project_name=config.project_name))
            logger.info(LOG_DRY_RUN_COMPLETE)
            sys.exit(0)

        if args.scenario:
            runner = get_scenario_runner(args.scenario, config)
            runner.run()
        else:
            # Run workflow
            orchestrator = Orchestrator(config)
            orchestrator.run()

    except Exception:
        # Fallback logging if logger isn't set up or fails
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Fatal error during execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
