import argparse
import logging
import sys
from pathlib import Path

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.defaults import (
    LOG_CONFIG_LOADED,
    LOG_DRY_RUN_COMPLETE,
    LOG_PROJECT_INIT,
)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive MLIP construction orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and exit without running"
    )
    parser.add_argument(
        "--scenario", type=str, help="Run a specific scenario (e.g., fept_mgo)"
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    try:
        config = load_config(config_path)
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
