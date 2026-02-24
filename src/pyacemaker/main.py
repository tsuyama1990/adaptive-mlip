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
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.logger import setup_logger
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario
from pyacemaker.utils.io import load_config


def _run_scenario(config: PyAceConfig, scenario_name: str, dry_run: bool) -> None:
    logger = logging.getLogger("pyacemaker.main")

    if dry_run:
        logger.info(f"Dry run: Scenario '{scenario_name}' configuration valid.")
        sys.exit(0)

    logger.info(f"Running scenario: {scenario_name}")
    if scenario_name == "fept_mgo":
        scenario = FePtMgoScenario(config)
        scenario.run()
    else:
        # Should be caught by argparse choices, but for safety
        msg = f"Unknown scenario: {scenario_name}"
        raise ValueError(msg)

    logger.info("Scenario execution complete.")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive MLIP construction orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and exit without running"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["fept_mgo"],
        help="Run a specific scenario instead of the active learning loop",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    try:
        config = load_config(config_path)

        # Initialize Logger
        logger = setup_logger(config.logging, config.project_name)
        logger.info(LOG_CONFIG_LOADED)

        # Handle Scenario Execution
        if args.scenario:
            # Ensure config has scenario section if needed
            if config.scenario is None:
                config.scenario = ScenarioConfig(name=args.scenario)
            elif config.scenario.name != args.scenario:
                # If CLI arg differs from config, warn but proceed with CLI arg
                logger.warning(
                    f"CLI scenario '{args.scenario}' overrides config '{config.scenario.name}'"
                )

            _run_scenario(config, args.scenario, args.dry_run)

        # Initialize Orchestrator
        orchestrator = Orchestrator(config)

        if args.dry_run:
            logger.info(LOG_PROJECT_INIT.format(project_name=config.project_name))
            logger.info(LOG_DRY_RUN_COMPLETE)
            sys.exit(0)

        # Run workflow
        orchestrator.run()

    except Exception:
        # Fallback logging if logger isn't set up or fails
        logging.basicConfig(level=logging.ERROR)
        logging.exception("Fatal error during execution")
        sys.exit(1)


if __name__ == "__main__":
    main()
