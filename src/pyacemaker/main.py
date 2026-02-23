import argparse
import logging
import sys
from pathlib import Path

from pyacemaker.constants import LOG_CONFIG_LOADED, LOG_DRY_RUN_COMPLETE, LOG_PROJECT_INIT
from pyacemaker.logger import setup_logger
from pyacemaker.orchestrator import Orchestrator
from pyacemaker.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive MLIP construction orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit without running")

    args = parser.parse_args()
    config_path = Path(args.config)

    try:
        config = load_config(config_path)
        # Initialize Logger
        logger = setup_logger(config.logging, config.project_name)
        logger.info(LOG_CONFIG_LOADED)

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
