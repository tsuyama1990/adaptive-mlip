import argparse
import sys
from pathlib import Path

from pyacemaker.orchestrator import Orchestrator
from pyacemaker.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive MLIP construction orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and exit without running"
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    try:
        config = load_config(config_path)
        print("Configuration loaded successfully.")  # noqa: T201

        # Initialize Orchestrator
        orchestrator = Orchestrator(config)

        if args.dry_run:
            print(f"Project: {config.project_name} initialized.")  # noqa: T201
            print("Dry run complete. Exiting.")  # noqa: T201
            sys.exit(0)

        # Run workflow
        orchestrator.run()

    except Exception as e:
        # Pydantic ValidationError prints nicely formatted errors
        # We print to stderr for script handling
        print(f"Error: {e}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
