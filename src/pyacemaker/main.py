import argparse
import sys
from pathlib import Path

import yaml

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="PyAceMaker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize workspace")
    init_parser.add_argument("--config", "-c", type=Path, required=True, help="Path to config.yaml")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run workflow step")
    run_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=False,
        help="Path to config.yaml (optional if state exists)",
    )
    run_parser.add_argument(
        "--step", "-s", type=int, required=True, choices=[1], help="Step to run (1=DIRECT)"
    )

    args = parser.parse_args()

    if args.command == "init":
        config_path = args.config
        if not config_path.exists():
            print(f"Error: Config file not found at {config_path}")  # noqa: T201
            sys.exit(1)

        try:
            with config_path.open("r") as f:
                config_data = yaml.safe_load(f)
            config = PyAceConfig(**config_data)
            orch = Orchestrator(config)
            orch.initialize_workspace()
            print("Workspace initialized successfully.")  # noqa: T201
        except Exception as e:
            print(f"Initialization failed: {e}")  # noqa: T201
            sys.exit(1)

    elif args.command == "run":
        config_path = args.config
        if not config_path:
            # Try local config.yaml
            config_path = Path("config.yaml")

        if not config_path.exists():
            print("Error: Config file not found. Please provide --config.")  # noqa: T201
            sys.exit(1)

        try:
            with config_path.open("r") as f:
                config_data = yaml.safe_load(f)
            config = PyAceConfig(**config_data)
            orch = Orchestrator(config)

            if args.step == 1:
                orch.run_step1()
                print("Step 1 completed.")  # noqa: T201
            else:
                print(f"Step {args.step} not implemented yet.")  # noqa: T201
        except Exception as e:
            print(f"Execution failed: {e}")  # noqa: T201
            sys.exit(1)


if __name__ == "__main__":
    main()
