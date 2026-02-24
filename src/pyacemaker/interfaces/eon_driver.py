import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.utils.path import validate_path_safe

logger = logging.getLogger(__name__)


class EONWrapper:
    """
    Interface to EON (Adaptive Kinetic Monte Carlo) software.
    Manages configuration generation and process execution.
    """

    def __init__(self, config: EONConfig) -> None:
        self.config = config

    def generate_config(self, output_path: Path) -> None:
        """
        Generates config.ini for EON based on the configuration.

        Args:
            output_path: Path to write the config.ini file.
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Basic EON configuration template
        config_content = [
            "[Main]",
            "job = akmc",
            f"temperature = {self.config.temperature}",
            f"random_seed = {12345}",  # Should be configurable or random
            "",
            "[Potential]",
            "potential = pace",
            f"potentials_path = {self.config.potential_path}",
            "",
            "[Saddle Search]",
            "method = min_mode",
            "",
            "[Structure]",
            f"supercell = {self.config.supercell}",
            "",
            "[Communicator]",
            "type = local",
            f"client_path = {self.config.eon_executable}",
        ]

        if self.config.mpi_command:
             pass

        try:
            output_path.write_text("\n".join(config_content))
            logger.info("Generated EON config at %s", output_path)
        except OSError as e:
            msg = f"Failed to write EON config: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def run(self, working_dir: Path) -> None:
        """
        Runs the EON client in the specified working directory.

        Args:
            working_dir: Directory where the simulation should run.
        """
        executable = self.config.eon_executable

        # Security: Validate paths
        try:
            if "/" in executable or "\\" in executable:
                validate_path_safe(Path(executable))

            cmd = [executable]

            if self.config.mpi_command:
                cmd = shlex.split(self.config.mpi_command) + cmd

            cmd_str = ' '.join(cmd)
            logger.info("Starting EON simulation in %s with command: %s", working_dir, cmd_str)

            # Execute
            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=working_dir,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("EON simulation completed successfully.")
            logger.debug("EON stdout: %s", result.stdout)

        except subprocess.CalledProcessError as e:
            msg = f"EON execution failed: {e.stderr}"
            logger.exception("EON execution failed with return code %s. Stderr: %s", e.returncode, e.stderr)
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"An error occurred during EON execution: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def parse_results(self, result_dir: Path) -> dict[str, Any]:
        """
        Parses output files from EON run (dynamics.txt, processtable.dat).

        Args:
            result_dir: Directory containing EON output files.

        Returns:
            Dictionary containing parsed data.
        """
        results = {}

        dynamics_file = result_dir / "dynamics.txt"
        if dynamics_file.exists():
            results["dynamics"] = dynamics_file.read_text()

        process_table = result_dir / "processtable.dat"
        if process_table.exists():
            results["processtable"] = process_table.read_text()

        return results
