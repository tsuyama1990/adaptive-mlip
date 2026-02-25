import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from pyacemaker.domain_models.constants import PACE_DRIVER_TEMPLATE
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.process import ProcessRunner, SubprocessRunner
from pyacemaker.utils.path import validate_path_safe

logger = logging.getLogger(__name__)


class EONWrapper:
    """
    Interface to EON (Adaptive Kinetic Monte Carlo) software.
    Manages configuration generation and process execution.
    """

    def __init__(self, config: EONConfig, runner: ProcessRunner | None = None) -> None:
        self.config = config
        self.runner = runner or SubprocessRunner()

    def _write_file_safe(self, path: Path, content: str, mode: int = 0o644) -> None:
        """Helper to write files securely with logging."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            path.chmod(mode)
            logger.info("Generated file at %s", path)
        except OSError as e:
            msg = f"Failed to write file {path}: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    def generate_driver_script(self, output_path: Path) -> None:
        """
        Generates the Python driver script for EON to call.

        Args:
            output_path: Path to write the script.
        """
        self._write_file_safe(output_path, PACE_DRIVER_TEMPLATE, mode=0o755)

    def generate_config(self, output_path: Path) -> None:
        """
        Generates config.ini for EON based on the configuration.

        Args:
            output_path: Path to write the config.ini file.
        """
        # Also generate the driver script in the same directory
        driver_script_name = "pace_driver.py"
        self.generate_driver_script(output_path.parent / driver_script_name)

        # Basic EON configuration template
        config_content = [
            "[Main]",
            "job = akmc",
            f"temperature = {self.config.temperature}",
            f"random_seed = {self.config.random_seed}",
            "",
            "[Potential]",
            "potential = command_line",
            f"command = {sys.executable} {driver_script_name}",
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

        self._write_file_safe(output_path, "\n".join(config_content))

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

            # Pass environment variable for potential path
            # ProcessRunner doesn't have get_env usually
            run_env = os.environ.copy()
            run_env["PACE_POTENTIAL_PATH"] = str(self.config.potential_path)

            # Execute using abstracted runner
            # We use check=True to raise CalledProcessError on non-zero exit
            result = self.runner.run(cmd, cwd=working_dir, env=run_env, check=True)

            logger.info("EON simulation completed successfully.")
            logger.debug("EON stdout: %s", result.stdout)

        except subprocess.CalledProcessError as e:
            msg = f"EON execution failed with return code {e.returncode}. Stderr: {e.stderr}"
            logger.exception(msg)
            # Differentiate between command not found (127) and runtime error
            if e.returncode == 127:
                not_found_msg = f"EON executable not found: {executable}"
                raise RuntimeError(not_found_msg) from e
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
