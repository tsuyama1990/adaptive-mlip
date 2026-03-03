import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from pyacemaker.domain_models.defaults import PACE_DRIVER_TEMPLATE
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

    def _write_file_safe(self, path: Path, content: str, mode: int = 0o600) -> None:
        """
        Helper to write files securely with logging.
        Default permissions are 0o600 (user read/write only) for security.
        """
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
        # Script needs execute permission (0o700 for user only)
        self._write_file_safe(output_path, PACE_DRIVER_TEMPLATE, mode=0o700)

    def generate_config(self, output_path: Path) -> None:
        """
        Generates config.ini for EON based on the configuration.

        Args:
            output_path: Path to write the config.ini file.
        """
        # Also generate the driver script in the same directory
        driver_script_name = "pace_driver.py"
        self.generate_driver_script(output_path.parent / driver_script_name)

        # Basic EON configuration template using config fields
        config_content = [
            "[Main]",
            f"job = {self.config.job_type}",
            f"temperature = {self.config.temperature}",
            f"random_seed = {self.config.random_seed}",
            "",
            "[Potential]",
            "potential = command_line",
            f"command = {sys.executable} {driver_script_name}",
            "",
            "[Saddle Search]",
            f"method = {self.config.saddle_search_method}",
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
        def _validate_and_run() -> None:
            validate_path_safe(Path(working_dir))

            if "/" in executable or "\\" in executable:
                validate_path_safe(Path(executable))

            # Use list for subprocess to avoid shell=True, which is safer
            cmd = [executable]

            if self.config.mpi_command:
                # Security: Strictly validate mpi_command to prevent injection
                mpi_cmd_str = self.config.mpi_command.strip()
                forbidden_chars = set(";&|`$()<>")
                if any(c in mpi_cmd_str for c in forbidden_chars):
                    msg = "Forbidden characters in mpi_command"
                    raise ValueError(msg)

                mpi_parts = shlex.split(mpi_cmd_str)
                if not mpi_parts or mpi_parts[0] not in ("mpirun", "mpiexec", "srun", "aprun"):
                    msg = f"Invalid MPI command executable: {mpi_parts[0] if mpi_parts else 'Empty'}"
                    raise ValueError(msg)

                # Strictly whitelist arguments to prevent flag injection (e.g. --allow-run-as-root)
                allowed_flags = {"-n", "-np", "-N", "--n", "-host", "-hostfile", "-machinefile"}
                i = 1
                while i < len(mpi_parts):
                    part = mpi_parts[i]
                    if part.startswith("-"):
                        if part not in allowed_flags:
                            msg = f"Forbidden MPI argument: {part}"
                            raise ValueError(msg)
                        i += 2  # skip the flag and its value (e.g., -n 4)
                    else:
                        # Extra bare values are not allowed
                        msg = f"Unexpected MPI argument formatting: {part}"
                        raise ValueError(msg)

                cmd = mpi_parts + cmd

            cmd_str = " ".join(cmd)
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

        try:
            _validate_and_run()
        except subprocess.CalledProcessError as e:
            msg = f"EON execution failed with return code {e.returncode}. Stderr: {e.stderr}"
            logger.exception(msg)
            # Differentiate between command not found (127) and runtime error
            if e.returncode == 127:
                not_found_msg = f"EON executable not found: {executable}"
                raise RuntimeError(not_found_msg) from e
            raise RuntimeError(msg) from e
        except OSError as e:
            # Handle system errors (e.g. file not found, permission denied)
            msg = f"System error executing EON: {e}"
            logger.exception(msg)
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"An unexpected error occurred during EON execution: {e}"
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
