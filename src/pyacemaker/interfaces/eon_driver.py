import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.process import ProcessRunner, SubprocessRunner
from pyacemaker.utils.path import validate_path_safe

logger = logging.getLogger(__name__)


PACE_DRIVER_SCRIPT = """
import sys
import os
import numpy as np
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS

# Read potential path from environment for security
POTENTIAL_PATH = os.environ.get("PACE_POTENTIAL_PATH")
if not POTENTIAL_PATH:
    sys.stderr.write("Error: PACE_POTENTIAL_PATH not set\\n")
    sys.exit(1)

def read_input():
    try:
        lines = sys.stdin.readlines()
        if not lines:
            return None, None, None

        # Parse EON format (assumed based on standard command line potentials)
        # 1. Number of atoms
        num_atoms = int(lines[0].strip())

        # 2. Box (3 lines)
        cell = np.zeros((3, 3))
        cell[0] = [float(x) for x in lines[1].split()]
        cell[1] = [float(x) for x in lines[2].split()]
        cell[2] = [float(x) for x in lines[3].split()]

        # 3. Coordinates (N lines)
        # EON usually sends only coordinates x y z, not species.
        # We need to map them to species from a template.
        coords = []
        for i in range(num_atoms):
            coords.append([float(x) for x in lines[4+i].split()])

        return num_atoms, cell, np.array(coords)
    except Exception as e:
        sys.stderr.write(f"Error reading input: {e}\\n")
        sys.exit(1)

def main():
    # 1. Load template structure (pos.con) to get species
    if not os.path.exists("pos.con"):
        sys.stderr.write("Error: pos.con not found\\n")
        sys.exit(1)

    try:
        # EON uses .con format, ASE supports it
        template = read("pos.con", format="eon")
    except Exception:
        # Fallback if ASE cannot read eon format directly or extension issue
        # Try reading as 'con' or just assume it works if supported
        template = read("pos.con")

    # 2. Read current configuration from stdin
    n, cell, coords = read_input()
    if n is None:
        sys.exit(0)

    if n != len(template):
        sys.stderr.write(f"Error: Atom count mismatch ({n} vs {len(template)})\\n")
        sys.exit(1)

    # 3. Update structure
    template.set_cell(cell)
    template.set_positions(coords)

    # 4. Setup Calculator
    # Using LAMMPS via ASE is robust.
    # For speed, one might use lammps python module directly,
    # but constructing the input script for PACE is complex.
    # We rely on ASE's LAMMPS calculator or similar.
    # We need to specify the potential file.

    # We construct a pair_style command for PACE.
    # Assuming 'pace' pair style is available in the LAMMPS binary.
    # pair_style pace
    # pair_coeff * * potential.yace Element1 Element2 ...

    species = sorted(list(set(template.get_chemical_symbols())))
    species_str = " ".join(species)

    parameters = {
        "pair_style": "pace",
        "pair_coeff": [f"* * {POTENTIAL_PATH} {species_str}"]
    }

    calc = LAMMPS(parameters=parameters, files=[POTENTIAL_PATH])
    template.calc = calc

    # 5. Compute
    try:
        energy = template.get_potential_energy()
        forces = template.get_forces()
    except Exception as e:
        sys.stderr.write(f"Error computing energy: {e}\\n")
        sys.exit(1)

    # 6. Output results
    # Format: Energy (1 line)
    # Forces (N lines, x y z)
    print(f"{energy:.16f}")
    for f in forces:
        print(f"{f[0]:.16f} {f[1]:.16f} {f[2]:.16f}")

if __name__ == "__main__":
    main()
"""


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
        self._write_file_safe(output_path, PACE_DRIVER_SCRIPT, mode=0o755)

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
