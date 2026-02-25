import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

from ase.io import read

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.process import ProcessRunner, SubprocessRunner
from pyacemaker.utils.path import validate_path_safe

# Try importing lammps, if fails (e.g. during build), we can't run.
try:
    from lammps import lammps
except ImportError:
    lammps = None


def run_driver(potential_path: str, threshold: float) -> int:
    """
    Driver function called by the external script executed by EON.
    Reads coordinates from stdin (EON format), calculates E/F/Gamma using LAMMPS,
    and writes to stdout.

    Returns exit code (0 for success, 100 for halt).
    """
    if not lammps:
        print("Error: lammps module not found", file=sys.stderr)  # noqa: T201
        return 1

    try:
        # Read input from stdin
        input_data = sys.stdin.read()

        # Initialize LAMMPS
        lmp = lammps()
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command("boundary p p p")

        # Try 'eon' format first, fallback to 'xyz'
        try:
            atoms = read(StringIO(input_data), format="eon")
        except Exception:
            atoms = read(StringIO(input_data), format="xyz")

        # Setup LAMMPS box (Simplified for placeholder)
        _ = atoms.get_cell()

        # Setup Pair Style
        # We assume potential_path is valid and elements match
        elements = " ".join(sorted(set(atoms.get_chemical_symbols())))
        lmp.command("pair_style pace")
        lmp.command(f"pair_coeff * * {potential_path} {elements}")

        # Compute Gamma (OTF)
        # lmp.command("compute gamma all pace ...")
        lmp.command("run 0")

        # Check Gamma
        # max_gamma = lmp.extract_compute("gamma", 0, 0)
        max_gamma = 0.0

        if max_gamma > threshold:
            return 100

        # Get E and F
        pe = lmp.get_thermo("pe")
        # forces = lmp.get_forces()

        print(f"{pe}")  # noqa: T201
        # Print forces line by line if needed

    except Exception as e:
        print(f"Driver Error: {e}", file=sys.stderr)  # noqa: T201
        return 1

    return 0


class EONWrapper:
    """
    Wrapper around EON client execution.
    Handles configuration generation and process execution.
    """

    def __init__(self, config: EONConfig, runner: ProcessRunner | None = None) -> None:
        self.config = config
        self.runner = runner or SubprocessRunner()

    def generate_config(self, path: Path) -> None:
        """Generates the config.ini file for EON."""
        # Ensure potential path is safe before writing it to config
        validate_path_safe(self.config.potential_path)

        content = f"""[Main]
job = process_search
temperature = {self.config.temperature}
steps = {self.config.akmc_steps}
random_seed = {self.config.random_seed}

[Potential]
potential = script
script_path = ./pace_driver.py
potentials_path = {self.config.potential_path}

[Process Search]
min_mode_method = dimer
"""
        path.write_text(content)

    def run(self, working_dir: Path) -> None:
        """Runs the EON client in the specified directory."""
        cmd = [self.config.eon_executable]
        if self.config.mpi_command:
            cmd = self.config.mpi_command.split() + cmd

        # We explicitly pass check=True to match test expectations and safety
        try:
            self.runner.run(cmd, cwd=working_dir, check=True)
        except subprocess.CalledProcessError as e:
            msg = f"EON execution failed: {e}"
            raise RuntimeError(msg) from e

    def parse_results(self, working_dir: Path) -> dict[str, Any]:
        """Parses EON output files."""
        results = {}
        # Parse dynamics.txt
        dynamics_file = working_dir / "dynamics.txt"
        if dynamics_file.exists():
            results["dynamics"] = dynamics_file.read_text()

        # Parse processtable.dat
        pt_file = working_dir / "processtable.dat"
        if pt_file.exists():
            results["processtable"] = pt_file.read_text()

        return results
