import logging
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from lammps import lammps

from pyacemaker.domain_models.constants import LAMMPS_SAFE_CMD_PATTERN

logger = logging.getLogger(__name__)


class LammpsDriver:
    """
    Wrapper for the LAMMPS Python interface.
    Handles initialization and data extraction.
    """

    # Whitelist of allowed characters in LAMMPS commands
    SAFE_CMD_PATTERN = re.compile(LAMMPS_SAFE_CMD_PATTERN)

    def __init__(self, cmdargs: list[str] | None = None) -> None:
        """
        Initialize the LAMMPS instance.

        Args:
            cmdargs: List of command-line arguments for LAMMPS (e.g., ["-screen", "none"]).

        Raises:
            RuntimeError: If LAMMPS library cannot be loaded.
        """
        try:
            # Create LAMMPS instance
            self.lmp = lammps(cmdargs=cmdargs)
        except Exception as e:
            # Catch OSError (missing lib) or other initialization errors
            msg = f"Failed to initialize LAMMPS: {e}"
            raise RuntimeError(msg) from e

    def _validate_command(self, cmd: str) -> None:
        """Validates a single command against security rules."""
        if not self.SAFE_CMD_PATTERN.match(cmd):
            msg = f"Command contains forbidden characters: {cmd}"
            raise ValueError(msg)

        tokens = cmd.split()
        if "shell" in tokens:
            msg = "Script contains forbidden command 'shell'."
            raise ValueError(msg)

    def run(self, script: str) -> None:
        """
        Execute a LAMMPS script provided as a string.

        Args:
            script: String containing LAMMPS commands (can be multi-line).

        Raises:
            ValueError: If script contains non-ASCII characters or unsafe commands.
        """
        if not script.isascii():
            msg = "Script contains non-ASCII characters, which may be unsafe."
            raise ValueError(msg)

        for line in script.split("\n"):
            cmd = line.strip()
            if cmd:
                self._validate_command(cmd)
                self.lmp.command(cmd)

    def run_file(self, filepath: str | Path) -> None:
        """
        Execute a LAMMPS script from a file.
        Preferable for large scripts to avoid memory overhead.

        Args:
            filepath: Path to the LAMMPS input script.
        """
        path = Path(filepath)
        if not path.exists():
            msg = f"Input script not found: {path}"
            raise FileNotFoundError(msg)

        # Basic security check: scan file for 'shell' command?
        # Reading file defeats the purpose of streaming if we read it all.
        # But for security, we might need to scan.
        # However, if we trust the generator, we can skip.
        # Given "Security" requirement, let's stream read and validate.
        # But lammps.file() executes the file. It doesn't validate line by line in Python.
        # If we use lammps.file(), we bypass _validate_command unless we pre-scan.
        # Pre-scanning line by line is O(N) IO but O(1) memory.

        # Security: Read, validate, and execute line-by-line.
        # Do not use self.lmp.file(str(path)) to avoid TOCTOU and file content injection.
        # This acts as a sandboxed execution by strictly controlling commands.
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                cmd = line.strip()
                # Ignore comments
                if cmd.startswith("#"):
                    continue
                if cmd:
                    # Remove comments from line end
                    cmd = cmd.split("#")[0].strip()
                    if cmd:
                        self._validate_command(cmd)
                        self.lmp.command(cmd)

    def extract_variable(self, name: str) -> float:
        """
        Extract a global variable from LAMMPS.

        Args:
            name: Name of the variable defined in LAMMPS.

        Returns:
            Value of the variable as a float.
        """
        val = self.lmp.extract_variable(name, None, 0)
        return float(val)

    def get_atoms(self, elements: list[str]) -> Atoms:
        """
        Retrieve the current state as an ASE Atoms object.

        Args:
            elements: List of chemical symbols corresponding to LAMMPS atom types (1-based).

        Returns:
            ASE Atoms object with current positions, cell, and species.
        """
        natoms = self.lmp.get_natoms()
        if natoms == 0:
            return Atoms()

        x_ptr = self.lmp.gather_atoms("x", 1, 3)
        positions_view = np.ctypeslib.as_array(x_ptr, shape=(natoms, 3))

        types_ptr = self.lmp.gather_atoms("type", 0, 1)
        types_view = np.ctypeslib.as_array(types_ptr, shape=(natoms,))

        try:
            symbols = [elements[t - 1] for t in types_view]
        except IndexError as e:
            msg = f"LAMMPS type index out of range for elements list: {e}"
            raise ValueError(msg) from e

        boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lmp.extract_box()

        lx = boxhi[0] - boxlo[0]
        ly = boxhi[1] - boxlo[1]
        lz = boxhi[2] - boxlo[2]

        cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])

        return Atoms(symbols=symbols, positions=positions_view, cell=cell, pbc=periodicity)

    def get_forces(self) -> np.ndarray:
        """
        Retrieve forces for all atoms.

        Returns:
            Numpy array of shape (N, 3) containing forces.
        """
        natoms = self.lmp.get_natoms()
        if natoms == 0:
            return np.zeros((0, 3))

        f_ptr = self.lmp.gather_atoms("f", 1, 3)
        forces_view = np.ctypeslib.as_array(f_ptr, shape=(natoms, 3))
        return forces_view.copy()

    def get_stress(self) -> np.ndarray:
        """
        Retrieve stress tensor for the system (Voigt notation).
        Units: Pressure units (usually Bar or similar).
        """
        try:
            pxx = self.extract_variable("pxx")
            pyy = self.extract_variable("pyy")
            pzz = self.extract_variable("pzz")
            pyz = self.extract_variable("pyz")
            pxz = self.extract_variable("pxz")
            pxy = self.extract_variable("pxy")
            return np.array([-pxx, -pyy, -pzz, -pyz, -pxz, -pxy])
        except Exception:
            logger.warning("Failed to extract stress tensor. Returning zero vector.")
            return np.zeros(6)
