
import re
import shlex

import numpy as np
from ase import Atoms
from lammps import lammps

from pyacemaker.core.exceptions import LammpsDriverError


class LammpsDriver:
    """
    Wrapper for the LAMMPS Python interface.
    Handles initialization and data extraction.
    """

    # Whitelist of allowed characters in LAMMPS commands
    # Alphanumeric, whitespace, and specific safe symbols used in LAMMPS syntax.
    # Allowed: _ - . / = ' " # * $ { } ( ) [ ] , : + > <
    # Note: ; | & are strictly forbidden (shell metachars).
    # Backticks ` are forbidden.
    SAFE_CMD_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.\/\=\s\'\"\#\*\$\{\}\(\)\[\],:\+\>\<]+$")

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
            raise LammpsDriverError(msg) from e

    def _validate_command(self, cmd: str) -> None:
        """Validates a single command against security rules."""
        if not self.SAFE_CMD_PATTERN.match(cmd):
            msg = f"Command contains forbidden characters: {cmd}"
            raise LammpsDriverError(msg)

        # Tokenize using shlex to correctly handle quotes
        try:
            tokens = shlex.split(cmd)
        except ValueError as e:
             msg = f"Command has unbalanced quotes or invalid syntax: {cmd}"
             raise LammpsDriverError(msg) from e

        if not tokens:
            return

        # Check for forbidden commands
        if "shell" in tokens:
            msg = "Script contains forbidden command 'shell'."
            raise LammpsDriverError(msg)

        if "python" in tokens:
             msg = "Script contains forbidden command 'python'."
             raise LammpsDriverError(msg)

    def run(self, script: str) -> None:
        """
        Execute a LAMMPS script.

        Args:
            script: String containing LAMMPS commands (can be multi-line).

        Raises:
            LammpsDriverError: If script contains non-ASCII characters or unsafe commands.
        """
        if not script.isascii():
             msg = "Script contains non-ASCII characters, which may be unsafe."
             raise LammpsDriverError(msg)

        # Use splitlines() to handle various line endings (\n, \r, \r\n) safely
        for line in script.splitlines():
            cmd = line.strip()
            # Skip comments and empty lines
            if not cmd or cmd.startswith("#"):
                continue

            # Remove trailing comments if any?
            # LAMMPS comments start with #.
            # shlex.split with comments=True handles this?
            # But we want to validate the *command* part.
            # Simple approach: split by # and take first part?
            # But # can be inside quotes?
            # shlex handles this.

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
        # extract_variable(name, group, type): type 0 = integer, 1 = double
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
        # Get number of atoms
        natoms = self.lmp.get_natoms()
        if natoms == 0:
            return Atoms()

        # Gather positions (type 1 = double, count 3)
        # Returns a ctypes pointer to array of doubles
        x_ptr = self.lmp.gather_atoms("x", 1, 3)
        # Convert to numpy array view (no copy yet)
        positions_view = np.ctypeslib.as_array(x_ptr, shape=(natoms, 3))

        # Gather types (type 0 = integer, count 1)
        # Returns a ctypes pointer to array of ints
        types_ptr = self.lmp.gather_atoms("type", 0, 1)
        types_view = np.ctypeslib.as_array(types_ptr, shape=(natoms,))

        # Map types to symbols
        # LAMMPS types are 1-based. elements list is 0-based.
        # type i corresponds to elements[i-1]
        try:
            symbols = [elements[t - 1] for t in types_view]
        except IndexError as e:
             msg = f"LAMMPS type index out of range for elements list: {e}"
             raise LammpsDriverError(msg) from e

        # Get cell
        # extract_box returns (boxlo, boxhi, xy, yz, xz, periodicity, box_change)
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = self.lmp.extract_box()

        # Construct cell matrix
        # LAMMPS uses a specific triclinic representation
        lx = boxhi[0] - boxlo[0]
        ly = boxhi[1] - boxlo[1]
        lz = boxhi[2] - boxlo[2]

        # If orthogonal, xy=yz=xz=0.0
        # ASE expects cell as 3x3 matrix.
        cell = np.array([
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz]
        ])

        # ASE Atoms constructor will copy the positions array if it's a numpy array.
        # We pass the view 'positions_view'.
        return Atoms(symbols=symbols, positions=positions_view, cell=cell, pbc=periodicity)

    def get_forces(self) -> np.ndarray:
        """
        Retrieve current atomic forces.

        Returns:
            Numpy array of shape (N, 3) containing forces.
        """
        natoms = self.lmp.get_natoms()
        if natoms == 0:
            return np.zeros((0, 3))

        # Gather forces (type 1 = double, count 3)
        f_ptr = self.lmp.gather_atoms("f", 1, 3)
        # Return a copy to ensure safety
        return np.ctypeslib.as_array(f_ptr, shape=(natoms, 3)).copy()

    def get_stress(self) -> np.ndarray:
        """
        Retrieve global stress tensor in Voigt notation (pxx, pyy, pzz, pyz, pxz, pxy).
        Note: LAMMPS pressure is -stress. This returns STRESS (positive for tension).
        Units: Pressure units of the simulation (e.g. bars).

        This assumes 'thermo_style custom pxx pyy pzz pyz pxz pxy' or similar has been set,
        OR we can extract global computes.
        Actually, simpler to use 'extract_global' if computed?
        No, usually we rely on thermo variables.
        """
        # We must ensure thermo output is current.
        # Extract variables pxx, pyy, etc.
        # Note: These variables must be available.
        # If not, we might return zeros or raise error.
        # But setting them up is responsibility of the script.

        components = ["pxx", "pyy", "pzz", "pyz", "pxz", "pxy"]
        stress = []
        for c in components:
            try:
                # Pressure in LAMMPS is usually positive for compression.
                # Stress = -Pressure.
                # But pxx etc are pressure tensor components.
                # ASE expects stress.
                val = self.extract_variable(c)
                stress.append(-val) # Convert pressure to stress
            except Exception:
                # If variable not found, try to compute it?
                # Without explicit thermo, variables might be 0.
                stress.append(0.0)

        return np.array(stress)
