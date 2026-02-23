
import numpy as np
from ase import Atoms
from lammps import lammps


class LammpsDriver:
    """
    Wrapper for the LAMMPS Python interface.
    Handles initialization and data extraction.
    """

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

    def run(self, script: str) -> None:
        """
        Execute a LAMMPS script.

        Args:
            script: String containing LAMMPS commands (can be multi-line).

        Raises:
            ValueError: If script contains non-ASCII characters or unsafe shell constructs.
        """
        if not script.isascii():
             msg = "Script contains non-ASCII characters, which may be unsafe."
             raise ValueError(msg)

        # Basic sanitization against injection via filenames/arguments
        # If a line contains 'shell', we might want to flag it unless strictly allowed?
        # But legitimate scripts might use shell.
        # However, for our engine, we don't expect 'shell' command.
        # We can blacklist 'shell' keyword for extra safety if generated internally.
        if "shell" in script:
             # This is a heuristic.
             # If "shell" appears, it might be an injection attempt if we didn't generate it.
             # Since we control the generator, we know we don't use 'shell'.
             msg = "Script contains forbidden command 'shell'."
             raise ValueError(msg)

        for line in script.split("\n"):
            cmd = line.strip()
            if cmd:
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
             raise ValueError(msg) from e

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
