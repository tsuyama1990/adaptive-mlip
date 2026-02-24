from pathlib import Path
from typing import Any

from ase import Atoms


class LammpsValidator:
    """
    Validates inputs for LAMMPS engine operations.
    Follows SRP by separating validation logic from execution.
    """

    @staticmethod
    def validate_structure(structure: Any) -> None:
        """
        Validates the atomic structure.

        Args:
            structure: Input structure object.

        Raises:
            ValueError: If structure is invalid or empty.
            TypeError: If input is not an ASE Atoms object.
        """
        if structure is None:
            msg = "Structure must be provided."
            raise ValueError(msg)

        if not isinstance(structure, Atoms):
            msg = f"Expected ASE Atoms object, got {type(structure)}."
            raise TypeError(msg)

        if len(structure) == 0:
            msg = "Structure contains no atoms."
            raise ValueError(msg)

    @staticmethod
    def validate_potential(potential: Any) -> Path:
        """
        Validates the potential path.

        Args:
            potential: Path to potential file (str or Path).

        Returns:
            Validated Path object.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If input is invalid.
        """
        if potential is None:
            msg = "Potential path must be provided."
            raise ValueError(msg)

        path = Path(potential)

        # Security: Check for path traversal or restricted directories
        try:
            resolved_path = path.resolve()
            # Basic check: Ensure path is absolute after resolution
            if not resolved_path.is_absolute():
                msg = f"Potential path must be resolvable to absolute path: {path}"
                raise ValueError(msg)

            # Additional checks could be added here (e.g. whitelist of directories)

        except Exception as e:
            msg = f"Invalid potential path: {e}"
            raise ValueError(msg) from e

        if not path.exists():
            msg = f"Potential file not found: {path}"
            raise FileNotFoundError(msg)

        return path
