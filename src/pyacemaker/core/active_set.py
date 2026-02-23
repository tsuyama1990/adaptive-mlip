import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from pyacemaker.core.exceptions import ActiveSetError


class ActiveSetSelector:
    """
    Selects the most informative structures from a candidate pool using D-optimality.
    Wraps the 'pace_activeset' command from Pacemaker.
    """

    def select(
        self,
        candidates: Iterable[Atoms],
        potential_path: str | Path,
        n_select: int,
    ) -> list[Atoms]:
        """
        Selects a subset of structures that maximize the information gain.

        Args:
            candidates: Iterable of candidate structures. Can be a generator.
            potential_path: Path to the current potential (used for descriptors).
            n_select: Number of structures to select. Must be > 0.

        Returns:
            List of selected Atoms objects.

        Raises:
            ActiveSetError: If the external command fails or inputs are invalid.
        """
        if n_select <= 0:
            msg = f"n_select must be positive, got {n_select}"
            raise ValueError(msg)

        potential_path = Path(potential_path)
        if not potential_path.exists():
            msg = f"Potential file not found: {potential_path}"
            raise ActiveSetError(msg)

        # Use a temporary directory to stream write candidates and run the command
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidates_file = tmp_path / "candidates.xyz"
            output_file = tmp_path / "selected.xyz"

            # Stream write candidates to disk to avoid loading all into memory
            try:
                # Open file once and write frame by frame
                with candidates_file.open("w") as f:
                    count = 0
                    for atoms in candidates:
                        write(f, atoms, format="extxyz")  # type: ignore[no-untyped-call]
                        count += 1
            except Exception as e:
                msg = f"Failed to write candidates to temporary file: {e}"
                raise ActiveSetError(msg) from e

            # Verify candidates were written (handle empty iterator case)
            if count == 0:
                # If no candidates, return empty list
                return []

            # Construct command safely
            self._validate_path_safe(candidates_file)
            self._validate_path_safe(potential_path)
            self._validate_path_safe(output_file)

            cmd = [
                "pace_activeset",
                "--dataset",
                str(candidates_file),
                "--potential",
                str(potential_path),
                "--select",
                str(n_select),
                "--output",
                str(output_file),
            ]

            self._run_pace_activeset(cmd)

            # Read selected structures
            if not output_file.exists():
                msg = "Active set selection failed: Output file not created."
                raise ActiveSetError(msg)

            # Read back selected structures
            try:
                selected_structures = read(output_file, index=":")
            except Exception as e:
                msg = f"Failed to read selected structures: {e}"
                raise ActiveSetError(msg) from e

            if isinstance(selected_structures, Atoms):
                return [selected_structures]
            return list(selected_structures)

    def _run_pace_activeset(self, cmd: list[str]) -> None:
        """Executes the pace_activeset command safely."""
        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,  # Security: explicit False
            )
        except subprocess.CalledProcessError as e:
            msg = f"Active set selection failed (exit code {e.returncode}): {e.stderr}"
            raise ActiveSetError(msg) from e
        except FileNotFoundError as e:
            msg = "pace_activeset command not found. Ensure Pacemaker is installed."
            raise ActiveSetError(msg) from e

    def _validate_path_safe(self, path: Path) -> None:
        """Ensures path does not contain suspicious characters for command execution context."""
        s = str(path)
        # Check for dangerous characters in path string
        # Although list args prevent shell injection, preventing weird chars is good practice
        # Whitelist: alphanumeric, dot, underscore, dash, slash
        import re
        if not re.match(r'^[\w\-\.\/]+$', s):
            # This might be too strict for some environments (e.g. spaces in paths), but safer.
            # If tempdir has spaces, this fails.
            # Let's relax to allow spaces but forbid shell metachars.
            # Blacklist approach for shell metachars is safer for general paths
            dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r"]
            if any(c in s for c in dangerous_chars):
                 msg = f"Path contains invalid characters: {path}"
                 raise ActiveSetError(msg)

        if s.startswith("-"):
             msg = f"Path cannot start with '-': {path}"
             raise ActiveSetError(msg)
