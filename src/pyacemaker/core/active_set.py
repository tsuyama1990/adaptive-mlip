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
            # ase.io.write supports writing multiple images from an iterable
            try:
                write(candidates_file, candidates)  # type: ignore[arg-type]
            except Exception as e:
                msg = f"Failed to write candidates to temporary file: {e}"
                raise ActiveSetError(msg) from e

            # Verify candidates were written (handle empty iterator case)
            if not candidates_file.exists() or candidates_file.stat().st_size == 0:
                # If no candidates, return empty list
                return []

            # Construct command safely
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
            # Depending on size of active set (usually small, e.g. < 1000), reading into list is fine.
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
