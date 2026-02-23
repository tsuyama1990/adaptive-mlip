import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from ase import Atoms
from ase.io import iread, write

from pyacemaker.core.exceptions import ActiveSetError
from pyacemaker.utils.misc import batched
from pyacemaker.utils.process import run_command


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
    ) -> Iterator[Atoms]:
        """
        Selects a subset of structures that maximize the information gain.

        Args:
            candidates: Iterable of candidate structures. Can be a generator.
            potential_path: Path to the current potential (used for descriptors).
            n_select: Number of structures to select. Must be > 0.

        Returns:
            Iterator of selected Atoms objects.

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

            # Stream write candidates to disk using buffering/chunking
            # This avoids I/O bottleneck of writing 1-by-1 and Memory bottleneck of loading all.
            # Using batched() from utils.misc
            count = 0
            try:
                # Open file once and write in chunks
                # We use append=True logic manually by keeping file open?
                # ASE write supports writing a list of images.
                # batched returns a tuple of items.
                BATCH_SIZE = 1000
                with candidates_file.open("w") as f:
                    for batch in batched(candidates, BATCH_SIZE):
                        # Convert tuple to list for ASE
                        # write supports list of Atoms
                        # We must specify format explicitly for streaming write usually
                        write(f, list(batch), format="extxyz")
                        count += len(batch)
            except Exception as e:
                msg = f"Failed to write candidates to temporary file: {e}"
                raise ActiveSetError(msg) from e

            # Verify candidates were written
            if count == 0:
                # If no candidates, return empty iterator
                yield from []
                return

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

            # Check file integrity
            if output_file.stat().st_size == 0:
                 msg = "Active set selection failed: Output file is empty."
                 raise ActiveSetError(msg)

            # Stream read selected structures to avoid memory spikes
            try:
                yield from iread(output_file, index=":")
            except Exception as e:
                msg = f"Failed to read selected structures: {e}"
                raise ActiveSetError(msg) from e

    def _run_pace_activeset(self, cmd: list[str]) -> None:
        """Executes the pace_activeset command safely."""
        try:
            run_command(cmd)
        except Exception as e:
            # Re-wrap exceptions to ActiveSetError context
            msg = f"Active set execution failed: {e}"
            raise ActiveSetError(msg) from e

    def _validate_path_safe(self, path: Path) -> None:
        """
        Ensures path does not contain suspicious shell metacharacters.
        Allows spaces and common filename characters.
        """
        s = str(path)
        dangerous_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", "\n", "\r"]
        if any(c in s for c in dangerous_chars):
             msg = f"Path contains invalid characters: {path}"
             raise ActiveSetError(msg)

        if s.startswith("-"):
             msg = f"Path cannot start with '-': {path}"
             raise ActiveSetError(msg)
