import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from ase import Atoms
from ase.io import iread, write

from pyacemaker.core.exceptions import ActiveSetError
from pyacemaker.utils.path import validate_path_safe
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
        anchor: Atoms | None = None,
    ) -> Iterator[Atoms]:
        """
        Selects a subset of structures that maximize the information gain.

        This method uses a temporary directory to stream candidates to disk, ensuring
        memory safety by processing in batches. It then runs the `pace_activeset`
        command and streams the results back.

        Args:
            candidates: Iterable of candidate structures. Can be a generator.
            potential_path: Path to the current potential (used for descriptors).
            n_select: Number of structures to select. Must be > 0.
            anchor: Optional structure to always include in the selection (e.g., the halt structure).

        Returns:
            Iterator of selected Atoms objects.

        Raises:
            ActiveSetError: If the external command fails or inputs are invalid.
        """
        if n_select <= 0:
            msg = f"n_select must be positive, got {n_select}"
            raise ValueError(msg)

        # Handle Anchor Logic
        actual_n_select = n_select
        if anchor is not None:
            yield anchor
            actual_n_select -= 1
            if actual_n_select <= 0:
                # If only requested 1 and anchor is provided, we are done.
                return

        potential_path = Path(potential_path)
        if not potential_path.exists():
            msg = f"Potential file not found: {potential_path}"
            raise ActiveSetError(msg)

        # Use a temporary directory to stream write candidates and run the command
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidates_file = tmp_path / "candidates.xyz"
            output_file = tmp_path / "selected.xyz"

            count = self._write_candidates(candidates, candidates_file)
            if count == 0:
                return

            self._execute_selection(candidates_file, potential_path, output_file, actual_n_select)

            # Stream read selected structures to avoid memory spikes
            if not output_file.exists():
                msg = "Active set selection failed: Output file not created."
                raise ActiveSetError(msg)

            if output_file.stat().st_size == 0:
                msg = "Active set selection failed: Output file is empty."
                raise ActiveSetError(msg)

            try:
                yield from iread(output_file, index=":")
            except Exception as e:
                msg = f"Failed to read selected structures: {e}"
                raise ActiveSetError(msg) from e

    def _write_candidates(self, candidates: Iterable[Atoms], file_path: Path) -> int:
        """
        Writes candidates to disk using pure streaming.
        Passes the iterator directly to ASE write to avoid any intermediate batch materialization.
        """
        count = 0

        # Generator wrapper to count items while streaming
        def counting_wrapper(iterable: Iterable[Atoms]) -> Iterator[Atoms]:
            nonlocal count
            for atoms in iterable:
                count += 1
                yield atoms

        try:
            # write() iterates over the generator and writes frame by frame.
            # This is O(1) memory usage relative to dataset size.
            write(file_path, counting_wrapper(candidates), format="extxyz")  # type: ignore[arg-type]
        except Exception as e:
            msg = f"Failed to write candidates to temporary file: {e}"
            raise ActiveSetError(msg) from e
        return count

    def _execute_selection(
        self, candidates_file: Path, potential_path: Path, output_file: Path, n_select: int
    ) -> None:
        """Constructs and runs the selection command."""
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
        Ensures path is safe using strict resolution and character allowlisting.
        Delegates to centralized utility.
        """
        try:
            validate_path_safe(path)
        except ValueError as e:
            raise ActiveSetError(str(e)) from e
