import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from ase import Atoms
from ase.io import iread, write

from pyacemaker.core.exceptions import ActiveSetError
from pyacemaker.domain_models.defaults import (
    DEFAULT_ACTIVE_SET_CHUNK_SIZE,
    DEFAULT_ACTIVE_SET_OVERSAMPLING_FACTOR,
)
from pyacemaker.utils.misc import batched
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
        memory safety by processing in batches (chunks). It then runs the `pace_activeset`
        command on each chunk and merges results for a final selection.

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

            # Streaming strategy:
            # 1. Split candidates into chunks.
            # 2. Select from each chunk into intermediate files.
            intermediate_files = self._process_chunks(
                candidates, potential_path, actual_n_select, tmp_path
            )

            if not intermediate_files:
                return  # No candidates

            # 3. Merge intermediate results
            merged_file = tmp_path / "merged_candidates.xyz"
            count_merged = self._merge_files(intermediate_files, merged_file)

            if count_merged == 0:
                return

            # 4. Final selection
            final_output_file = tmp_path / "final_selected.xyz"
            # If we have fewer candidates than requested after merge, select all (or max possible)
            n_final_select = min(actual_n_select, count_merged)

            self._execute_selection(
                merged_file, potential_path, final_output_file, n_final_select
            )

            # Stream read selected structures to avoid memory spikes
            if not final_output_file.exists():
                msg = "Active set selection failed: Output file not created."
                raise ActiveSetError(msg)

            if final_output_file.stat().st_size == 0:
                msg = "Active set selection failed: Output file is empty."
                raise ActiveSetError(msg)

            try:
                yield from iread(final_output_file, index=":")
            except Exception as e:
                msg = f"Failed to read selected structures: {e}"
                raise ActiveSetError(msg) from e

    def _process_chunks(
        self,
        candidates: Iterable[Atoms],
        potential_path: Path,
        n_select: int,
        tmp_path: Path,
    ) -> list[Path]:
        """Process chunks of candidates and return paths to intermediate selection files."""
        intermediate_files: list[Path] = []

        # Use centralized default constant for chunk size
        # iterate over batched(candidates) which yields tuples (materialized chunks)
        # This is memory safe as long as CHUNK_SIZE is reasonable.
        for chunk_idx, batch in enumerate(batched(candidates, DEFAULT_ACTIVE_SET_CHUNK_SIZE)):
            chunk_file = tmp_path / f"chunk_{chunk_idx}.xyz"
            selected_chunk_file = tmp_path / f"selected_chunk_{chunk_idx}.xyz"

            # Write chunk
            count = self._write_candidates(batch, chunk_file)
            if count > 0:
                # Determine how many to select from this chunk
                n_chunk_select = min(
                    n_select * DEFAULT_ACTIVE_SET_OVERSAMPLING_FACTOR, count
                )

                # Run selection on chunk
                self._execute_selection(
                    chunk_file, potential_path, selected_chunk_file, n_chunk_select
                )

                if selected_chunk_file.exists() and selected_chunk_file.stat().st_size > 0:
                    intermediate_files.append(selected_chunk_file)

        return intermediate_files

    def _merge_files(self, file_paths: list[Path], output_file: Path) -> int:
        """Merges multiple XYZ files into one."""
        def merged_iterator() -> Iterator[Atoms]:
            for f in file_paths:
                yield from iread(f, index=":")

        return self._write_candidates(merged_iterator(), output_file)

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
