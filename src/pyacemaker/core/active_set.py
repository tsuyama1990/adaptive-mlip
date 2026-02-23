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
            candidates: List or Iterable of candidate structures.
            potential_path: Path to the current potential (used for descriptors).
            n_select: Number of structures to select.

        Returns:
            List of selected Atoms objects.

        Raises:
            ActiveSetError: If the external command fails.
        """
        potential_path = Path(potential_path)
        if not potential_path.exists():
            # If potential doesn't exist, we can't compute descriptors.
            # In a cold start scenario, we might return random selection,
            # but here we assume a potential exists (even if it's just initialized).
            msg = f"Potential file not found: {potential_path}"
            raise ActiveSetError(msg)

        # Ensure candidates is a list for length check, though we might stream write
        candidates_list = list(candidates)
        if not candidates_list:
            return []

        if n_select >= len(candidates_list):
            return candidates_list

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidates_file = tmp_path / "candidates.xyz"
            output_file = tmp_path / "selected.xyz"

            # Write candidates to disk
            write(candidates_file, candidates_list)

            # Construct command
            # pace_activeset -d candidates.xyz -p potential.yace -n n_select -o selected.xyz
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

            try:
                subprocess.run(  # noqa: S603
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                msg = f"Active set selection failed: {e.stderr}"
                raise ActiveSetError(msg) from e
            except FileNotFoundError as e:
                # Handle case where pace_activeset is not installed
                msg = "pace_activeset command not found. Ensure Pacemaker is installed."
                raise ActiveSetError(msg) from e

            # Read selected structures
            if not output_file.exists():
                msg = "Active set selection failed: Output file not created."
                raise ActiveSetError(msg)

            selected_structures = read(output_file, index=":")

            # Ensure return type is list[Atoms]
            if isinstance(selected_structures, Atoms):
                return [selected_structures]
            return list(selected_structures)
