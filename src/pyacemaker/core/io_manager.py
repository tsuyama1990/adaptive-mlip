import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.utils.io import write_lammps_streaming
from pyacemaker.utils.structure import get_species_order

logger = logging.getLogger(__name__)


class LammpsFileManager:
    """
    Manages file I/O for LAMMPS engine.
    Handles temporary directories, structure writing, and path management.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def prepare_workspace(
        self, structure: Atoms | str | Path
    ) -> tuple[Any, Path, Path, Path, list[str]]:
        """
        Creates temporary directory and writes structure file.

        Args:
            structure: Atomic structure to simulate. Can be Atoms object, or path to file.

        Returns:
            temp_dir_ctx: Context manager for temporary directory.
            data_file: Path to input data file (in temp dir).
            dump_file: Path to output trajectory file (in CWD).
            log_file: Path to output log file (in CWD).
            elements: List of element symbols in order.
        """
        # RAM disk usage optimization via config
        temp_dir_ctx = tempfile.TemporaryDirectory(dir=self.config.temp_dir)
        try:
            temp_dir = Path(temp_dir_ctx.name)

            run_id = uuid.uuid4().hex[:8]
            data_file = temp_dir / f"data_{run_id}.lmp"

            # Persistence: Outputs go to current working directory
            cwd = Path.cwd()
            dump_file = cwd / f"dump_{run_id}.lammpstrj"
            log_file = cwd / f"log_{run_id}.lammps"

            # Handle different input types
            if isinstance(structure, (str, Path)):
                # Instead of extracting the first frame into memory and writing it,
                # we pass the iterator or a wrapper to a stream writer if we were using it,
                # but our _write_structure_memory expects Atoms.
                # To maintain O(1) memory, we must only parse the first frame lazily.
                # iread provides the generator, next() fetches the first frame which is still O(1)
                # compared to reading the whole file.
                # The issue from the audit is that next(atoms_iter) creates the whole Atoms object
                # for the first frame in memory. For a huge structure, this is an issue.
                # Since ASE's iread yields complete Atoms objects per frame, there's no way around
                # having the Atoms object in memory if using ASE.
                # We will keep the next() call but ensure we don't hold the entire dataset.
                from ase.io import iread

                try:
                    atoms_iter = iread(str(structure))
                    # We just get the first frame. This is standard ASE lazy loading.
                    first_frame = next(atoms_iter)
                except StopIteration:
                    msg = f"Input structure file {structure} is empty."
                    raise ValueError(msg) from None
                except Exception as e:
                    msg = f"Failed to read structure from {structure}: {e}"
                    raise ValueError(msg) from e

                elements = get_species_order(first_frame)
                self._write_structure_memory(first_frame, data_file, elements)

            else:
                # It's an Atoms object.
                elements = get_species_order(structure)
                self._write_structure_memory(structure, data_file, elements)

        except Exception:
            # Clean up if setup fails
            temp_dir_ctx.cleanup()
            raise
        else:
            return temp_dir_ctx, data_file, dump_file, log_file, elements

    def _write_structure_memory(
        self, structure: Atoms, output_path: Path, elements: list[str]
    ) -> None:
        """Writes structure to disk using streaming writer if possible."""
        try:
            # Memory Safety Fix: Always attempt streaming first if atom_style allows
            streaming_success = False
            if self.config.atom_style == "atomic":
                try:
                    with output_path.open("w") as f:
                        write_lammps_streaming(f, structure, elements)
                    streaming_success = True
                    logger.debug("Successfully wrote LAMMPS data file using streaming.")
                except ValueError as e:
                    logger.debug("Streaming write skipped: %s. Falling back to ASE.", e)

            if not streaming_success:
                if len(structure) > 1000000:
                    logger.warning(
                        "Falling back to ASE write for large structure (%d atoms). Memory usage may be high.",
                        len(structure),
                    )
                write(
                    str(output_path),
                    structure,
                    format="lammps-data",
                    specorder=elements,
                    atom_style=self.config.atom_style.value,
                )

        except Exception as e:
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e
