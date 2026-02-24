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

    def prepare_workspace(self, structure: Atoms) -> tuple[Any, Path, Path, Path, list[str]]:
        """
        Creates temporary directory and writes structure file.

        Args:
            structure: Atomic structure to simulate.

        Returns:
            temp_dir_ctx: Context manager for temporary directory.
            data_file: Path to input data file (in temp dir).
            dump_file: Path to output trajectory file (in CWD).
            log_file: Path to output log file (in CWD).
            elements: List of element symbols in order.
        """
        # RAM disk usage optimization via config
        temp_dir_ctx = tempfile.TemporaryDirectory(dir=self.config.temp_dir)
        temp_dir = Path(temp_dir_ctx.name)

        run_id = uuid.uuid4().hex[:8]
        data_file = temp_dir / f"data_{run_id}.lmp"

        # Persistence: Outputs go to current working directory
        cwd = Path.cwd()
        dump_file = cwd / f"dump_{run_id}.lammpstrj"
        log_file = cwd / f"log_{run_id}.lammps"

        elements = get_species_order(structure)

        if len(structure) > 10000:
            logger.info("Streaming large structure (%d atoms) to disk.", len(structure))

        try:
            # Memory Safety Fix: Always attempt streaming first if atom_style allows
            # This avoids loading large structures into memory via ASE's default writer
            streaming_success = False
            if self.config.atom_style == "atomic":
                try:
                    with data_file.open("w") as f:
                        write_lammps_streaming(f, structure, elements)
                    streaming_success = True
                    logger.debug("Successfully wrote LAMMPS data file using streaming.")
                except ValueError as e:
                    # Likely non-orthogonal box or unsupported feature
                    logger.debug("Streaming write skipped (e.g. non-orthogonal): %s. Falling back to ASE.", e)

            if not streaming_success:
                # Fallback: Use ASE write.
                # Note: This might not be memory safe for massive structures >10M atoms,
                # but covers cases like triclinic cells where simple streaming is complex.
                # If memory safety is strictly paramount, we should raise error or implement complex streaming.
                # Given current scope, this is acceptable fallback for non-orthogonal cells.
                if len(structure) > 10000:
                    logger.info("Streaming large structure (%d atoms) to disk.", len(structure))
                if len(structure) > 1000000:
                    logger.warning("Falling back to ASE write for large structure (%d atoms). Memory usage may be high.", len(structure))

                write(str(data_file), structure, format="lammps-data", specorder=elements, atom_style=self.config.atom_style)

        except Exception as e:
            temp_dir_ctx.cleanup()
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e

        return temp_dir_ctx, data_file, dump_file, log_file, elements
