import logging
import tempfile
import uuid
from contextlib import AbstractContextManager
from pathlib import Path

from ase import Atoms
from ase.io import write

from pyacemaker.domain_models.defaults import DEFAULT_MD_ATOM_STYLE
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
        self, structure: Atoms
    ) -> tuple[AbstractContextManager[str], Path, Path, Path, list[str]]:
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
        # MDConfig doesn't have temp_dir, using default temp dir (None)
        temp_dir_ctx = tempfile.TemporaryDirectory()
        temp_dir = Path(temp_dir_ctx.name)

        run_id = uuid.uuid4().hex[:8]
        data_file = temp_dir / f"data_{run_id}.lmp"

        # Persistence: Outputs go to current working directory
        cwd = Path.cwd()
        dump_file = cwd / f"dump_{run_id}.lammpstrj"
        log_file = cwd / f"log_{run_id}.lammps"

        elements = get_species_order(structure)
        atom_style = DEFAULT_MD_ATOM_STYLE

        try:
            # Scalability: Always attempt streaming write first for efficiency.
            # This avoids loading the entire formatted string into memory.
            try:
                with data_file.open("w") as f:
                    write_lammps_streaming(f, structure, elements, atom_style=atom_style)
            except ValueError:
                # Streaming might fail for non-orthogonal boxes or unsupported styles.
                # Fallback to ASE write which is robust but less memory efficient.
                logger.warning(
                    "Streaming write failed (likely non-orthogonal box). Falling back to ASE."
                )
                write(
                    str(data_file),
                    structure,
                    format="lammps-data",
                    specorder=elements,
                    atom_style=atom_style,
                )
        except Exception as e:
            temp_dir_ctx.cleanup()
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e

        return temp_dir_ctx, data_file, dump_file, log_file, elements
