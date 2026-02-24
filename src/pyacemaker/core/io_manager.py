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

        try:
            # Optimization: Use streaming writer for large structures to avoid OOM
            # Only supports 'atomic' style and orthogonal boxes for now.
            if len(structure) > 10000 and self.config.atom_style == "atomic":
                logger.info("Streaming large structure (%d atoms) to disk.", len(structure))
                try:
                    with data_file.open("w") as f:
                        write_lammps_streaming(f, structure, elements)
                except ValueError as e:
                    # Fallback if non-orthogonal or other issue
                    logger.warning(
                        "Streaming failed for large structure (Error: %s). Falling back to standard ASE write (memory intensive).",
                        e,
                    )
                    write(
                        str(data_file),
                        structure,
                        format="lammps-data",
                        specorder=elements,
                        atom_style=self.config.atom_style,
                    )
            else:
                write(
                    str(data_file),
                    structure,
                    format="lammps-data",
                    specorder=elements,
                    atom_style=self.config.atom_style,
                )
        except Exception as e:
            temp_dir_ctx.cleanup()
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e

        return temp_dir_ctx, data_file, dump_file, log_file, elements
