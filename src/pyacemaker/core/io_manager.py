import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from pyacemaker.domain_models.md import MDConfig
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

        # Validation for large structures (OOM/IO risk)
        if len(structure) > 10000:
            logger.warning(
                "Writing large structure (%d atoms) to disk. "
                "Consider streaming or using RAM disk if not already enabled.",
                len(structure)
            )

        elements = get_species_order(structure)

        try:
            # ase.io.write handles buffering usually, but for very large structures
            # we rely on system memory.
            write(str(data_file), structure, format="lammps-data", specorder=elements, atom_style=self.config.atom_style)
        except Exception as e:
            # Ensure cleanup if write fails? context manager handles it if we raise.
            # But we must close context if we don't return it?
            # We return context, caller must enter it or we enter it here?
            # Better design: This method yields paths?
            # Or simpler: run() uses context manager, passes temp_dir path to manager?
            # Let's refactor: Manager provides a context manager method.
            temp_dir_ctx.cleanup()
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e

        return temp_dir_ctx, data_file, dump_file, log_file, elements
