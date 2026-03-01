import logging
import tempfile
import uuid
from pathlib import Path

from ase import Atoms

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
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, list[str]]:
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
                # Security: Validate input path before processing
                from pyacemaker.utils.path import validate_path_safe
                safe_structure_path = validate_path_safe(Path(structure))

                # 100% Streaming approach: Do not load any Atoms objects.
                from pyacemaker.utils.io import detect_elements, stream_extxyz_to_lammps

                elements = detect_elements(safe_structure_path, max_frames=1)
                with data_file.open("w") as f:
                    stream_extxyz_to_lammps(safe_structure_path, f, elements)

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

    @staticmethod
    def _raise_value_error(msg: str) -> None:
        raise ValueError(msg)

    def _write_structure_memory(
        self, structure: Atoms, output_path: Path, elements: list[str]
    ) -> None:
        """Writes structure to disk using streaming writer."""
        try:
            # Memory Safety Fix: Always use streaming. ASE fallback removed per strictly streaming constraints.
            if self.config.atom_style != "atomic":
                msg = f"Atom style {self.config.atom_style} is not supported by streaming writer. Only 'atomic' is currently supported to prevent OOM errors."
                LammpsFileManager._raise_value_error(msg)

            with output_path.open("w") as f:
                write_lammps_streaming(f, structure, elements)
            logger.debug("Successfully wrote LAMMPS data file using streaming.")

        except Exception as e:
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e
