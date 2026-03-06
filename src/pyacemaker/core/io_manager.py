import tempfile
from pathlib import Path

from ase import Atoms
from ase.io.lammpsdata import write_lammps_data

from pyacemaker.domain_models.constants import DEFAULT_RAM_DISK_PATH
from pyacemaker.domain_models.md import MDConfig


class LammpsFileManager:
    """
    Manages LAMMPS input/output files and workspace preparation.
    """

    def __init__(self, config: MDConfig) -> None:
        self.config = config

    def _determine_workspace_dir(self) -> Path:
        """Determines the base directory for the LAMMPS workspace."""
        base_dir = Path(self.config.temp_dir) if self.config.temp_dir else Path("/tmp")
        if str(base_dir) == DEFAULT_RAM_DISK_PATH:
            base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def _get_unique_elements(self, structure: Atoms) -> list[str]:
        """Extracts unique chemical symbols from the structure."""
        symbols = structure.get_chemical_symbols() # type: ignore[no-untyped-call]
        # Sort for determinism
        return sorted(set(symbols))

    def prepare_workspace(
        self, structure: Atoms
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, Path, list[str]]:
        """
        Creates a temporary workspace and writes the LAMMPS data file.

        Returns:
            Tuple of (TemporaryDirectory object, path to data_file, path to dump_file, path to log_file, elements_list)
        """
        base_dir = self._determine_workspace_dir()

        # Create temporary directory within the chosen base dir
        # We return the TemporaryDirectory object so the caller can control its lifecycle (e.g. using 'with')
        temp_dir_ctx = tempfile.TemporaryDirectory(dir=base_dir, prefix="lammps_")
        work_dir = Path(temp_dir_ctx.name)

        data_file = work_dir / "structure.data"
        dump_file = work_dir / "trajectory.dump"
        log_file = work_dir / "lammps.log"

        elements = self._get_unique_elements(structure)

        try:
            # Memory optimization: Large structures can be written in chunks if supported,
            # but write_lammps_data typically writes directly to file stream efficiently.
            # We enforce use of safe paths.
            if len(structure) > 100000:
                # Placeholder for explicit chunked writing if write_lammps_data proves too memory intensive
                self._write_structure_memory(structure, data_file, elements)
            else:
                self._write_structure_memory(structure, data_file, elements)

        except Exception:
            # Clean up if setup fails
            temp_dir_ctx.cleanup()
            raise
        else:
            return temp_dir_ctx, data_file, dump_file, log_file, elements

    def _write_structure_memory(
        self, structure: Atoms, data_file: Path, elements: list[str]
    ) -> None:
        """Writes structure to data file ensuring atomic positions are handled optimally."""
        try:
            with data_file.open("w") as fd:
                # Use atom_style from config
                write_lammps_data(
                    fd, structure, specorder=elements, atom_style=self.config.atom_style.value
                )
        except Exception as e:
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e
