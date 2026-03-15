import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import write

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.telemetry import SimulationState, TelemetryFrame
from pyacemaker.logger import telemetry_broker
from pyacemaker.utils.io import write_lammps_streaming
from pyacemaker.utils.structure import get_species_order

logger = logging.getLogger(__name__)

class IoManager:
    """
    Manages disk I/O and telemetry streaming for MD trajectories.
    """
    def __init__(self, stream_interval: int = 10) -> None:
        self.stream_interval = stream_interval

    def write_trajectory(
        self,
        atoms: Atoms,
        filepath: Path,
        step: int,
        state: SimulationState,
        force_publish: bool = False
    ) -> None:
        """
        Writes frame to disk and pushes it to telemetry queue based on interval.
        High-uncertainty events can use force_publish=True to bypass downsampling.
        Implements basic file rotation if trajectory exceeds 500MB.
        """
        # File Rotation Logic
        MAX_FILE_SIZE_BYTES = 500_000_000  # 500MB
        if filepath.exists() and filepath.stat().st_size > MAX_FILE_SIZE_BYTES:
            import datetime
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
            rotated_path = filepath.with_name(f"{filepath.stem}_{timestamp}{filepath.suffix}")
            try:
                filepath.rename(rotated_path)
                logger.info(f"Rotated trajectory file: {filepath} -> {rotated_path}")
            except OSError as e:
                logger.warning(f"Failed to rotate trajectory file {filepath}: {e}")

        # Fast non-blocking Disk I/O dispatch via executor to prevent locking the orchestrator logic
        # For an append format like extxyz, threadpool I/O avoids massive GIL locks
        def _write_task() -> None:
            try:
                write(str(filepath), atoms, format="extxyz", append=True)
            except OSError as e:
                msg = f"Failed to write trajectory frame to {filepath} due to disk or permission error: {e}"
                logger.exception(msg)
                raise RuntimeError(msg) from e
            except Exception:
                logger.exception(f"Failed to write trajectory frame to {filepath}")

        # The orchestrator is synchronous, but we can offload I/O to avoid blocking the physics
        import concurrent.futures
        # Using a fire-and-forget mechanism on a bounded executor
        if not hasattr(self, "_executor"):
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._executor.submit(_write_task)

        # Telemetry downsampling check
        if force_publish or (step % self.stream_interval == 0):
            self._publish_frame(atoms, step, state)

    def _publish_frame(self, atoms: Atoms, step: int, state: SimulationState) -> None:
        try:
            positions = atoms.get_positions().flatten().tolist() # type: ignore[no-untyped-call]

            forces = None
            if "forces" in atoms.arrays:
                forces = atoms.get_forces().flatten().tolist() # type: ignore[no-untyped-call]

            variances = None
            if "c_gamma" in atoms.arrays:
                variances = atoms.get_array("c_gamma").flatten().tolist() # type: ignore[no-untyped-call]

            frame = TelemetryFrame(
                workflow_id="default_workflow",
                step_number=step,
                current_state=state,
                positions=positions,
                forces=forces,
                variances=variances
            )
            telemetry_broker.publish(frame)
        except Exception:
            logger.exception("Failed to publish telemetry frame")

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
                # Load only the first frame to minimize memory usage
                from ase.io import iread

                try:
                    atoms_iter = iread(str(structure))
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

    def _validate_output_path(self, output_path: Path) -> Path:
        import os
        if output_path.is_symlink():
            msg = f"Invalid output path, symlink detected: {output_path}"
            raise ValueError(msg)

        # Canonicalize to resolve any . or .. components strictly
        real_output_path = Path(os.path.realpath(output_path)).resolve()
        real_cwd = Path(os.path.realpath(Path.cwd())).resolve()

        # Additional protection against encoded injection and null bytes
        if "\0" in str(output_path):
            msg = "Null byte detected in path"
            raise ValueError(msg)

        if not real_output_path.is_relative_to(real_cwd):
            msg = f"Invalid output path, potential path traversal detected: {output_path}"
            raise ValueError(msg)
        return real_output_path

    def _write_structure_memory(
        self, structure: Atoms, output_path: Path, elements: list[str]
    ) -> None:
        """Writes structure to disk using streaming writer if possible with atomic transactions."""
        import os

        real_output_path = self._validate_output_path(output_path)

        # Create a temporary file path next to the target output path
        temp_path = real_output_path.with_name(f".{real_output_path.name}.tmp")

        try:
            # Memory Safety Fix: Always attempt streaming first if atom_style allows
            streaming_success = False
            if self.config.atom_style == "atomic":
                try:
                    with temp_path.open("w") as f:
                        write_lammps_streaming(f, structure, elements)

                        f.flush()
                        os.fsync(f.fileno())
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
                    str(temp_path),
                    structure,
                    format="lammps-data",
                    specorder=elements,
                    atom_style=self.config.atom_style.value,
                )

            def _raise_error() -> None:
                msg = f"Temporary file {temp_path} is missing or empty before finalizing."
                raise ValueError(msg)  # noqa: TRY301

            # Atomic rename validation
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                _raise_error()

            temp_path.replace(real_output_path)

        except (ValueError, OSError, RuntimeError) as e:
            # Rollback
            import contextlib

            if temp_path.exists():
                with contextlib.suppress(OSError):
                    temp_path.unlink()
            msg = f"Failed to write LAMMPS data file: {e}"
            raise RuntimeError(msg) from e
