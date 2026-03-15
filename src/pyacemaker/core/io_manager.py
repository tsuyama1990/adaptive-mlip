import concurrent.futures
import contextlib
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, cast

from ase import Atoms
from ase.io import write

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.telemetry import SimulationState, TelemetryFrame
from pyacemaker.logger import telemetry_broker
from pyacemaker.utils.io import write_lammps_streaming
from pyacemaker.utils.path import validate_path_safe
from pyacemaker.utils.structure import get_species_order

logger = logging.getLogger(__name__)

# Global shared I/O executor to prevent thread spawning overhead and control max bounds across application
# Utilizing defaults (min(32, os.cpu_count() + 4)) to scale dynamically with the container or server limits
GLOBAL_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="io_manager_writer")


class IoManager:
    """
    Manages disk I/O and telemetry streaming for MD trajectories.
    """

    def __init__(self, stream_interval: int = 10) -> None:
        self.stream_interval = stream_interval
        self._executor = GLOBAL_IO_EXECUTOR

    def _rotate_file_if_needed(self, filepath: Path) -> None:
        import datetime
        import fcntl

        from pyacemaker.domain_models.constants import MAX_FILE_SIZE_BYTES

        if filepath.exists() and filepath.stat().st_size > MAX_FILE_SIZE_BYTES:
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d%H%M%S")
            rotated_path = filepath.with_name(f"{filepath.stem}_{timestamp}{filepath.suffix}")

            try:
                with filepath.open("a") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        filepath.replace(rotated_path)
                        logger.info(f"Rotated trajectory file: {filepath} -> {rotated_path}")
                    except (BlockingIOError, OSError) as e:
                        logger.debug(f"Skipped file rotation for {filepath} (lock busy): {e}")
                    finally:
                        with contextlib.suppress(OSError):
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception as e:
                logger.warning(f"Failed to rotate trajectory file {filepath}: {e}")

    def write_trajectory(
        self,
        atoms: Atoms,
        filepath: Path,
        step: int,
        state: SimulationState,
        force_publish: bool = False,
    ) -> None:
        """
        Writes frame to disk and pushes it to telemetry queue based on interval.
        High-uncertainty events can use force_publish=True to bypass downsampling.
        Implements basic file rotation if trajectory exceeds 500MB.
        """
        real_filepath = validate_path_safe(filepath)
        self._rotate_file_if_needed(real_filepath)

        def _write_task() -> None:
            import time

            max_retries = 3
            backoff_factor = 2.0
            delay = 0.1

            for attempt in range(1, max_retries + 1):
                try:
                    write(str(real_filepath), atoms, format="extxyz", append=True)
                except OSError as e:
                    if attempt == max_retries:
                        msg = f"Failed to write trajectory to {real_filepath} after {max_retries} attempts: {e}"
                        logger.exception(msg)
                        return

                    logger.warning(
                        f"I/O Error appending to {real_filepath}. Retrying in {delay}s (Attempt {attempt}/{max_retries})"
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
                except Exception:
                    logger.exception(
                        f"Unexpected error writing trajectory frame to {real_filepath}"
                    )
                    return
                else:
                    return

        self._executor.submit(_write_task)

        if force_publish or (step % self.stream_interval == 0):
            self._publish_frame(atoms, step, state)

    def _publish_frame(self, atoms: Atoms, step: int, state: SimulationState) -> None:
        try:
            positions = atoms.get_positions().flatten().tolist()  # type: ignore[no-untyped-call]

            forces = None
            if "forces" in atoms.arrays:
                forces = atoms.get_forces().flatten().tolist()  # type: ignore[no-untyped-call]

            variances = None
            if "c_gamma" in atoms.arrays:
                variances = atoms.get_array("c_gamma").flatten().tolist()  # type: ignore[no-untyped-call]

            frame = TelemetryFrame(
                workflow_id="default_workflow",
                step_number=step,
                current_state=state,
                positions=positions,
                forces=forces,
                variances=variances,
            )
            telemetry_broker.publish(cast("dict[str, Any]", frame))
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

    def _write_structure_memory(
        self, structure: Atoms, output_path: Path, elements: list[str]
    ) -> None:
        """Writes structure to disk using streaming writer if possible with atomic transactions."""
        import os

        real_output_path = validate_path_safe(output_path)

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

            # Atomic rename validation guarantees no zero-byte files are committed
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
