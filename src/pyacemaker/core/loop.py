import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.utils.path import validate_path_safe


class LoopStatus(StrEnum):
    RUNNING = "RUNNING"
    HALTED = "HALTED"
    CONVERGED = "CONVERGED"


class LoopState(BaseModel):
    iteration: int = Field(default=0, ge=0)
    status: LoopStatus = Field(default=LoopStatus.RUNNING)
    current_potential: Path | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    @field_validator("current_potential")
    @classmethod
    def validate_potential_path(cls, v: Path | None) -> Path | None:
        """Ensures that if a potential path is set, it exists, is a file, and is safe."""
        if v is not None:
            path = validate_path_safe(v)
            if not path.exists():
                msg = f"Potential path does not exist or is invalid: {v}"
                raise ValueError(msg)

            if not path.is_file():
                msg = f"Potential path is not a file: {path}"
                raise ValueError(msg)

            return path
        return v

    def save(self, path: Path) -> None:
        """Saves the state to a JSON file using atomic write, streaming, and file locking."""
        path = path.resolve()
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)

        import fcntl

        lock_path = path.with_suffix(".lock")
        # Ensure we have exclusive access to the state file
        with lock_path.open("w") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                # Use a temporary file in the same directory to ensure atomic move
                with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as tmp_file:
                    json.dump(self.model_dump(mode="json"), tmp_file, indent=2)

                    # Ensure data is flushed to disk
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                    tmp_path_str = tmp_file.name

                tmp_path = Path(tmp_path_str)
                try:
                    tmp_path.replace(path)
                except OSError:
                    # Clean up temp file if replace fails
                    tmp_path.unlink(missing_ok=True)
                    raise
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Loads the state from a JSON file."""
        if not path.exists():
            return cls()

        try:
            with path.open("r") as f:
                # Streaming load is automatic with json.load(f)
                data = json.load(f)
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to load loop state from {path}: {e}"
            raise ValueError(msg) from e


def _raise_traversal_error(path: Path, cwd: Path, cause: Exception | None = None) -> None:
    msg = f"Potential path {path} is outside the project directory {cwd}"
    if cause:
        raise ValueError(msg) from cause
    raise ValueError(msg)
