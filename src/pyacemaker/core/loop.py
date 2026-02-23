import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        """Ensures that if a potential path is set, it exists and is a file."""
        if v is not None:
            # Resolve to absolute path to prevent traversal/ambiguity
            try:
                path = Path(v).resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as e:
                # strict=True raises FileNotFoundError if it doesn't exist
                msg = f"Potential path does not exist or is invalid: {v}"
                raise ValueError(msg) from e

            if not path.is_file():
                msg = f"Potential path is not a file: {path}"
                raise ValueError(msg)

            # Additional check: ensure it is not just a root directory or sensitive path?
            # Without a chroot/sandbox config, we assume the user has permissions.
            return path
        return v

    def save(self, path: Path) -> None:
        """Saves the state to a JSON file using atomic write."""
        data = self.model_dump(mode="json")
        path = path.resolve()

        # Use a temporary file in the same directory to ensure atomic move
        # (os.replace is atomic on POSIX)
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as tmp_file:
            json.dump(data, tmp_file, indent=2)
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

    @classmethod
    def load(cls, path: Path) -> Self:
        """Loads the state from a JSON file."""
        if not path.exists():
            return cls()

        # Safe loading: Read strictly necessary data
        try:
            with path.open("r") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            # If corrupted, raise ValueError
            msg = f"Failed to load loop state from {path}: {e}"
            raise ValueError(msg) from e
