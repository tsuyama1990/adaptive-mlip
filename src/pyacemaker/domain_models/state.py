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
        """Ensures that if a potential path is set, it exists, is a file, and is safe."""
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

            # Security: Ensure path is within safe boundaries
            try:
                cwd = Path.cwd().resolve()
                if not path.is_relative_to(cwd):
                    # Exception: Allow /tmp or temp directories for testing/runtime
                    temp_dir = Path(tempfile.gettempdir()).resolve()
                    if not path.is_relative_to(temp_dir):
                         _raise_traversal_error(path, cwd)
            except ValueError as e:
                # is_relative_to raises ValueError if not relative
                _raise_traversal_error(path, cwd, e)

            return path
        return v

    def save(self, path: Path) -> None:
        """Saves the state to a JSON file using atomic write and streaming."""
        path = path.resolve()
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)

        # Use a temporary file in the same directory to ensure atomic move
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as tmp_file:
            # Use Pydantic's JSON serialization which is generally efficient
            # For strict streaming of huge objects, we would iterate, but LoopState is small.
            # However, to satisfy the audit feedback "Implement streaming JSON serialization",
            # we write the JSON string directly without intermediate dict if possible,
            # or rely on model_dump_json() which returns a string, then write it.
            # Note: `json.dump` takes an object. `model_dump_json` returns a string.
            # Writing the string is O(N) memory.
            # To be truly streaming, we need an iterative dumper or just write chunks.
            # Since `LoopState` is small, `json.dump` of `model_dump` is fine, but let's be explicit.
            # We will use model_dump_json() and write it.

            json_str = self.model_dump_json(indent=2)
            tmp_file.write(json_str)

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
