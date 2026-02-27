import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.domain_models.workflow import WorkflowStep


class LoopStatus(StrEnum):
    RUNNING = "RUNNING"
    HALTED = "HALTED"
    CONVERGED = "CONVERGED"


class LoopState(BaseModel):
    iteration: int = Field(default=0, ge=0)
    status: LoopStatus = Field(default=LoopStatus.RUNNING)
    current_potential: Path | None = Field(default=None)
    current_step: WorkflowStep | None = Field(
        default=None, description="Current step in distillation workflow"
    )
    mode: str = Field(
        default="legacy", description="Workflow mode: 'legacy' or 'distillation'"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        from pyacemaker.domain_models.defaults import (
            WORKFLOW_MODE_DISTILLATION,
            WORKFLOW_MODE_LEGACY,
        )
        if v not in (WORKFLOW_MODE_LEGACY, WORKFLOW_MODE_DISTILLATION):
             msg = f"Invalid mode: {v}. Must be one of: {WORKFLOW_MODE_LEGACY}, {WORKFLOW_MODE_DISTILLATION}"
             raise ValueError(msg)
        return v

    @field_validator("current_potential")
    @classmethod
    def validate_potential_path(cls, v: Path | None) -> Path | None:
        """
        Ensures that if a potential path is set, it exists, is a file, and is safe.

        Security:
        - Strictly disallows symbolic links to prevent aliasing/masking of targets.
        - Resolves path to canonical absolute form.
        - Verifies that the resolved path is within the allowed project boundaries (CWD or Temp).

        Args:
            v: The path to validate.

        Returns:
            The validated, resolved Path object, or None.

        Raises:
            ValueError: If path is invalid, unsafe, or a symlink.
        """
        if v is not None:
            # Strict security: Disallow symbolic links to prevent potential ambiguity or traversal tricks
            # even before resolution.
            if Path(v).is_symlink():
                msg = f"Potential path cannot be a symbolic link: {v}"
                raise ValueError(msg)

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
        """Saves the state to a JSON file using atomic write."""
        path = path.resolve()
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)

        # LoopState is small, so loading into memory for dump is acceptable.
        # For larger datasets, we would stream.
        data = self.model_dump(mode="json")

        # Use a temporary file in the same directory to ensure atomic move
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

        try:
            with path.open("r") as f:
                # Streaming load is automatic with json.load(f)
                data = json.load(f)
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to load loop state from {path}: {e}"
            raise ValueError(msg) from e


def _raise_traversal_error(path: Path, base: Path, cause: Exception | None = None) -> None:
    """Raises a ValueError indicating path traversal attempt. Inputs should be resolved."""
    msg = f"Potential path {path} is outside the allowed directory {base}"
    if cause:
        raise ValueError(msg) from cause
    raise ValueError(msg)
