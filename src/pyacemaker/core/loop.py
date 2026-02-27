import json
import os
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.domain_models.workflow import WorkflowStep
from pyacemaker.utils.path import validate_path_safe


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
        Uses centralized path validation utility.
        """
        if v is not None:
            # The centralized utility handles symlink prevention, bounds checking, etc.
            # However, LoopState requires it to exist as a file if it's set.
            # validate_path_safe just ensures the path is safe to use.
            safe_path = validate_path_safe(Path(v))

            if not safe_path.is_file():
                msg = f"Potential path is not a file: {safe_path}"
                raise ValueError(msg)

            return safe_path
        return v

    def save(self, path: Path) -> None:
        """Saves the state to a JSON file using atomic write."""
        path = validate_path_safe(path)
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=True)

        # LoopState is small, so loading into memory for dump is acceptable.
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
        safe_path = validate_path_safe(path)
        if not safe_path.exists():
            return cls()

        try:
            with safe_path.open("r") as f:
                # Streaming load is automatic with json.load(f)
                data = json.load(f)
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to load loop state from {safe_path}: {e}"
            raise ValueError(msg) from e
