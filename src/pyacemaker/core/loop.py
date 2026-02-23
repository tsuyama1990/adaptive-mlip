import json
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class LoopStatus(StrEnum):
    RUNNING = "RUNNING"
    HALTED = "HALTED"
    CONVERGED = "CONVERGED"


class LoopState(BaseModel):
    iteration: int = Field(default=0, ge=0)
    status: LoopStatus = Field(default=LoopStatus.RUNNING)
    current_potential: Path | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    def save(self, path: Path) -> None:
        """Saves the state to a JSON file."""
        data = self.model_dump(mode="json")
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Loads the state from a JSON file."""
        if not path.exists():
            return cls()
        with path.open("r") as f:
            data = json.load(f)
        return cls.model_validate(data)
