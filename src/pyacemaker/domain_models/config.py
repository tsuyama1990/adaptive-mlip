import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .dft import DFTConfig
from .eon import EONConfig
from .logging import LoggingConfig
from .md import MDConfig
from .scenario import ScenarioConfig
from .structure import StructureConfig
from .training import TrainingConfig
from .validation import ValidationConfig
from .workflow import WorkflowConfig


def _get_default_temp_dir() -> str:
    """Gets the system's preferred temporary directory for RAM disks."""
    _ram_disk_candidate = "/dev/shm"  # noqa: S108
    if Path(_ram_disk_candidate).exists():
        return _ram_disk_candidate
    return tempfile.gettempdir()


class GlobalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ram_disk_path: str = Field(
        default_factory=lambda: os.getenv("PYACEMAKER_TEMP_DIR", _get_default_temp_dir()),
        description="Path to the RAM disk or temp directory.",
    )
    lammps_velocity_seed: int = Field(
        default_factory=lambda: int(os.getenv("PYACEMAKER_LAMMPS_VELOCITY_SEED", "12345")),
        description="Seed for LAMMPS velocity initialization.",
    )
    md_base_energy: float = Field(
        default_factory=lambda: float(os.getenv("PYACEMAKER_MD_BASE_ENERGY", "-100.0")),
        description="Base energy for mock calculations.",
    )
    md_default_forces: list[list[float]] = Field(
        default=[[0.0, 0.0, 0.0]],
        description="Default forces for mock calculations.",
    )
    md_units: str = Field(
        default="metal",
        description="Default units for LAMMPS simulations.",
    )
    md_atom_style: str = Field(
        default="atomic",
        description="Default atom style for LAMMPS.",
    )
    md_neighbor_skin: float = Field(
        default=2.0,
        description="Default neighbor list skin distance.",
    )
    md_tdamp_factor: float = Field(
        default=100.0,
        description="Default temperature damping factor.",
    )
    md_pdamp_factor: float = Field(
        default=1000.0,
        description="Default pressure damping factor.",
    )


class PyAceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_name: str = Field(..., min_length=1, description="Name of the project")
    structure: StructureConfig
    dft: DFTConfig
    training: TrainingConfig
    md: MDConfig
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig, description="Validation configuration"
    )
    workflow: WorkflowConfig
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    eon: EONConfig | None = Field(None, description="EON configuration")
    scenario: ScenarioConfig | None = Field(None, description="Scenario configuration")
    api_cors_origins: list[str] = Field(
        default_factory=lambda: os.getenv("PYACEMAKER_CORS_ORIGINS", "http://localhost:3000").split(
            ","
        ),
        description="Allowed CORS origins for the API gateway",
    )
