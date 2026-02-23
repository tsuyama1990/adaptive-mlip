import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from pyacemaker.domain_models.constants import DEFAULT_RAM_DISK_PATH


def _get_default_temp_dir() -> str | None:
    """Returns RAM disk path if available and writable, else None."""
    shm_path = Path(DEFAULT_RAM_DISK_PATH)
    if shm_path.exists() and shm_path.is_dir() and os.access(shm_path, os.W_OK):
        return str(shm_path)
    return None


class HybridParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zbl_cut_inner: PositiveFloat = Field(
        2.0, description="Inner cutoff radius for ZBL potential (Angstrom)"
    )
    zbl_cut_outer: PositiveFloat = Field(
        2.5, description="Outer cutoff radius for ZBL potential (Angstrom)"
    )


class MDSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy: float = Field(..., description="Final potential energy of the system")
    forces: list[list[float]] = Field(..., description="Forces on atoms in the final frame")
    halted: bool = Field(..., description="Whether the simulation was halted early")
    max_gamma: float = Field(..., description="Maximum extrapolation grade observed")
    n_steps: int = Field(..., description="Number of steps actually performed")
    temperature: float = Field(..., description="Average or final temperature")
    trajectory_path: str | None = Field(None, description="Path to the trajectory file")
    log_path: str | None = Field(None, description="Path to the simulation log file")
    halt_structure_path: str | None = Field(
        None, description="Path to the structure where halt occurred"
    )


class MDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: PositiveFloat = Field(..., description="Simulation temperature in Kelvin")
    pressure: float = Field(..., ge=0.0, description="Simulation pressure in Bar")
    timestep: PositiveFloat = Field(..., description="Timestep in ps")
    n_steps: int = Field(..., gt=0, description="Number of MD steps")

    # Output Control
    thermo_freq: PositiveInt = Field(
        10, description="Frequency of thermodynamic output (steps)"
    )
    dump_freq: PositiveInt = Field(
        100, description="Frequency of trajectory dump (steps)"
    )
    minimize: bool = Field(False, description="Perform energy minimization before MD")
    neighbor_skin: PositiveFloat = Field(
        2.0, description="Neighbor list skin distance (Angstrom)"
    )
    atom_style: str = Field(
        "atomic", description="LAMMPS atom style (e.g. atomic, charge)"
    )

    # Advanced Settings
    temp_dir: str | None = Field(
        default_factory=_get_default_temp_dir,
        description="Directory for temporary files (e.g., /dev/shm for RAM disk)"
    )
    tdamp_factor: PositiveFloat = Field(
        100.0, description="Temperature damping factor (multiplies timestep)"
    )
    pdamp_factor: PositiveFloat = Field(
        1000.0, description="Pressure damping factor (multiplies timestep)"
    )

    # Mocking Parameters (Audit Requirement)
    base_energy: float = Field(
        -100.0, description="Baseline energy for mock simulation"
    )
    default_forces: list[list[float]] = Field(
        default=[[0.0, 0.0, 0.0]], description="Default forces for mock simulation"
    )

    # Spec Section 3.4 (Hybrid Potential & OTF)
    hybrid_potential: bool = Field(
        False, description="Use hybrid potential (ACE + LJ/ZBL)"
    )
    hybrid_params: HybridParams = Field(
        default_factory=HybridParams, description="Parameters for hybrid potential baseline"
    )

    # Spec Section 3.4 (OTF)
    fix_halt: bool = Field(
        False, description="Enable OTF halting based on uncertainty"
    )
    uncertainty_threshold: float = Field(
        5.0, gt=0.0, description="Gamma threshold for halting simulation"
    )
    check_interval: int = Field(
        10, gt=0, description="Step interval for uncertainty check"
    )
