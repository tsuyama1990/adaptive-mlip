from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyacemaker.domain_models.defaults import (
    DEFAULT_MD_DUMP_FREQ,
    DEFAULT_MD_HYBRID_ZBL_OUTER,
    DEFAULT_MD_N_STEPS,
    DEFAULT_MD_NEIGHBOR_SKIN,
    DEFAULT_MD_PDAMP,
    DEFAULT_MD_TDAMP,
    DEFAULT_MD_THERMO_FREQ,
    DEFAULT_MD_TIMESTEP,
)
from pyacemaker.utils.path import resolve_path


class HybridParams(BaseModel):
    """
    Parameters for the hybrid/overlay baseline potential (e.g., ZBL).
    """

    model_config = ConfigDict(extra="forbid")

    zbl_cutoffs: dict[tuple[str, str], float] = Field(
        default_factory=dict,
        description="Cutoff distances for ZBL (inner, outer) per element pair.",
    )
    # Re-introduced global cutoffs to match implementation needs
    zbl_global_cutoff: float = Field(
        DEFAULT_MD_HYBRID_ZBL_OUTER,
        description="Global outer cutoff for ZBL (Angstroms)"
    )


class MDConfig(BaseModel):
    """
    Configuration for Molecular Dynamics simulations.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(..., gt=0, description="Simulation temperature (K)")
    pressure: float = Field(0.0, ge=0.0, description="Simulation pressure (Bar)") # Typically positive or zero
    timestep: float = Field(DEFAULT_MD_TIMESTEP, gt=0, description="Timestep (ps)")
    n_steps: int = Field(DEFAULT_MD_N_STEPS, gt=0, description="Number of MD steps")
    thermo_freq: int = Field(
        DEFAULT_MD_THERMO_FREQ, gt=0, description="Frequency of thermodynamic output"
    )
    dump_freq: int = Field(
        DEFAULT_MD_DUMP_FREQ, gt=0, description="Frequency of trajectory dump"
    )
    uncertainty_threshold: float | None = Field(
        None, gt=0, description="Max allowed extrapolation grade"
    )
    check_interval: int | None = Field(
        None, gt=0, description="Interval for uncertainty check"
    )
    # Hybrid potential settings
    hybrid_potential: bool = Field(
        False, description="Use hybrid/overlay potential (ACE + ZBL/LJ)"
    )
    hybrid_params: HybridParams = Field(
        default_factory=HybridParams, description="Parameters for hybrid potential"
    )
    neighbor_skin: float = Field(
        DEFAULT_MD_NEIGHBOR_SKIN, gt=0, description="Neighbor list skin distance"
    )
    tdamp_factor: float = Field(
        DEFAULT_MD_TDAMP, gt=0, description="Thermostat damping factor (x timestep)"
    )
    pdamp_factor: float = Field(
        DEFAULT_MD_PDAMP, gt=0, description="Barostat damping factor (x timestep)"
    )
    fix_halt: bool = Field(
        False, description="Enable fix halt for uncertainty-driven early termination"
    )
    minimize: bool = Field(True, description="Perform minimization before MD run")
    potential_path: str | None = Field(None, description="Path to potential file")

    @field_validator("potential_path", mode="before")
    @classmethod
    def validate_potential_path(cls, v: Any) -> Any:
        if v is not None:
            return str(resolve_path(v))
        return v


class MDSimulationResult(BaseModel):
    """
    Result of an MD simulation.
    """

    model_config = ConfigDict(extra="forbid")

    energy: float
    forces: list[list[float]]  # List of [fx, fy, fz]
    stress: list[float]  # Voigt notation [pxx, pyy, pzz, pyz, pxz, pxy]
    halted: bool
    max_gamma: float
    n_steps: int
    temperature: float
    trajectory_path: str
    log_path: str
    halt_structure_path: str | None = None
    halt_step: int | None = None
