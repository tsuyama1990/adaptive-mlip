import os
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from pyacemaker.domain_models.constants import (
    DEFAULT_MC_SEED,
    DEFAULT_MD_MINIMIZE_FTOL,
    DEFAULT_MD_MINIMIZE_TOL,
    DEFAULT_RAM_DISK_PATH,
    LAMMPS_MINIMIZE_MAX_ITER,
    LAMMPS_MINIMIZE_STEPS,
    LAMMPS_VELOCITY_SEED,
    MAX_MD_DURATION,
    MAX_MD_PRESSURE,
)
from pyacemaker.domain_models.defaults import (
    DEFAULT_MD_ATOM_STYLE,
    DEFAULT_MD_BASE_ENERGY,
    DEFAULT_MD_CHECK_INTERVAL,
    DEFAULT_MD_DUMP_FREQ,
    DEFAULT_MD_HYBRID_ZBL_INNER,
    DEFAULT_MD_HYBRID_ZBL_OUTER,
    DEFAULT_MD_NEIGHBOR_SKIN,
    DEFAULT_MD_PDAMP_FACTOR,
    DEFAULT_MD_TDAMP_FACTOR,
    DEFAULT_MD_THERMO_FREQ,
    DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
)


def _get_default_temp_dir() -> str | None:
    """Returns RAM disk path if available and writable, else None."""
    shm_path = Path(DEFAULT_RAM_DISK_PATH)
    if shm_path.exists() and shm_path.is_dir() and os.access(shm_path, os.W_OK):
        return str(shm_path)
    return None


class AtomStyle(str, Enum):
    ATOMIC = "atomic"
    CHARGE = "charge"
    FULL = "full"


class HybridParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zbl_cut_inner: PositiveFloat = Field(
        DEFAULT_MD_HYBRID_ZBL_INNER, description="Inner cutoff radius for ZBL potential (Angstrom)"
    )
    zbl_cut_outer: PositiveFloat = Field(
        DEFAULT_MD_HYBRID_ZBL_OUTER, description="Outer cutoff radius for ZBL potential (Angstrom)"
    )


class MDRampingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temp_start: float | None = Field(None, ge=0.0, description="Starting temperature (K)")
    temp_end: float | None = Field(None, ge=0.0, description="Ending temperature (K)")
    press_start: float | None = Field(None, ge=0.0, description="Starting pressure (Bar)")
    press_end: float | None = Field(None, ge=0.0, description="Ending pressure (Bar)")

    @model_validator(mode="after")
    def validate_ramping(self) -> "MDRampingConfig":
        return self


class MCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    swap_freq: int = Field(..., gt=0, description="Frequency of MC swaps (steps)")
    swap_prob: float = Field(..., gt=0.0, le=1.0, description="Probability of swapping atoms")
    seed: int = Field(DEFAULT_MC_SEED, description="Random seed for MC swaps")


class MDSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    energy: float = Field(..., description="Final potential energy of the system")
    forces: list[list[float]] = Field(..., description="Forces on atoms in the final frame")
    stress: list[float] = Field(
        default_factory=lambda: [0.0] * 6,
        description="Stress tensor (Voigt: xx, yy, zz, yz, xz, xy) in Bar"
    )
    halted: bool = Field(..., description="Whether the simulation was halted early")
    max_gamma: float = Field(..., description="Maximum extrapolation grade observed")
    n_steps: int = Field(..., description="Number of steps actually performed")
    temperature: float = Field(..., description="Average or final temperature")
    trajectory_path: str | None = Field(None, description="Path to the trajectory file")
    log_path: str | None = Field(None, description="Path to the simulation log file")
    halt_structure_path: str | None = Field(
        None, description="Path to the structure where halt occurred"
    )
    halt_step: int | None = Field(None, description="The step at which the simulation was halted")

    @model_validator(mode="after")
    def validate_physical_values(self) -> "MDSimulationResult":
        # Validate energy is finite
        if not np.isfinite(self.energy):
            raise ValueError("Energy must be a finite number")

        # Validate forces shape and values
        for f in self.forces:
            if len(f) != 3:
                raise ValueError("Forces must be 3D vectors (list of 3 floats)")
            if not np.isfinite(f).all():
                raise ValueError("Forces must contain finite numbers")

        # Validate stress
        if len(self.stress) != 6:
             raise ValueError("Stress must be a 6-element list (Voigt notation)")
        if not np.isfinite(self.stress).all():
             raise ValueError("Stress must contain finite numbers")

        return self


class MDConfig(BaseModel):
    """
    Configuration for Molecular Dynamics simulations.
    """
    model_config = ConfigDict(extra="forbid")

    # Basic Physics
    temperature: float = Field(..., ge=0.0, description="Simulation temperature in Kelvin")
    pressure: float = Field(..., ge=0.0, le=MAX_MD_PRESSURE, description="Simulation pressure in Bar")
    timestep: PositiveFloat = Field(..., gt=0.0, le=10.0, description="Timestep in ps")
    n_steps: int = Field(..., gt=0, le=1000000000, description="Number of MD steps")

    # Output Control
    thermo_freq: PositiveInt = Field(
        DEFAULT_MD_THERMO_FREQ, description="Frequency of thermodynamic output (steps)"
    )
    dump_freq: PositiveInt = Field(
        DEFAULT_MD_DUMP_FREQ, description="Frequency of trajectory dump (steps)"
    )
    minimize: bool = Field(False, description="Perform energy minimization before MD")
    neighbor_skin: PositiveFloat = Field(
        DEFAULT_MD_NEIGHBOR_SKIN, description="Neighbor list skin distance (Angstrom)"
    )
    atom_style: AtomStyle = Field(
        AtomStyle(DEFAULT_MD_ATOM_STYLE), description="LAMMPS atom style"
    )

    # Configurable LAMMPS Parameters (No Hardcoding)
    velocity_seed: int = Field(
        LAMMPS_VELOCITY_SEED, description="Random seed for velocity initialization"
    )
    minimize_steps: int = Field(
        LAMMPS_MINIMIZE_STEPS, description="Max iterations for minimization (steps)"
    )
    minimize_max_iter: int = Field(
        LAMMPS_MINIMIZE_MAX_ITER, description="Max force evaluations for minimization"
    )
    minimize_tol: float = Field(
        DEFAULT_MD_MINIMIZE_TOL, description="Energy tolerance for minimization"
    )
    minimize_ftol: float = Field(
        DEFAULT_MD_MINIMIZE_FTOL, description="Force tolerance for minimization"
    )
    # Added field for minimization style (defaulting to cg)
    minimize_style: str = Field(
        "cg", description="LAMMPS minimization style (e.g., 'cg', 'hftn', 'sd')"
    )

    # Advanced Settings
    temp_dir: str | None = Field(
        default_factory=_get_default_temp_dir,
        description="Directory for temporary files (e.g., /dev/shm for RAM disk)"
    )
    tdamp_factor: float = Field(
        DEFAULT_MD_TDAMP_FACTOR, gt=0.0, description="Temperature damping factor (multiplies timestep)"
    )
    pdamp_factor: float = Field(
        DEFAULT_MD_PDAMP_FACTOR, gt=0.0, description="Pressure damping factor (multiplies timestep)"
    )

    # Mocking Parameters (Audit Requirement)
    base_energy: float = Field(
        DEFAULT_MD_BASE_ENERGY, description="Baseline energy for mock simulation"
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
        DEFAULT_OTF_UNCERTAINTY_THRESHOLD, gt=0.0, description="Gamma threshold for halting simulation"
    )
    check_interval: int = Field(
        DEFAULT_MD_CHECK_INTERVAL, gt=0, description="Step interval for uncertainty check"
    )

    # Spec Section 3.1: Ramping and MC
    ramping: MDRampingConfig | None = Field(None, description="Configuration for T/P ramping")
    mc: MCConfig | None = Field(None, description="Configuration for Monte Carlo atom swapping")

    @model_validator(mode="after")
    def validate_simulation_physics(self) -> "MDConfig":
        total_time = self.n_steps * self.timestep
        if total_time > MAX_MD_DURATION:
             raise ValueError(f"Total simulation time {total_time} ps exceeds maximum {MAX_MD_DURATION} ps.")
        if self.temperature < 0.0 or self.temperature > 10000.0:
             raise ValueError("Temperature must be between 0 and 10000 K.")
        return self

    @model_validator(mode="after")
    def validate_otf_settings(self) -> "MDConfig":
        if self.fix_halt and self.check_interval <= 0:
             raise ValueError("check_interval must be positive when fix_halt is enabled.")
        return self

    @model_validator(mode="after")
    def validate_temp_dir(self) -> "MDConfig":
        if self.temp_dir:
            p = Path(self.temp_dir)
            if not p.exists() or not os.access(p, os.W_OK):
                raise ValueError(f"Temporary directory {p} does not exist or is not writable.")
        return self

    @model_validator(mode="after")
    def validate_default_forces(self) -> "MDConfig":
        for f in self.default_forces:
            if len(f) != 3:
                raise ValueError("Default forces must be a list of 3D vectors (list of 3 floats)")
            if not all(isinstance(x, (int, float)) for x in f):
                 raise ValueError("Default forces elements must be numeric")
        return self
