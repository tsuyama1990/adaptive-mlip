import os
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from pyacemaker.domain_models.constants import (
    DEFAULT_MC_SEED,
    DEFAULT_RAM_DISK_PATH,
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
    DEFAULT_MD_MINIMIZE_FTOL,
    DEFAULT_MD_MINIMIZE_TOL,
    DEFAULT_LAMMPS_MINIMIZE_MAX_ITER,
    DEFAULT_LAMMPS_MINIMIZE_STEPS,
    DEFAULT_LAMMPS_VELOCITY_SEED,
    DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
    MAX_MD_STEPS,
)
from pyacemaker.domain_models.env import safe_env_float, safe_env_int


def _get_default_temp_dir() -> str | None:
    """Returns RAM disk path if available and writable, else None."""
    from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS

    shm_path = Path(DEFAULT_RAM_DISK_PATH)
    if any(char in str(shm_path) for char in DANGEROUS_PATH_CHARS):
        return None
    if shm_path.exists() and shm_path.is_dir() and os.access(shm_path, os.W_OK):
        return str(shm_path)
    return None


class AtomStyle(StrEnum):
    ATOMIC = "atomic"
    CHARGE = "charge"
    FULL = "full"


class ZBLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    zbl_cut_inner: PositiveFloat = Field(
        default=DEFAULT_MD_HYBRID_ZBL_INNER,
        description="Inner cutoff radius for ZBL potential (Angstrom)",
    )
    zbl_cut_outer: PositiveFloat = Field(
        default=DEFAULT_MD_HYBRID_ZBL_OUTER,
        description="Outer cutoff radius for ZBL potential (Angstrom)",
    )

    @model_validator(mode="after")
    def validate_zbl_cutoffs(self) -> "ZBLConfig":
        if self.zbl_cut_inner >= self.zbl_cut_outer:
            msg = "zbl_cut_inner must be strictly less than zbl_cut_outer"
            raise ValueError(msg)
        return self


class MDRampingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    temp_start: float | None = Field(None, ge=0.0, description="Starting temperature (K)")
    temp_end: float | None = Field(None, ge=0.0, description="Ending temperature (K)")
    press_start: float | None = Field(None, ge=0.0, description="Starting pressure (Bar)")
    press_end: float | None = Field(None, ge=0.0, description="Ending pressure (Bar)")

    @model_validator(mode="after")
    def validate_ramping(self) -> "MDRampingConfig":
        if (
            self.temp_start is not None
            and self.temp_end is not None
            and self.temp_start > self.temp_end
        ):
            msg = "temp_start must be less than or equal to temp_end"
            raise ValueError(msg)
        return self


class MCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    swap_freq: int = Field(..., gt=0, description="Frequency of MC swaps (steps)")
    swap_prob: float = Field(..., gt=0.0, le=1.0, description="Probability of swapping atoms")
    seed: int = Field(DEFAULT_MC_SEED, description="Random seed for MC swaps")


class MDSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    energy: float = Field(..., description="Final potential energy of the system")
    forces: list[list[float]] = Field(..., description="Forces on atoms in the final frame")
    stress: list[float] = Field(
        default_factory=lambda: [0.0] * 6,
        description="Stress tensor (Voigt: xx, yy, zz, yz, xz, xy) in Bar",
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
            msg = "Energy must be a finite number"
            raise ValueError(msg)

        # Validate forces shape and values
        for f in self.forces:
            if len(f) != 3:
                msg = "Forces must be 3D vectors (list of 3 floats)"
                raise ValueError(msg)
            if not np.isfinite(f).all():
                msg = "Forces must contain finite numbers"
                raise ValueError(msg)

        # Validate stress
        if len(self.stress) != 6:
            msg = "Stress must be a 6-element list (Voigt notation)"
            raise ValueError(msg)
        if not np.isfinite(self.stress).all():
            msg = "Stress must contain finite numbers"
            raise ValueError(msg)

        return self


class MDConfig(BaseModel):
    """
    Configuration for Molecular Dynamics simulations.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # Basic Physics
    temperature: float = Field(..., ge=0.0, description="Simulation temperature in Kelvin")
    pressure: float = Field(
        ..., ge=0.0, le=MAX_MD_PRESSURE, description="Simulation pressure in Bar"
    )
    timestep: PositiveFloat = Field(..., gt=0.0, description="Timestep in ps")
    n_steps: int = Field(..., gt=0, le=MAX_MD_STEPS, description="Number of MD steps")

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
    units: str = Field("metal", description="LAMMPS unit style")
    atom_style: AtomStyle = Field(AtomStyle(DEFAULT_MD_ATOM_STYLE), description="LAMMPS atom style")

    # Configurable LAMMPS Parameters (No Hardcoding)
    velocity_seed: int = Field(
        default_factory=lambda: safe_env_int(
            "PYACEMAKER_LAMMPS_VELOCITY_SEED", DEFAULT_LAMMPS_VELOCITY_SEED
        ),
        description="Random seed for velocity initialization",
    )
    minimize_steps: int = Field(
        default_factory=lambda: safe_env_int(
            "PYACEMAKER_LAMMPS_MINIMIZE_STEPS", DEFAULT_LAMMPS_MINIMIZE_STEPS
        ),
        description="Max iterations for minimization (steps)",
    )
    minimize_max_iter: int = Field(
        default_factory=lambda: safe_env_int(
            "PYACEMAKER_LAMMPS_MINIMIZE_MAX_ITER", DEFAULT_LAMMPS_MINIMIZE_MAX_ITER
        ),
        description="Max force evaluations for minimization",
    )
    minimize_tol: float = Field(
        default_factory=lambda: safe_env_float(
            "PYACEMAKER_MD_MINIMIZE_TOL", DEFAULT_MD_MINIMIZE_TOL
        ),
        description="Energy tolerance for minimization",
    )
    minimize_ftol: float = Field(
        default_factory=lambda: safe_env_float(
            "PYACEMAKER_MD_MINIMIZE_FTOL", DEFAULT_MD_MINIMIZE_FTOL
        ),
        description="Force tolerance for minimization",
    )

    # Advanced Settings
    temp_dir: str | None = Field(
        default_factory=_get_default_temp_dir,
        description="Directory for temporary files (e.g., /dev/shm for RAM disk)",
    )
    tdamp_factor: float = Field(
        DEFAULT_MD_TDAMP_FACTOR,
        gt=0.0,
        description="Temperature damping factor (multiplies timestep)",
    )
    pdamp_factor: float = Field(
        DEFAULT_MD_PDAMP_FACTOR, gt=0.0, description="Pressure damping factor (multiplies timestep)"
    )

    # Mocking Parameters (Audit Requirement)
    base_energy: float = Field(
        default_factory=lambda: safe_env_float("PYACEMAKER_MD_BASE_ENERGY", DEFAULT_MD_BASE_ENERGY),
        description="Baseline energy for mock simulation",
    )
    default_forces: list[list[float]] = Field(
        default_factory=lambda: [[0.0, 0.0, 0.0]], description="Default forces for mock simulation"
    )

    # Spec Section 3.4 (Hybrid Potential & OTF)
    hybrid_potential: bool = Field(False, description="Use hybrid potential (ACE + LJ/ZBL)")
    zbl: ZBLConfig = Field(
        default_factory=ZBLConfig, description="ZBL Potential specific parameters"
    )

    # Spec Section 3.4 (OTF)
    fix_halt: bool = Field(False, description="Enable OTF halting based on uncertainty")
    uncertainty_threshold: float = Field(
        DEFAULT_OTF_UNCERTAINTY_THRESHOLD,
        gt=0.0,
        description="Gamma threshold for halting simulation",
    )
    check_interval: int = Field(
        DEFAULT_MD_CHECK_INTERVAL, gt=0, description="Step interval for uncertainty check"
    )

    # Spec Section 3.1: Ramping and MC
    ramping: MDRampingConfig | None = Field(None, description="Configuration for T/P ramping")
    mc: MCConfig | None = Field(None, description="Configuration for Monte Carlo atom swapping")

    # Cycle 05: Seamless Resume Soft Start (Thermalization)
    soft_start_steps: int = Field(
        0, ge=0, description="Steps for strong Langevin thermalization upon resume"
    )
    soft_start_langevin_damp: float = Field(
        0.1, gt=0.0, description="Damping parameter (ps) for soft start Langevin thermostat"
    )

    # Cycle 03: Spatial Tagging
    custom_initialization_commands: list[str] | None = Field(
        default=None, description="Custom LAMMPS commands generated for initialization"
    )

    @model_validator(mode="after")
    def validate_simulation_physics(self) -> "MDConfig":
        from pyacemaker.domain_models.defaults import DEFAULT_MAX_TIMESTEP

        if self.timestep > DEFAULT_MAX_TIMESTEP:
            msg = f"Timestep {self.timestep} ps exceeds maximum {DEFAULT_MAX_TIMESTEP} ps"
            raise ValueError(msg)

        total_time = self.n_steps * self.timestep
        if total_time > MAX_MD_DURATION:
            msg = f"Total time {total_time} ps exceeds maximum {MAX_MD_DURATION} ps"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_otf_settings(self) -> "MDConfig":
        if self.fix_halt and self.check_interval <= 0:
            msg = "check_interval must be positive when fix_halt is enabled."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_custom_initialization_commands(self) -> "MDConfig":
        if self.custom_initialization_commands:
            import re
            from pyacemaker.domain_models.constants import LAMMPS_SAFE_CMD_PATTERN

            pattern = re.compile(LAMMPS_SAFE_CMD_PATTERN)
            for cmd in self.custom_initialization_commands:
                if not pattern.match(cmd):
                    msg = f"Invalid spatial command generated: {cmd}"
                    raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_temp_dir(self) -> "MDConfig":
        if self.temp_dir:
            p = Path(self.temp_dir)
            if not p.exists() or not os.access(p, os.W_OK):
                msg = f"Temporary directory {p} does not exist or is not writable."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_default_forces(self) -> "MDConfig":
        for f in self.default_forces:
            if len(f) != 3:
                msg = "Default forces must be a list of 3D vectors (list of 3 floats)"
                raise ValueError(msg)
            if not all(isinstance(x, (int, float)) for x in f):
                msg = "Default forces elements must be numeric"
                raise ValueError(msg)
        return self
