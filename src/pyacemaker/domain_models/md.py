
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt


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
    uncertainty_threshold: float = Field(
        5.0, gt=0.0, description="Gamma threshold for halting simulation"
    )
    check_interval: int = Field(
        10, gt=0, description="Step interval for uncertainty check"
    )
