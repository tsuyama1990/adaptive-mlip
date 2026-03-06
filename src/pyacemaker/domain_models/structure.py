from enum import StrEnum

from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExplorationPolicy(StrEnum):
    COLD_START = "cold_start"
    RANDOM_RATTLE = "random_rattle"
    STRAIN = "strain"
    DEFECTS = "defects"


class LocalGenerationStrategy(StrEnum):
    RANDOM_DISPLACEMENT = "random_displacement"
    NORMAL_MODE = "normal_mode"
    MD_MICRO_BURST = "md_micro_burst"


class StrainMode(StrEnum):
    VOLUME = "volume"
    SHEAR = "shear"
    MIXED = "mixed"


class StructureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[str] = Field(
        ..., min_length=1, description="List of elements in the system"
    )
    supercell_size: list[int] = Field(
        ..., min_length=3, max_length=3, description="Supercell size [nx, ny, nz]"
    )

    # Exploration Policy Configuration
    # Refactored to support multiple policies (Composite Policy)
    active_policies: list[ExplorationPolicy] = Field(
        default=[ExplorationPolicy.COLD_START],
        description="List of exploration policies to apply sequentially"
    )

    # Deprecated single policy field, kept for compatibility.
    # It syncs with the first element of active_policies.
    policy_name: ExplorationPolicy | None = Field(
        default=None,
        description="Deprecated: Use active_policies instead."
    )

    rattle_stdev: float = Field(
        default=0.1, ge=0.0, description="Standard deviation for random rattle (Angstrom)"
    )
    strain_mode: StrainMode = Field(
        default=StrainMode.VOLUME, description="Mode for strain application"
    )
    strain_magnitude: float = Field(
        default=0.05, ge=0.0, description="Magnitude of strain to apply (e.g., 0.05 for 5%)"
    )
    vacancy_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Rate of vacancies to introduce"
    )
    num_structures: int = Field(
        default=1, ge=1, description="Number of structures to generate per policy"
    )

    # Local Active Learning Settings
    local_generation_strategy: LocalGenerationStrategy = Field(
        default=LocalGenerationStrategy.RANDOM_DISPLACEMENT,
        description="Strategy for generating local candidates around halt structures"
    )
    local_extraction_radius: float = Field(
        default=6.0, gt=0.0, description="Radius for extracting local clusters around high uncertainty atoms (Angstrom)"
    )
    local_buffer_radius: float = Field(
        default=4.0, ge=0.0, description="Buffer radius added to extraction for force masking (Angstrom)"
    )
    local_md_steps: int = Field(
        default=50, gt=0, description="Number of MD steps for local micro-burst generation"
    )
    local_md_temp: float = Field(
        default=2000.0, gt=0.0, description="Temperature for local micro-burst MD (K)"
    )

    @field_validator("elements")
    @classmethod
    def validate_elements(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "Elements list cannot be empty"
            raise ValueError(msg)

        # Check for duplicates
        if len(v) != len(set(v)):
             msg = "Elements list cannot contain duplicates"
             raise ValueError(msg)

        valid_symbols = set(chemical_symbols)
        for el in v:
            if el not in valid_symbols:
                msg = f"Invalid chemical symbol: {el}"
                raise ValueError(msg)
        return v

    @field_validator("supercell_size")
    @classmethod
    def validate_supercell(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            msg = "Supercell dimensions must be positive integers"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def sync_policy_fields(self) -> "StructureConfig":
        """Syncs policy_name and active_policies for backward compatibility."""
        if self.policy_name is not None and self.policy_name not in self.active_policies:
            # If policy_name is set (e.g. from config file), ensure it's in active_policies
            # Overwrite active_policies to respect legacy config
            self.active_policies = [self.policy_name]

        # Ensure policy_name is set for legacy readers
        if self.active_policies and self.policy_name is None:
            self.policy_name = self.active_policies[0]

        return self
