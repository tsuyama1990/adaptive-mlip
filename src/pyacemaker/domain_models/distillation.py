from pydantic import BaseModel, ConfigDict, Field


class ActiveLearningThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold_call_dft: float = Field(
        0.05, description="The system-wide max uncertainty required to halt MD."
    )
    threshold_add_train: float = Field(
        0.02,
        description="The per-atom uncertainty required to include an atom in the local learning set (the epicenter).",
    )
    smooth_steps: int = Field(
        3,
        ge=1,
        description="The number of consecutive steps the threshold must be exceeded to trigger a halt (thermal noise exclusion).",
    )


class CutoutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    core_radius: float = Field(
        4.0, ge=0.0, description="Radius in Angstroms around the epicenter where force_weight=1.0."
    )
    buffer_radius: float = Field(
        3.0, ge=0.0, description="Additional radius for the boundary layer (force_weight=0.0)."
    )
    enable_pre_relaxation: bool = Field(
        True, description="Whether to relax the buffer layer using MACE."
    )
    enable_passivation: bool = Field(True, description="Whether to auto-passivate dangling bonds.")
    passivation_element: str = Field(
        "H", min_length=1, description="The element to use for passivation."
    )


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(True, description="Enable Phase 1 (Zero-Shot Distillation).")
    mace_model_path: str = Field(
        "mace-mp-0-medium", description="Path or name of the foundational MACE model."
    )
    uncertainty_threshold: float = Field(
        0.05, ge=0.0, description="The threshold below which MACE is considered confident."
    )
    sampling_structures_per_system: int = Field(
        1000, ge=1, description="Number of structures to sample per system."
    )


class LoopStrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_tiered_oracle: bool = Field(True, description="Use a two-tier oracle strategy.")
    incremental_update: bool = Field(True, description="Enable incremental delta learning.")
    replay_buffer_size: int = Field(
        500, ge=0, description="Size of the replay buffer for incremental updates."
    )
    baseline_potential_type: str = Field("LJ", description="Type of the baseline potential.")
    thresholds: ActiveLearningThresholds = Field(
        default_factory=ActiveLearningThresholds, description="Active learning thresholds."
    )
