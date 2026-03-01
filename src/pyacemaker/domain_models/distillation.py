from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class DistillationConfig(BaseModel):
    """Configuration for Phase 1: Zero-Shot Distillation & Baseline Construction."""
    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=True, description="Enable distillation phase")
    mace_model_path: str = Field(..., description="Path to the MACE foundation model")
    uncertainty_threshold: float = Field(
        default=0.1, gt=0, description="Threshold for MACE confidence filtering"
    )
    sampling_counts: PositiveInt = Field(
        default=1000, description="Number of structures to extract via DIRECT sampling"
    )
