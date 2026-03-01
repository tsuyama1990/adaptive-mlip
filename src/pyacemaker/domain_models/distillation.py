from pydantic import BaseModel, Field


class DistillationConfig(BaseModel):
    """Phase 1: Zero-Shot Distillation configuration"""

    enable: bool = True
    mace_model_path: str = "mace-mp-0-medium"
    uncertainty_threshold: float = Field(0.05, description="MACE confidence threshold")
    sampling_structures_per_system: int = 1000
