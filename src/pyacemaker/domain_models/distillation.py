from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from pyacemaker.domain_models.active_learning import DescriptorConfig
from pyacemaker.domain_models.defaults import DEFAULT_BATCH_SIZE, DEFAULT_CANDIDATE_MULTIPLIER


class Step1DirectSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    target_points: PositiveInt = Field(default=100, description="Number of structures to generate")
    candidate_multiplier: PositiveInt = Field(
        default=DEFAULT_CANDIDATE_MULTIPLIER,
        description="Multiplier for initial candidate pool size"
    )
    # Replaced loose str validation with Literal string to enforce Type Safety / Enum natively via Pydantic
    objective: Literal["maximize_entropy", "random"] = Field(
        default="maximize_entropy",
        description="Objective function for sampling"
    )
    batch_size: PositiveInt = Field(default=DEFAULT_BATCH_SIZE, description="Batch size for processing")
    descriptor: DescriptorConfig = Field(..., description="Descriptor configuration for sampling")


class Step2ActiveLearningConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    uncertainty_threshold: float = Field(default=0.8, gt=0.0, description="Uncertainty threshold for DFT calculation")
    n_active: PositiveInt = Field(default=10, description="Maximum number of structures to select for DFT")
    dft_calculator: str = Field(default="VASP", description="DFT code to use")
    batch_size: PositiveInt = Field(default=DEFAULT_BATCH_SIZE, description="Batch size for processing")


class Step3MaceFinetuneConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    base_model: str = Field(default="MACE-MP-0", description="Base MACE model to fine-tune")
    epochs: PositiveInt = Field(default=50, description="Number of fine-tuning epochs")


class Step4SurrogateSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    target_points: PositiveInt = Field(default=1000, description="Number of surrogate structures to generate")
    method: str = Field(default="mace_md", description="Method for surrogate sampling (e.g., mace_md)")


class Step7PacemakerFinetuneConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enable: bool = Field(default=True, description="Enable Delta Learning phase")
    weight_dft: float = Field(default=10.0, gt=0.0, description="Weight of DFT data in loss function")


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enable_mace_distillation: bool = Field(default=False, description="Enable the 7-step MACE distillation workflow")

    step1_direct_sampling: Step1DirectSamplingConfig | None = Field(
        default=None,
        description="Step 1 configuration"
    )
    step2_active_learning: Step2ActiveLearningConfig | None = Field(
        default=None, description="Step 2 configuration"
    )
    step3_mace_finetune: Step3MaceFinetuneConfig | None = Field(
        default=None, description="Step 3 configuration"
    )
    step4_surrogate_sampling: Step4SurrogateSamplingConfig | None = Field(
        default=None, description="Step 4 configuration"
    )
    step7_pacemaker_finetune: Step7PacemakerFinetuneConfig | None = Field(
        default=None, description="Step 7 configuration"
    )

    @model_validator(mode="after")
    def validate_enabled_config(self) -> "DistillationConfig":
        """
        Validates that necessary configurations are sound when distillation is enabled.
        Also validates that when disabled, optional fields aren't inexplicably provided.
        """
        if self.enable_mace_distillation:
            if self.step1_direct_sampling is None:
                 raise ValueError("Step 1 configuration is required when distillation is enabled.")
            if self.step2_active_learning is None:
                 raise ValueError("Step 2 configuration is required when distillation is enabled.")
            if self.step3_mace_finetune is None:
                 raise ValueError("Step 3 configuration is required when distillation is enabled.")

            # Validate Step 1
            if self.step1_direct_sampling.target_points < 10:
                msg = "Step 1 target points must be at least 10."
                raise ValueError(msg)

            # Validate DFT calculator
            valid_calculators = {"VASP", "QE", "MOCK"}
            calc = self.step2_active_learning.dft_calculator.upper()
            if calc not in valid_calculators:
                msg = f"Invalid DFT calculator: {calc}. Must be one of {valid_calculators}"
                raise ValueError(msg)

            # Validate MACE model
            if not self.step3_mace_finetune.base_model.strip():
                raise ValueError("Step 3 base model cannot be empty.")
        # If distillation is disabled, prevent setting these configurations to avoid confusion.
        elif any(step is not None for step in [
            self.step1_direct_sampling,
            self.step2_active_learning,
            self.step3_mace_finetune,
            self.step4_surrogate_sampling,
            self.step7_pacemaker_finetune
        ]):
             raise ValueError("Distillation step configs must be None when enable_mace_distillation is False")

        return self
