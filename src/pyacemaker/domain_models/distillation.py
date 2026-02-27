from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from pyacemaker.domain_models.active_learning import DescriptorConfig


class Step1DirectSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_points: PositiveInt = Field(default=100, description="Number of structures to generate")
    objective: str = Field(default="maximize_entropy", description="Objective function for sampling")
    descriptor: DescriptorConfig = Field(
        default_factory=lambda: DescriptorConfig(
            method="soap",
            species=["H"], # Default placeholder, should be overwritten by user
            r_cut=5.0,
            n_max=8,
            l_max=6,
            sigma=0.5
        ),
        description="Descriptor configuration for sampling"
    )


class Step2ActiveLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(default=0.8, gt=0.0, description="Uncertainty threshold for DFT calculation")
    n_active: PositiveInt = Field(default=10, description="Maximum number of structures to select for DFT")
    dft_calculator: str = Field(default="VASP", description="DFT code to use")


class Step3MaceFinetuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: str = Field(default="MACE-MP-0", description="Base MACE model to fine-tune")
    epochs: PositiveInt = Field(default=50, description="Number of fine-tuning epochs")


class Step4SurrogateSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_points: PositiveInt = Field(default=1000, description="Number of surrogate structures to generate")
    method: str = Field(default="mace_md", description="Method for surrogate sampling (e.g., mace_md)")


class Step7PacemakerFinetuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = Field(default=True, description="Enable Delta Learning phase")
    weight_dft: float = Field(default=10.0, gt=0.0, description="Weight of DFT data in loss function")


class DistillationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_mace_distillation: bool = Field(default=False, description="Enable the 7-step MACE distillation workflow")

    step1_direct_sampling: Step1DirectSamplingConfig = Field(
        default_factory=Step1DirectSamplingConfig, description="Step 1 configuration"
    )
    step2_active_learning: Step2ActiveLearningConfig = Field(
        default_factory=Step2ActiveLearningConfig, description="Step 2 configuration"
    )
    step3_mace_finetune: Step3MaceFinetuneConfig = Field(
        default_factory=Step3MaceFinetuneConfig, description="Step 3 configuration"
    )
    step4_surrogate_sampling: Step4SurrogateSamplingConfig = Field(
        default_factory=Step4SurrogateSamplingConfig, description="Step 4 configuration"
    )
    step7_pacemaker_finetune: Step7PacemakerFinetuneConfig = Field(
        default_factory=Step7PacemakerFinetuneConfig, description="Step 7 configuration"
    )

    @model_validator(mode="after")
    def validate_enabled_config(self) -> "DistillationConfig":
        """
        Validates that necessary configurations are sound when distillation is enabled.
        Since sub-configs have defaults, they are always present, but we can add
        cross-field validation here if needed.
        """
        if self.enable_mace_distillation:
            # Example: Ensure step 1 target points is reasonable
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
        return self
