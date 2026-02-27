from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Step1DirectSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_points: PositiveInt = Field(default=100, description="Number of structures to generate")
    objective: str = Field(default="maximize_entropy", description="Objective function for sampling")


class Step2ActiveLearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uncertainty_threshold: float = Field(default=0.8, gt=0.0, description="Uncertainty threshold for DFT calculation")
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
