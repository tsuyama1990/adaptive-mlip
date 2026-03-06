import re
from pathlib import Path

# Instead of messing with Pydantic properties dynamically, which keeps failing,
# Let's revert src/pyacemaker/core/trainer.py to just not enforce suffix checking against config if it's too much of a pain to mock.
# Wait, no, we must follow the architecture fix: "Move to external configuration with validation."
# Let's just fix the mock properly using simple dictionary or mock object if needed, or better,
# just use the real domain_models defaults.

test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()
test_c = test_c.replace(
    '''    from unittest.mock import MagicMock
    mock_config = MagicMock()
    mock_config.potential_type = "ace"
    mock_config.cutoff_radius = 5.0
    mock_config.supported_formats = [".pckl", ".xyz", ".extxyz", ".gzip"]
    mock_config.output_filename = "test_pot.yace"
    mock_config.max_basis_size = 2
    mock_config.seed = 123
    mock_config.max_iterations = 500
    mock_config.batch_size = 20
    mock_config.energy_weight = 1.0
    mock_config.force_weight = 1.0
    mock_config.stress_weight = 0.01
    mock_config.display_step = 50
    mock_config.delta_learning = True
    mock_config.active_set_optimization = False
    mock_config.active_set_size = None
    return mock_config''',
    '''    return TrainingConfig(potential_type="ace", cutoff_radius=5.0)'''
)
test_p.write_text(test_c)

# We define supported_formats right in TrainingConfig as a fixed attribute with a proper default factory
p_train = Path("src/pyacemaker/domain_models/training.py")
train_c = p_train.read_text()

# Completely replace the class block for TrainingConfig to ensure it works
replacement = """class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    potential_type: str = Field(..., description="Type of potential to train")
    cutoff_radius: float = Field(..., gt=0, description="Cutoff radius in Angstroms")
    max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")
    supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed training data formats")
"""
train_c = re.sub(r'class TrainingConfig\(BaseModel\):[\s\S]*?max_basis_size: PositiveInt = Field\(.*?description="Maximum basis size"\)', replacement, train_c)

# Also fix the fact that we might have added `supported_formats` multiple times previously due to script regex
train_c = re.sub(r'    supported_formats: list\[str\].*\n(    supported_formats: list\[str\].*\n)+', '    supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed training data formats")\n', train_c)

p_train.write_text(train_c)
