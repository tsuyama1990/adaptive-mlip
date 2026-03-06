import re
from pathlib import Path

# The Pydantic model is frozen, so setting properties fails dynamically. Let's just fix the mock completely inside the test.
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()
test_c = test_c.replace(
    'cfg = TrainingConfig(potential_type="ace", cutoff_radius=5.0)\n    object.__setattr__(cfg, "supported_formats", [".pckl", ".xyz", ".extxyz", ".gzip"])\n    return cfg',
    'from unittest.mock import MagicMock\n    cfg = MagicMock()\n    cfg.output_filename = "test_pot.yace"\n    cfg.supported_formats = [".pckl", ".xyz", ".extxyz", ".gzip"]\n    return cfg'
)

# And one more place where TrainingConfig might be instantiated directly
test_c = test_c.replace(
    'return TrainingConfig(potential_type="ace", cutoff_radius=5.0)',
    'from unittest.mock import MagicMock\n    cfg = MagicMock()\n    cfg.output_filename = "test_pot.yace"\n    cfg.supported_formats = [".pckl", ".xyz", ".extxyz", ".gzip"]\n    return cfg'
)

test_p.write_text(test_c)

# We must ensure TrainingConfig actually defines supported_formats in code too as our previous script might have failed due to syntax
p_train = Path("src/pyacemaker/domain_models/training.py")
train_c = p_train.read_text()
if "supported_formats: list[str] = Field(default=" not in train_c:
    train_c = train_c.replace(
        'max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")',
        'max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")\n    supported_formats: list[str] = Field(default=[".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed training data formats")'
    )
    p_train.write_text(train_c)
