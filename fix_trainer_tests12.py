import re
from pathlib import Path

# Finally, let's fix the tests by correctly defining the Pydantic schema default value so it gets instantiated natively.
# Our previous fix modified `default_factory` to `default=` which triggers type issues.
# We should define `supported_formats` with a list explicitly in the test initialization!

test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()
# Return original mock config with the supported_formats directly.
test_c = test_c.replace(
    'return mock_config',
    'return TrainingConfig(potential_type="ace", cutoff_radius=5.0, supported_formats=[".pckl", ".xyz", ".extxyz", ".gzip"])'
)
test_p.write_text(test_c)

# Reset domain models back to proper format
p_train = Path("src/pyacemaker/domain_models/training.py")
train_c = p_train.read_text()
train_c = re.sub(
    r'supported_formats: list\[str\] = Field\(.*?\)',
    'supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed training data formats")',
    train_c
)
p_train.write_text(train_c)
