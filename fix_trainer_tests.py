import re
from pathlib import Path

# Add supported_formats to TrainingConfig schema
p = Path("src/pyacemaker/domain_models/training.py")
c = p.read_text()
if "supported_formats" not in c:
    c = c.replace('max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")', 'max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")\n    supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed formats")')
p.write_text(c)

# Remove the test execution log check script output since it was dumping failures
# Also fix the tests.
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()
# Ensure we don't accidentally override os in a bad way
if "import os" in test_c and "os.environ" in test_c:
    test_c = test_c.replace("os.environ['PACE_TRAIN_EXECUTABLE'] = 'echo'", "import os\nif 'PACE_TRAIN_EXECUTABLE' not in os.environ:\n    os.environ['PACE_TRAIN_EXECUTABLE'] = 'echo'")
    test_p.write_text(test_c)
