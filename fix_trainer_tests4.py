import re
from pathlib import Path

# Add mock supported_formats to TrainingConfig fixtures directly where needed, as Pydantic models with `extra="forbid"` don't like missing fields during dynamic testing.
# Or we can fix the Pydantic schema default initialization.

test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()

# We need to make the tests pass by mocking the config properly
test_c = test_c.replace(
    'return TrainingConfig(potential_type="ace", cutoff_radius=5.0)',
    'return TrainingConfig(potential_type="ace", cutoff_radius=5.0, supported_formats=[".pckl", ".xyz", ".extxyz", ".gzip"])'
)
test_p.write_text(test_c)
