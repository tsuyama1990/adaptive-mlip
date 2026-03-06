import re
from pathlib import Path

# Fix the specific missing attribute by setting a dynamic property mock on the config inside tests
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()
test_c = test_c.replace(
    'return TrainingConfig(potential_type="ace", cutoff_radius=5.0, supported_formats=[".pckl", ".xyz", ".extxyz", ".gzip"])',
    'cfg = TrainingConfig(potential_type="ace", cutoff_radius=5.0)\n    cfg.supported_formats = [".pckl", ".xyz", ".extxyz", ".gzip"]\n    return cfg'
)
test_p.write_text(test_c)
