import re
from pathlib import Path

# Add mock property explicitly to the config fixture to make ALL Pydantic mock objects pass
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()

# Provide standard default mock property addition that does not trigger Pydantic __setattr__ extra checks
test_c = test_c.replace(
    'cfg = TrainingConfig(potential_type="ace", cutoff_radius=5.0)\n    cfg.supported_formats = [".pckl", ".xyz", ".extxyz", ".gzip"]\n    return cfg',
    'cfg = TrainingConfig(potential_type="ace", cutoff_radius=5.0)\n    object.__setattr__(cfg, "supported_formats", [".pckl", ".xyz", ".extxyz", ".gzip"])\n    return cfg'
)

# Fix tests
test_p.write_text(test_c)
