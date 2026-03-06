import re
from pathlib import Path

# Finally, fixing the Pydantic test mock correctly without hitting Pydantic frozen limits.
# Just use standard object mocking for the config entirely in these specific tests.
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()

# We need to replace `return cfg` in the fixture with a fully mocked config since the real one throws errors when modified or checked
fixture_replacement = """    from unittest.mock import MagicMock
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
    return mock_config"""

test_c = re.sub(
    r'    from unittest\.mock import MagicMock\n    cfg = MagicMock\(\)\n    cfg\.output_filename = "test_pot\.yace"\n    cfg\.supported_formats = \[\"\.pckl\", \"\.xyz\", \"\.extxyz\", \"\.gzip\"\]\n    return cfg',
    fixture_replacement,
    test_c
)

test_p.write_text(test_c)
