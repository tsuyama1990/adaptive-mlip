import re
from pathlib import Path

# Instead of fighting Pydantic dynamic property addition, let's patch the underlying mock
# to just return the expected list when asked for `supported_formats` instead of setting it
# as an attribute.

test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()

# We replace the MockConfig definition to just correctly handle anything
fixture_replacement = """@pytest.fixture
def trainer():
    class MockConfig:
        def __init__(self):
            self.output_filename = "test_pot.yace"
            self.supported_formats = [".pckl", ".xyz", ".extxyz", ".gzip"]
            self.potential_type = "ace"
            self.cutoff_radius = 5.0
            self.max_basis_size = 2
            self.seed = 123
            self.max_iterations = 500
            self.batch_size = 20
            self.energy_weight = 1.0
            self.force_weight = 1.0
            self.stress_weight = 0.01
            self.display_step = 50
            self.delta_learning = True
            self.active_set_optimization = False
            self.active_set_size = None
        def model_copy(self):
            return self

    mock_cfg = MockConfig()

    t = PacemakerTrainer.__new__(PacemakerTrainer)
    t.config = mock_cfg # type: ignore
    t.config_generator = MagicMock()
    return t
"""

test_c = re.sub(
    r'@pytest\.fixture\ndef trainer\(\):[\s\S]*?return t\n',
    fixture_replacement,
    test_c
)

test_p.write_text(test_c)
