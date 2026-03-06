import re
from pathlib import Path

# Add missing mock to our unit tests
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()

# We need to mock os.environ since our code looks up PACE_TRAIN_EXECUTABLE directly
test_c = test_c.replace(
    'def test_train_missing_executable(',
    '@pytest.fixture(autouse=True)\ndef mock_env(monkeypatch):\n    monkeypatch.setenv("PACE_TRAIN_EXECUTABLE", "pace_train")\n\ndef test_train_missing_executable('
)
test_p.write_text(test_c)
