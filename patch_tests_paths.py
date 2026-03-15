import os
import re

with open("src/pyacemaker/domain_models/workflow.py", "r") as f:
    content = f.read()

# We need to allow absolute paths for pytest tmp_path fixtures in testing, but strictly block them in production?
# Actually, the memory says: "In tests involving security validations (e.g., file path boundaries), do not use patch() to mock the validation function. Instead, use monkeypatch.setattr() to redirect base configuration directories to pytest's tmp_path and perform actual file system interactions, including tests for path traversal and symlink attacks."
# Wait, `validate_workflow_paths` blocks absolute paths because `val.startswith("/")`.
# But `pytest.tmp_path` is always absolute!
# Let's check `tests/e2e/test_orchestrator.py` to see how it passes these configs.
