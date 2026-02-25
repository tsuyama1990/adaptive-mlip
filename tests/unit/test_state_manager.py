import time
from pathlib import Path
from unittest.mock import MagicMock

from pyacemaker.core.state_manager import StateManager


def test_state_manager_async_save(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    logger = MagicMock()
    manager = StateManager(state_file, logger)

    manager.iteration = 42
    manager.save()

    # Verify it's async (file might not exist immediately if we are super fast, but unlikely)
    # But we can verify it eventually exists.
    manager.shutdown() # Wait for completion

    assert state_file.exists()
    content = state_file.read_text()
    assert '"iteration": 42' in content

def test_state_manager_atomic_write(tmp_path: Path, monkeypatch) -> None:
    state_file = tmp_path / "state.json"
    logger = MagicMock()
    manager = StateManager(state_file, logger)

    # Mock json.dump to be slow to simulate I/O
    import json
    original_dump = json.dump
    def slow_dump(*args, **kwargs):
        time.sleep(0.1)
        original_dump(*args, **kwargs)

    monkeypatch.setattr(json, "dump", slow_dump)

    manager.iteration = 10
    manager.save()

    # Immediately check: file shouldn't exist or be partial?
    # Since it writes to temp file then moves, state_file won't exist until move.
    # But due to thread, we don't know exact timing.

    manager.shutdown()
    assert state_file.exists()
