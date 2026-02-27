from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyacemaker.core.exceptions import ConfigError
from pyacemaker.core.state_manager import StateManager


def test_state_manager_path_validation() -> None:
    # Test path traversal prevention
    mock_logger = MagicMock()
    with pytest.raises(ValueError, match=".*outside.*"): # validate_path_safe raises ValueError or PermissionError
        StateManager(Path("/etc/passwd"), mock_logger)

def test_state_manager_property_removal() -> None:
    # Verify that properties were removed and state is accessed directly
    mock_logger = MagicMock()
    # Using a valid temp path to avoid validation error
    manager = StateManager(Path("state.json"), mock_logger)

    # This should work now
    manager.state.iteration = 5
    assert manager.state.iteration == 5

    # The old way should raise AttributeError
    with pytest.raises(AttributeError):
        manager.iteration = 6 # type: ignore[attr-defined]
