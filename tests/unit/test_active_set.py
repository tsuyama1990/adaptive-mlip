from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.exceptions import ActiveSetError


@pytest.fixture
def selector() -> ActiveSetSelector:
    return ActiveSetSelector()

@pytest.fixture
def candidates() -> list[Atoms]:
    return [Atoms('H', positions=[[0, 0, 0]]) for _ in range(20)]

def test_select_active_set_mocked_subprocess(selector: ActiveSetSelector, candidates: list[Atoms]) -> None:
    # Mock subprocess.run to simulate pace_activeset
    # The output of pace_activeset is typically a file containing indices or the structures themselves
    # Here we assume it returns indices of selected structures

    # We need to mock how the selector interacts with the external tool
    # Assume selector writes candidates to a file, runs command, reads result

    with patch("subprocess.run") as mock_run, \
         patch("pyacemaker.core.active_set.read") as mock_read, \
         patch("pyacemaker.core.active_set.write"), \
         patch.object(Path, "exists", return_value=True):

        # Configure mock_run
        mock_run.return_value = MagicMock(returncode=0, stdout="Selected 5 structures")

        # Configure mock_read to return a subset
        mock_read.return_value = candidates[:5]

        selected = selector.select(candidates, potential_path="dummy.yace", n_select=5)

        assert len(selected) == 5
        mock_run.assert_called_once()
        # Verify command arguments roughly
        args = mock_run.call_args[0][0]
        assert "pace_activeset" in args
        assert str(5) in args # n_select should be passed

def test_select_active_set_failure(selector: ActiveSetSelector, candidates: list[Atoms]) -> None:
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("pace_activeset failed")

        with pytest.raises(ActiveSetError):
            selector.select(candidates, potential_path="dummy.yace", n_select=5)
