import subprocess
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
    with patch("subprocess.run") as mock_run, \
         patch("pyacemaker.core.active_set.read") as mock_read, \
         patch("pyacemaker.core.active_set.write"), \
         patch.object(Path, "exists", return_value=True), \
         patch("pyacemaker.core.active_set.Path.stat") as mock_stat:

        # Configure mock_run
        mock_run.return_value = MagicMock(returncode=0, stdout="Selected 5 structures")

        # Mock file non-empty check
        mock_stat.return_value.st_size = 100

        # Configure mock_read to return a subset
        mock_read.return_value = candidates[:5]

        # Test with generator
        cand_gen = (c for c in candidates)
        selected = selector.select(cand_gen, potential_path="dummy.yace", n_select=5)

        assert len(selected) == 5
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "pace_activeset" in args
        assert str(5) in args

def test_select_active_set_invalid_n_select(selector: ActiveSetSelector, candidates: list[Atoms]) -> None:
    with pytest.raises(ValueError, match="n_select must be positive"):
        selector.select(candidates, "dummy.yace", n_select=0)

def test_select_active_set_subprocess_fail(selector: ActiveSetSelector, candidates: list[Atoms]) -> None:
     with patch("subprocess.run") as mock_run, \
          patch("pyacemaker.core.active_set.write"), \
          patch.object(Path, "exists", return_value=True), \
          patch("pyacemaker.core.active_set.Path.stat") as mock_stat:

        mock_stat.return_value.st_size = 100
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with pytest.raises(ActiveSetError, match="Active set selection failed"):
            selector.select(candidates, "dummy.yace", n_select=5)
