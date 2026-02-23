import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.io import write

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.exceptions import ActiveSetError


@pytest.fixture
def selector() -> ActiveSetSelector:
    return ActiveSetSelector()

@pytest.fixture
def candidates() -> list[Atoms]:
    return [Atoms('H', positions=[[0, 0, 0]]) for _ in range(20)]

def test_select_active_set_io(selector: ActiveSetSelector, candidates: list[Atoms], tmp_path: Path) -> None:
    """Test I/O operations without mocking read/write."""

    # We define a side effect for subprocess.run to create the output file
    # AND verify the input file exists while we are "running"
    def subprocess_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        # args[0] is cmd list
        cmd = args[0]
        try:
            # Verify input file exists
            ds_idx = cmd.index("--dataset")
            cand_file = Path(cmd[ds_idx + 1])
            assert cand_file.exists(), "Candidate file passed to subprocess does not exist!"
            assert cand_file.stat().st_size > 0, "Candidate file is empty!"

            # Create the output file (simulating success)
            out_idx = cmd.index("--output")
            out_path = Path(cmd[out_idx + 1])
            write(out_path, candidates[:5], format="extxyz")
        except ValueError:
            pass # Should verify args elsewhere if needed
        except AssertionError as e:
            # Propagate assertion error out of side effect
            raise e

        return MagicMock(returncode=0)

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess_side_effect

        # We need a dummy potential file
        pot_path = tmp_path / "dummy.yace"
        pot_path.touch()

        selected = selector.select(candidates, pot_path, n_select=5)

        assert len(selected) == 5

        # Verify subprocess was called
        mock_run.assert_called_once()

        # We checked file existence inside side_effect, so we know it worked.


def test_select_active_set_invalid_n_select(selector: ActiveSetSelector, candidates: list[Atoms]) -> None:
    with pytest.raises(ValueError, match="n_select must be positive"):
        selector.select(candidates, "dummy.yace", n_select=0)

def test_select_active_set_subprocess_fail(selector: ActiveSetSelector, candidates: list[Atoms], tmp_path: Path) -> None:
     pot_path = tmp_path / "dummy.yace"
     pot_path.touch()

     with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with pytest.raises(ActiveSetError, match="Active set selection failed"):
            selector.select(candidates, pot_path, n_select=5)

def test_active_set_path_validation(selector: ActiveSetSelector) -> None:
    """Test internal path validation."""
    # We can test private method or trigger via select

    bad_path = Path("path/with/bad/&/char")

    with pytest.raises(ActiveSetError, match="Path contains invalid characters"):
        selector._validate_path_safe(bad_path)
