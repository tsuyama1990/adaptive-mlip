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
    return [Atoms("H", positions=[[0, 0, 0]]) for _ in range(20)]


def test_select_active_set_io_streaming_batched(
    selector: ActiveSetSelector, candidates: list[Atoms], tmp_path: Path
) -> None:
    """Test I/O operations with batching."""

    # Mock run_command to create output file
    def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
        # Create output file
        out_idx = cmd.index("--output")
        out_path = Path(cmd[out_idx + 1])
        # write only 5 atoms
        write(out_path, candidates[:5], format="extxyz")
        return MagicMock(returncode=0)

    with patch("pyacemaker.core.active_set.run_command") as mock_run:
        mock_run.side_effect = side_effect

        pot_path = tmp_path / "dummy.yace"
        pot_path.touch()

        # select() returns iterator
        # Test large batch if needed, but 20 < BATCH_SIZE(1000) so just checks basic flow
        result_iter = selector.select(candidates, pot_path, n_select=5)

        # Consume iterator
        selected = list(result_iter)

        assert len(selected) == 5
        mock_run.assert_called_once()


def test_select_active_set_write_fail(
    selector: ActiveSetSelector, candidates: list[Atoms], tmp_path: Path
) -> None:
    """Test handling of write failure (e.g. permission error simulated by exception)."""

    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()

    with patch("pyacemaker.core.active_set.write") as mock_write:
        mock_write.side_effect = OSError("Disk full")

        with pytest.raises(ActiveSetError, match="Failed to write candidates"):
            list(selector.select(candidates, pot_path, n_select=5))


def test_select_active_set_partial_failure(
    selector: ActiveSetSelector, candidates: list[Atoms], tmp_path: Path
) -> None:

    def side_effect(cmd: list[str], **kwargs: Any) -> MagicMock:
        out_idx = cmd.index("--output")
        out_path = Path(cmd[out_idx + 1])
        out_path.touch()  # Create empty file
        return MagicMock(returncode=0)

    with patch("pyacemaker.core.active_set.run_command") as mock_run:
        mock_run.side_effect = side_effect

        pot_path = tmp_path / "dummy.yace"
        pot_path.touch()

        with pytest.raises(ActiveSetError, match="Output file is empty"):
            list(selector.select(candidates, pot_path, n_select=5))


def test_select_active_set_process_fail(
    selector: ActiveSetSelector, candidates: list[Atoms], tmp_path: Path
) -> None:
    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()

    with patch("pyacemaker.core.active_set.run_command") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr="error")

        with pytest.raises(ActiveSetError, match="Active set execution failed"):
            list(selector.select(candidates, pot_path, n_select=5))


def test_active_set_path_validation_strict(selector: ActiveSetSelector) -> None:
    bad_path = Path("path/with/;/injection")
    with pytest.raises(ActiveSetError, match="Path contains invalid characters"):
        selector._validate_path_safe(bad_path)
