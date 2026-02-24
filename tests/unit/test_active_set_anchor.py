from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.exceptions import ActiveSetError


def test_select_with_anchor_basic(tmp_path: Path) -> None:
    selector = ActiveSetSelector()
    candidates = [Atoms('H') for _ in range(10)]
    anchor = Atoms('He', positions=[[0,0,0]])
    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()

    # Mock run_command to simulate pace_activeset output
    # pace_activeset should be called with n_select = 4 (since total 5 requested)
    with patch("pyacemaker.core.active_set.run_command") as mock_run:
        # Side effect creates output file
        def side_effect(cmd: list[str]) -> MagicMock:
            out_idx = cmd.index("--output")
            out_path = Path(cmd[out_idx + 1])
            # Write 4 atoms
            from ase.io import write
            write(out_path, [Atoms('Li') for _ in range(4)], format="extxyz")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        result = list(selector.select(candidates, pot_path, n_select=5, anchor=anchor))

        # Check total count
        assert len(result) == 5

        # Check first is anchor
        assert result[0].get_chemical_symbols()[0] == 'He'  # type: ignore[no-untyped-call]

        # Check remaining are from mock output
        assert result[1].get_chemical_symbols()[0] == 'Li'  # type: ignore[no-untyped-call]

        # Check cmd arguments
        args, _ = mock_run.call_args
        cmd = args[0]
        select_idx = cmd.index("--select")
        assert cmd[select_idx + 1] == "4" # 5 - 1 = 4

def test_select_with_anchor_only_one(tmp_path: Path) -> None:
    selector = ActiveSetSelector()
    candidates = [Atoms('H') for _ in range(10)]
    anchor = Atoms('He')
    pot_path = tmp_path / "dummy.yace"
    pot_path.touch()

    # If n_select=1 and anchor provided, run_command should NOT be called
    with patch("pyacemaker.core.active_set.run_command") as mock_run:
        result = list(selector.select(candidates, pot_path, n_select=1, anchor=anchor))

        assert len(result) == 1
        assert result[0].get_chemical_symbols()[0] == 'He'  # type: ignore[no-untyped-call]
        mock_run.assert_not_called()

def test_validate_path_resolution_safety() -> None:
    """Test that paths are resolved to absolute, mitigating simple flag injection and traversal."""
    selector = ActiveSetSelector()
    # These should NOT raise because they resolve to safe absolute paths
    selector._validate_path_safe(Path("foo/../bar"))
    selector._validate_path_safe(Path("-rf"))

def test_validate_path_invalid_chars() -> None:
    """Test that path validation rejects shell metacharacters."""
    selector = ActiveSetSelector()
    # % is now invalid
    with pytest.raises(ActiveSetError, match="Path contains invalid characters"):
        selector._validate_path_safe(Path("foo%20bar"))

    # ; is invalid
    with pytest.raises(ActiveSetError, match="Path contains invalid characters"):
        selector._validate_path_safe(Path("foo;bar"))
