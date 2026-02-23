import os
import tempfile
from pathlib import Path

import pytest

from pyacemaker.core.loop import LoopState, LoopStatus


def test_loop_state_initialization() -> None:
    state = LoopState()
    assert state.iteration == 0
    assert state.status == LoopStatus.RUNNING
    assert state.current_potential is None


def test_loop_state_save_load(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()
    state = LoopState(iteration=5, status=LoopStatus.HALTED, current_potential=pot_path)

    state.save(state_file)
    assert state_file.exists()

    loaded_state = LoopState.load(state_file)
    assert loaded_state.iteration == 5
    assert loaded_state.status == LoopStatus.HALTED
    assert loaded_state.current_potential == pot_path.resolve()


def test_loop_state_load_non_existent(tmp_path: Path) -> None:
    state_file = tmp_path / "non_existent.json"
    state = LoopState.load(state_file)
    assert state.iteration == 0
    assert state.status == LoopStatus.RUNNING
    assert state.current_potential is None


def test_loop_state_validation_path_not_exists(tmp_path: Path) -> None:
    """Test validation fails if potential path does not exist."""
    pot_path = tmp_path / "missing.yace"
    with pytest.raises(ValueError, match="Potential path does not exist"):
        LoopState(current_potential=pot_path)


def test_loop_state_validation_path_is_dir(tmp_path: Path) -> None:
    """Test validation fails if potential path is a directory."""
    pot_dir = tmp_path / "pot_dir"
    pot_dir.mkdir()
    with pytest.raises(ValueError, match="Potential path is not a file"):
        LoopState(current_potential=pot_dir)


def test_loop_state_validation_path_traversal(tmp_path: Path) -> None:
    """Test validation fails if potential path is outside project root."""
    # Create project structure
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    unsafe_file = outside_dir / "unsafe.yace"
    unsafe_file.touch()

    # We need to change CWD to project_dir for the test to work
    cwd = Path.cwd()
    os.chdir(project_dir)
    try:
        # Mock tempfile.gettempdir to force the check to fail even if in /tmp
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(tempfile, "gettempdir", lambda: "/nonexistent_temp")

            with pytest.raises(ValueError, match="outside the project directory"):
                LoopState(current_potential=unsafe_file)
    finally:
        os.chdir(cwd)


def test_loop_state_corrupted_load(tmp_path: Path) -> None:
    """Test that loading a corrupted JSON raises ValueError."""
    state_file = tmp_path / "corrupted.json"
    state_file.write_text("{invalid_json")

    with pytest.raises(ValueError, match="Failed to load loop state"):
        LoopState.load(state_file)
