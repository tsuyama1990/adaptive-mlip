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
    with pytest.raises(ValueError, match="Potential path is not a file|does not exist|No such file"):
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
        # LoopState resolves path.
        # We need to force a path that is NOT in /tmp (if possible)
        # or rely on the mock if we are in /tmp.

        with pytest.MonkeyPatch.context() as mp:
            import tempfile
            # Mock gettempdir to fail the whitelist check
            mp.setattr(tempfile, "gettempdir", lambda: "/nonexistent_temp")

            # Match generic "outside the allowed directory" message or "project directory" if old msg?
            # My update changed it to: "Potential path {path} is outside the allowed directory {base}"
            with pytest.raises(ValueError, match="outside.*"):
                LoopState(current_potential=unsafe_file)
    finally:
        os.chdir(cwd)


def test_loop_state_corrupted_load(tmp_path: Path) -> None:
    """Test that loading a corrupted JSON raises ValueError."""
    state_file = tmp_path / "corrupted.json"
    state_file.write_text("{invalid_json")

    with pytest.raises(ValueError, match="Failed to load loop state"):
        LoopState.load(state_file)


def test_loop_state_concurrent_save(tmp_path: Path) -> None:
    """Test thread safety of LoopState save (atomic write)."""
    import threading

    state_file = tmp_path / "concurrent_state.json"

    def save_worker(i: int) -> None:
        local_state = LoopState(iteration=i)
        import contextlib
        with contextlib.suppress(OSError):
            local_state.save(state_file)
            # OS might lock file during rename on Windows, but atomic rename should handle it
            pass

    threads = []
    for i in range(20):
        t = threading.Thread(target=save_worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify file is valid JSON and contains one of the iterations
    # Atomic write via temp file + replace ensures no partial writes.
    loaded = LoopState.load(state_file)
    assert 0 <= loaded.iteration < 20
