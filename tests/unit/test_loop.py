from pathlib import Path

from pyacemaker.core.loop import LoopState, LoopStatus


def test_loop_state_initialization() -> None:
    state = LoopState()
    assert state.iteration == 0
    assert state.status == LoopStatus.RUNNING
    assert state.current_potential is None


def test_loop_state_save_load(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    pot_path = tmp_path / "pot.yace"
    state = LoopState(iteration=5, status=LoopStatus.HALTED, current_potential=pot_path)

    state.save(state_file)
    assert state_file.exists()

    loaded_state = LoopState.load(state_file)
    assert loaded_state.iteration == 5
    assert loaded_state.status == LoopStatus.HALTED
    assert loaded_state.current_potential == pot_path


def test_loop_state_load_non_existent(tmp_path: Path) -> None:
    state_file = tmp_path / "non_existent.json"
    state = LoopState.load(state_file)
    assert state.iteration == 0
    assert state.status == LoopStatus.RUNNING
    assert state.current_potential is None
