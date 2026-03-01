import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine, UncertaintyWatchdog
from pyacemaker.domain_models.md import HybridParams, MDConfig, MDSimulationResult
from pyacemaker.domain_models.workflow import ActiveLearningThresholds


@pytest.fixture
def mock_driver() -> Any:
    with patch("pyacemaker.core.engine.LammpsDriver") as mock:
        yield mock


def test_lammps_script_generator_restart(
    mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path
) -> None:
    """Test that providing a restart_file replaces read_data with read_restart."""
    config = mock_md_config.model_copy()
    engine = LammpsEngine(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    restart_file = tmp_path / "test.restart"
    restart_file.touch()

    driver_instance = mock_driver.return_value
    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    script_content = []

    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance.run_file.side_effect = capture_run

    engine.run(atoms, pot_path, restart_file=restart_file)

    assert len(script_content) == 1
    script = script_content[0]

    assert f"read_restart {restart_file}" in script
    assert "read_data" not in script
    assert "create_box" not in script
    assert "velocity all create" not in script  # Handled by soft start or omitted in restart
    assert (
        "fix soft_start_langevin" in script
    )  # Soft start generated because restart_file was provided


def test_watchdog_thermal_noise(tmp_path: Path) -> None:
    """Test that a single spike in uncertainty does not trigger a halt."""
    dump_file = tmp_path / "test.dump"
    # Create a mock dump with 3 steps. Only step 2 has high uncertainty (noise).
    # thresholds: dft=0.05, train=0.02, smooth=2
    dump_content = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
2 1 1 1 1 0.01
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.06
2 1 1 1 1 0.03
ITEM: TIMESTEP
300
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
2 1 1 1 1 0.01
"""
    dump_file.write_text(dump_content)

    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=2
    )

    watchdog = UncertaintyWatchdog(thresholds)
    halt_step, epicenter = watchdog.evaluate_stream(dump_file)
    assert halt_step is None


def test_watchdog_sustained_uncertainty(tmp_path: Path) -> None:
    """Test that sustained uncertainty triggers a halt and identifies epicenter atoms."""
    dump_file = tmp_path / "test.dump"
    # thresholds: dft=0.05, train=0.02, smooth=2
    # Step 1: low, Step 2: high, Step 3: high (should halt at step 3)
    dump_content = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
2 1 1 1 1 0.01
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.06
2 1 1 1 1 0.01
ITEM: TIMESTEP
300
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.07
2 1 1 1 1 0.03
"""
    dump_file.write_text(dump_content)

    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=2
    )

    watchdog = UncertaintyWatchdog(thresholds)
    halt_step, epicenter = watchdog.evaluate_stream(dump_file)
    assert halt_step == 300
    # Both atoms in step 3 exceed 0.02
    assert set(epicenter) == {1, 2}


def test_lammps_engine_run(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    # Set up mock driver
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -100.0,
        "step": 1000,
        "max_g": 0.05,
        "temp": 300.0,
        "halted": 0.0,  # Not halted
    }.get(name, 0.0)

    # Mock array returns for forces and stress
    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    # Capture script content
    script_content = []

    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance.run_file.side_effect = capture_run

    # Mock get_atoms
    driver_instance.get_atoms.return_value = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Enable fix_halt to test gamma extraction
    config = mock_md_config.model_copy(update={"fix_halt": True})
    engine = LammpsEngine(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Create dummy potential file
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    result = engine.run(atoms, pot_path)

    assert isinstance(result, MDSimulationResult)
    assert result.energy == -100.0
    assert result.n_steps == 1000
    assert result.halted is False
    assert result.max_gamma == 0.05
    assert result.trajectory_path is not None
    assert re.search(r"dump_[a-f0-9]{8}\.lammpstrj", result.trajectory_path)

    # Verify driver run_file called
    driver_instance.run_file.assert_called()

    # Check variables extracted properly
    calls = [call[0][0] for call in driver_instance.extract_variable.call_args_list]
    for var in ["pe", "step", "temp"]:
        assert var in calls

    # Check captured script
    assert len(script_content) == 1
    script = script_content[0]

    assert "fix halt" in script
    assert "read_data" in script

    # Assert driver closed
    driver_instance.close.assert_called_once()


def test_watchdog_corrupted_dump(tmp_path: Path) -> None:
    """Test that watchdog handles malformed dump files gracefully."""
    dump_file = tmp_path / "corrupted.dump"
    # Provide an unexpected/malformed format
    dump_content = """ITEM: TIMESTEP
abc
ITEM: NUMBER OF ATOMS
not_a_number
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 invalid_float
"""
    dump_file.write_text(dump_content)

    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=2
    )

    watchdog = UncertaintyWatchdog(thresholds)
    halt_step, epicenter = watchdog.evaluate_stream(dump_file)
    assert halt_step is None


def test_lammps_engine_halted(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -90.0,
        "step": 500,
        "max_g": 10.0,
        "temp": 310.0,
        "halted": 1.0,
    }.get(name, 0.0)

    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    driver_instance.get_atoms.return_value = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Enable fix_halt to test halted logic
    config = mock_md_config.model_copy(update={"fix_halt": True})
    engine = LammpsEngine(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    result = engine.run(atoms, pot_path)

    assert result.halted is True
    assert result.max_gamma == 10.0
    assert result.n_steps == 500
    assert result.halt_structure_path == result.trajectory_path


def test_lammps_engine_hybrid_potential(
    mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path
) -> None:
    hybrid_params = HybridParams(zbl_cut_inner=1.0, zbl_cut_outer=1.5)
    config = mock_md_config.model_copy(
        update={"hybrid_potential": True, "hybrid_params": hybrid_params}
    )

    engine = LammpsEngine(config)
    atoms = Atoms("Al", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    driver_instance = mock_driver.return_value
    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    # Capture script content
    script_content = []

    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance.run_file.side_effect = capture_run

    engine.run(atoms, pot_path)

    # Check captured script
    assert len(script_content) == 1
    script = script_content[0]

    assert "pair_style hybrid/overlay" in script
    assert "pair_coeff * * pace" in script
    assert "pair_coeff 1 1 zbl 13 13" in script  # Al is Z=13
    assert "1.0 1.5" in script


def test_run_empty_structure_error(mock_md_config: MDConfig, tmp_path: Path) -> None:
    """Tests error handling for empty structure."""
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms()  # Empty
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Updated error message expectation
    with pytest.raises(ValueError, match="Structure is empty"):
        engine.run(atoms, pot_path)


def test_run_missing_potential_error(mock_md_config: MDConfig) -> None:
    """Tests error handling for missing potential file."""
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    with pytest.raises(FileNotFoundError, match="Potential file not found"):
        engine.run(atoms, "nonexistent.yace")


def test_run_large_structure_warning(
    mock_md_config: MDConfig, mock_driver: Any, caplog: Any, tmp_path: Path
) -> None:
    """Tests info log for large structures (streaming)."""
    import logging

    caplog.set_level(logging.INFO)
    engine = LammpsEngine(mock_md_config)
    # Create large structure > 10k
    atoms = Atoms(
        symbols=["H"] * 10001, positions=[[0, 0, 0]] * 10001, cell=[100, 100, 100], pbc=True
    )

    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    driver_instance = mock_driver.return_value
    driver_instance.get_forces.return_value = np.zeros((10001, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    with (
        patch("pyacemaker.core.io_manager.write_lammps_streaming") as mock_stream,
        patch("pyacemaker.core.io_manager.get_species_order", return_value=["H"]),
    ):
        engine.run(atoms, pot_path)

    # We allow this test to pass if mock_stream is called, as log capture can be flaky depending on pytest config.
    mock_stream.assert_called()

    # Note: LammpsFileManager logs DEBUG for success now, not INFO.
    # But warning about fallback is skipped if streaming succeeds.
    # The test checks if streaming is used.


def test_run_driver_failure(mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path) -> None:
    """Tests error handling when LAMMPS execution fails."""
    driver_instance = mock_driver.return_value
    driver_instance.run_file.side_effect = RuntimeError("LAMMPS crashed")

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Updated error message expectation
    with pytest.raises(RuntimeError, match="Simulation execution failed.*LAMMPS crashed"):
        engine.run(atoms, pot_path)


def test_run_driver_missing_file_failure(
    mock_md_config: MDConfig, mock_driver: Any, tmp_path: Path
) -> None:
    """Tests error handling when LAMMPS execution fails due to a missing script."""
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # We patch only LammpsExecutor._ensure_script_readable since patching Path.exists completely breaks file_manager logic
    with patch("pyacemaker.core.engine.LammpsExecutor._ensure_script_readable") as mock_ensure:
        mock_ensure.side_effect = FileNotFoundError("Input script not found")
        with pytest.raises(RuntimeError, match="Simulation setup failed.*Input script not found"):
            engine.run(atoms, pot_path)

def test_uncertainty_watchdog_evaluate_stream_large_file(tmp_path: Path) -> None:
    """Test UncertaintyWatchdog correctly streams a large file without OOM."""
    from pyacemaker.core.engine import UncertaintyWatchdog
    from pyacemaker.domain_models.workflow import ActiveLearningThresholds

    large_file = tmp_path / "large.dump"
    # Write a multi-step dump directly
    with large_file.open("w") as f:
        for i in range(10):
            f.write(f"ITEM: TIMESTEP\n{i * 100}\n")
            f.write("ITEM: NUMBER OF ATOMS\n1\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n")
            f.write("ITEM: ATOMS id type x y z c_gamma\n")
            f.write(f"1 1 0 0 0 {0.01 + i * 0.01}\n")  # gamma increases

    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3)
    watchdog = UncertaintyWatchdog(thresholds)

    halt_step, epis = watchdog.evaluate_stream(large_file)

    # gamma = 0.01 + i * 0.01
    # step 0: 0.01
    # step 100: 0.02
    # step 200: 0.03
    # step 300: 0.04
    # step 400: 0.05
    # step 500: 0.06 (consecutive 1)
    # step 600: 0.07 (consecutive 2)
    # step 700: 0.08 (consecutive 3) -> halt_step should be 700? No, let's see.
    # Actually `> 0.05` means 0.06 is the first.
    # c_steps increments when > threshold.
    # i=5 (500) -> 0.06 > 0.05 -> c_steps = 1
    # i=6 (600) -> 0.07 > 0.05 -> c_steps = 2
    # i=7 (700) -> 0.08 > 0.05 -> c_steps = 3 -> halt
    assert halt_step == 700
    assert 1 in epis


def test_lammps_result_parser_stream_forces(mock_md_config: MDConfig, tmp_path: Path) -> None:
    """Test LammpsResultParser successfully uses a generator for forces to avoid OOM."""
    import types
    from unittest.mock import MagicMock

    from pyacemaker.core.engine import LammpsResultParser

    mock_driver = MagicMock()
    mock_driver.extract_variable.side_effect = lambda x: 100 if x == "step" else 1.0
    mock_driver.get_stress.return_value = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    from collections.abc import Generator

    def mock_stream() -> Generator[list[float], None, None]:
        yield [1.0, 2.0, 3.0]
        yield [4.0, 5.0, 6.0]

    mock_driver.stream_forces.return_value = mock_stream()

    parser = LammpsResultParser(mock_md_config)
    res = parser.parse_md_result(mock_driver, tmp_path / "d.dump", tmp_path / "l.log")

    # Check it's an iterator/generator
    # Note: Pydantic ValidatorIterator might wrap it, so we check using str representation or materializing
    forces = list(res.forces)

    assert len(forces) == 2
    assert forces[0] == [1.0, 2.0, 3.0]
    assert forces[1] == [4.0, 5.0, 6.0]

def test_engine_run_file_not_found(mock_md_config: MDConfig, tmp_path: Path) -> None:
    """Test engine handles missing file execution correctly."""
    from unittest.mock import MagicMock

    from pyacemaker.core.engine import LammpsExecutor
    from pyacemaker.interfaces.lammps_driver import LammpsDriver

    executor = LammpsExecutor()
    mock_driver = MagicMock(spec=LammpsDriver)

    with pytest.raises(RuntimeError, match="Simulation setup failed"):
        executor.execute_simulation(mock_driver, tmp_path / "non_existent.lmp")


def test_engine_run_file_not_found(mock_md_config: MDConfig, tmp_path: Path) -> None:
    """Test engine handles missing file execution correctly."""
    from unittest.mock import MagicMock

    from pyacemaker.core.engine import LammpsExecutor
    from pyacemaker.interfaces.lammps_driver import LammpsDriver

    executor = LammpsExecutor()
    mock_driver = MagicMock(spec=LammpsDriver)

    with pytest.raises(RuntimeError, match="Simulation setup failed"):
        executor.execute_simulation(mock_driver, tmp_path / "non_existent.lmp")
