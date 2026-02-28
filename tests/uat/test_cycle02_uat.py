from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine, UncertaintyWatchdog
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from pyacemaker.domain_models.workflow import ActiveLearningThresholds
from pyacemaker.domain_models.md import MDConfig
from tests.conftest import MockCalculator
from tests.constants import TEST_ENERGY_H2O


@pytest.fixture
def uat_dft_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "H.UPF").touch()
    (tmp_path / "O.UPF").touch()

    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF", "O": "O.UPF"},
    )


def test_uat_02_01_single_point_calculation(uat_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario 02-01: Single Point Calculation.
    Verify that the system can run a simple DFT calculation (mocked).
    """
    # 1. Preparation: H2O molecule
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True)

    # 2. Action: Run DFTManager with mocked driver
    # We patch QEDriver but we also need to ensure the driver instance returned
    # has a get_calculator method that returns our calculator

    # We patch at the source where DFTManager imports it or uses it
    # DFTManager imports QEDriver from interfaces.qe_driver

    with patch("pyacemaker.core.oracle.QEDriver") as MockDriverClass:
        mock_driver_instance = MockDriverClass.return_value
        # Mock get_calculator to return a MockCalculator instance with H2O energy
        # Accept **kwargs to handle 'directory' argument
        mock_driver_instance.get_calculator.side_effect = lambda atoms, config, **kwargs: MockCalculator(
            fail_count=0, test_energy=TEST_ENERGY_H2O
        )

        manager = DFTManager(uat_dft_config)

        # Use explicit iteration
        gen = manager.compute(iter([h2o]))
        result = next(gen)

        # 3. Expectation
        assert result.get_potential_energy() == TEST_ENERGY_H2O  # type: ignore[no-untyped-call]
        assert result.get_forces().shape == (3, 3)  # type: ignore[no-untyped-call]


def test_uat_02_02_self_healing(uat_dft_config: DFTConfig, caplog: pytest.LogCaptureFixture) -> None:
    """
    Scenario 02-02: Self-Healing Test.
    Verify that the system recovers from a simulated SCF convergence failure.
    """
    # 1. Preparation
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True)

    # 2. Action: Run DFTManager with failure
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriverClass:
        mock_driver_instance = MockDriverClass.return_value

        # Mock failure on first attempt, success on second
        # We need side_effect to return distinct calculator instances or handle state
        # But here get_calculator is called with (atoms, config)
        # We can use side_effect on the mock method

        calc_fail = MockCalculator(fail_count=1, test_energy=TEST_ENERGY_H2O)
        calc_success = MockCalculator(fail_count=0, test_energy=TEST_ENERGY_H2O)

        mock_driver_instance.get_calculator.side_effect = [calc_fail, calc_success]

        manager = DFTManager(uat_dft_config)

        gen = manager.compute(iter([h2o]))
        result = next(gen)

        # 3. Expectation
        assert result.get_potential_energy() == TEST_ENERGY_H2O  # type: ignore[no-untyped-call]

        # Verify that get_calculator was called twice (original + retry)
        assert mock_driver_instance.get_calculator.call_count == 2

        # Verify second call had reduced mixing_beta
        # First call: original (0.7)
        # Second call: reduced (0.35)
        args, _ = mock_driver_instance.get_calculator.call_args  # Last call
        final_config = args[1]
        assert final_config.mixing_beta < 0.7
        assert final_config.mixing_beta == 0.35

import numpy as np

from pyacemaker.core.engine import LammpsEngine, UncertaintyWatchdog
from pyacemaker.domain_models.workflow import ActiveLearningThresholds
from pyacemaker.domain_models.md import MDConfig


def test_uat_02_03_seamless_resume(mock_md_config: MDConfig, tmp_path: Path) -> None:
    """
    Scenario UAT-02-03: Seamless MD Resume
    Verify that halting a run mid-trajectory and passing a restart file
    resumes the system successfully.
    """
    # Create an initial configuration mimicking an NPT bulk system running 100 steps
    config = mock_md_config.model_copy(update={"n_steps": 100, "temperature": 300.0, "pressure": 1.0})
    engine = LammpsEngine(config)
    atoms = Atoms("Ar", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    # Mock LammpsDriver to capture executed scripts and output correct state
    with patch("pyacemaker.core.engine.LammpsDriver") as mock_driver:
        driver_instance = mock_driver.return_value
        driver_instance.extract_variable.return_value = 100
        driver_instance.get_forces.return_value = np.zeros((1, 3))
        driver_instance.get_stress.return_value = np.zeros(6)

        script_content = []

        def capture_run(path: str) -> None:
            script_content.append(Path(path).read_text())

        driver_instance.run_file.side_effect = capture_run

        # 1. Run full 100 steps
        engine.run(atoms, pot_path)

        # 2. Simulate resuming at step 50
        restart_file = tmp_path / "resume_50.restart"
        restart_file.touch()

        config_resume = mock_md_config.model_copy(update={"n_steps": 50})
        engine_resume = LammpsEngine(config_resume)

        script_content.clear()

        engine_resume.run(atoms, pot_path, restart_file=restart_file)

        # Check generated script starts with read_restart
        assert len(script_content) == 1
        script = script_content[0]
        assert f"read_restart {restart_file}" in script
        assert "read_data" not in script
        assert "velocity all create" not in script

        # We assert the engine ran correctly through the soft start
        assert "run 50" in script
        assert "fix soft_start_langevin" in script


def test_uat_02_04_thermal_noise_exclusion(tmp_path: Path) -> None:
    """
    Scenario UAT-02-04: Thermal Noise Exclusion
    Verify the two-tier threshold system correctly filters out single-step spikes
    and only halts when uncertainty is sustained.
    """
    dump_file = tmp_path / "test.dump"

    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05,
        threshold_add_train=0.02,
        smooth_steps=3
    )

    # 1. Test Thermal Noise (Single Spike)
    dump_content_noise = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.06
ITEM: TIMESTEP
300
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
"""
    dump_file.write_text(dump_content_noise)

    halt_step_noise, epicenter_noise = UncertaintyWatchdog._evaluate_uncertainty_stream(dump_file, thresholds)
    assert halt_step_noise is None  # Should not halt

    # 2. Test Sustained Uncertainty
    dump_content_sustained = """ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.06
ITEM: TIMESTEP
300
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.07
ITEM: TIMESTEP
400
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.08
ITEM: TIMESTEP
500
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_gamma
1 1 0 0 0 0.01
"""
    dump_file.write_text(dump_content_sustained)

    halt_step_sustained, epicenter_sustained = UncertaintyWatchdog._evaluate_uncertainty_stream(dump_file, thresholds)
    assert halt_step_sustained == 400  # Should halt exactly on the 3rd consecutive step > 0.05
    assert set(epicenter_sustained) == {1}
