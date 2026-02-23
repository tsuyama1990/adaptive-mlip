from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.constants import TEST_ENERGY_H2O


class UATMockCalculator(Calculator):
    """
    Mock ASE calculator for testing purposes.
    Can simulate failures and setup errors.
    """

    def __init__(self, fail_count: int = 0, setup_error: bool = False) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.implemented_properties = ["energy", "forces", "stress"]
        self.fail_count = fail_count
        self.setup_error = setup_error
        self.attempts = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        self.attempts += 1

        if self.setup_error:
            msg = "Setup failed"
            raise RuntimeError(msg)

        if self.attempts <= self.fail_count:
            # Simulate SCF failure
            msg = "Convergence not achieved"
            raise RuntimeError(msg)

        self.results = {
            "energy": TEST_ENERGY_H2O,
            "forces": np.zeros((len(atoms) if atoms else 3, 3)),
            "stress": np.array([0.0] * 6),
        }


@pytest.fixture
def uat_dft_config(tmp_path, monkeypatch) -> DFTConfig:
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


def test_uat_02_01_single_point_calculation(uat_dft_config: DFTConfig, monkeypatch) -> None:
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
        # Mock get_calculator to return a UATMockCalculator instance
        mock_driver_instance.get_calculator.side_effect = lambda atoms, config: UATMockCalculator(fail_count=0)

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

        calc_fail = UATMockCalculator(fail_count=1)
        calc_success = UATMockCalculator(fail_count=0)

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
