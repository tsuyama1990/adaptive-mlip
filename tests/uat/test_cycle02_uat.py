from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig


class MockCalculator(Calculator):
    """Mock Calculator for UAT."""
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def __init__(self, fail_count: int = 0) -> None:
        super().__init__()
        self.fail_count = fail_count
        self.attempts = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            msg = "Convergence not achieved"
            raise RuntimeError(msg)

        self.results = {
            "energy": -14.5,  # Matches UAT expectation
            "forces": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "stress": np.array([0.0] * 6),
        }


@pytest.fixture
def uat_dft_config() -> DFTConfig:
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


def test_uat_02_01_single_point_calculation(uat_dft_config: DFTConfig) -> None:
    """
    Scenario 02-01: Single Point Calculation.
    Verify that the system can run a simple DFT calculation (mocked).
    """
    # 1. Preparation: H2O molecule
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True)

    # 2. Action: Run DFTManager with mocked driver
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(uat_dft_config)
        mock_driver_instance = MockDriver.return_value
        mock_driver_instance.get_calculator.return_value = MockCalculator(fail_count=0)

        results = list(manager.compute(iter([h2o])))

        # 3. Expectation
        assert len(results) == 1
        assert results[0].get_potential_energy() == -14.5
        assert results[0].get_forces().shape == (3, 3)


def test_uat_02_02_self_healing(uat_dft_config: DFTConfig, caplog: pytest.LogCaptureFixture) -> None:
    """
    Scenario 02-02: Self-Healing Test.
    Verify that the system recovers from a simulated SCF convergence failure.
    """
    # 1. Preparation
    h2o = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True)

    # 2. Action: Run DFTManager with failure
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(uat_dft_config)
        mock_driver_instance = MockDriver.return_value

        # Mock failure on first attempt, success on second
        calc_fail = MockCalculator(fail_count=1)
        calc_success = MockCalculator(fail_count=0)
        mock_driver_instance.get_calculator.side_effect = [calc_fail, calc_success]

        results = list(manager.compute(iter([h2o])))

        # 3. Expectation
        assert len(results) == 1
        assert results[0].get_potential_energy() == -14.5

        # Verify that get_calculator was called twice (original + retry)
        assert mock_driver_instance.get_calculator.call_count == 2

        # Verify second call had reduced mixing_beta
        # First call: original (0.7)
        # Second call: reduced (0.35)
        args, _ = mock_driver_instance.get_calculator.call_args  # Last call
        final_config = args[1]
        assert final_config.mixing_beta < 0.7
        assert final_config.mixing_beta == 0.35
