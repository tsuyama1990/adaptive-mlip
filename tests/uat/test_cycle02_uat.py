from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.constants import TEST_ENERGY_H2O
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.conftest import MockCalculator

# Monkeypatch calculate to return specific UAT values if needed,
# or better, rely on conftest MockCalculator if generic values suffice.
# UAT checks for -14.5. Conftest gives -13.6.
# Let's subclass to keep UAT values.


class UATMockCalculator(MockCalculator):
    """Subclass to provide UAT-specific energy values."""

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] | None = None,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        # Override with UAT specific values
        self.results["energy"] = TEST_ENERGY_H2O
        # Ensure forces shape matches atoms
        n_atoms = len(atoms) if atoms else 3
        self.results["forces"] = np.zeros((n_atoms, 3))


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
    h2o = Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True
    )

    # 2. Action: Run DFTManager with mocked driver
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(uat_dft_config)
        mock_driver_instance = MockDriver.return_value
        mock_driver_instance.get_calculator.return_value = UATMockCalculator(fail_count=0)

        # Use explicit iteration to avoid list() materialization risk in principle,
        # though [h2o] is small.
        gen = manager.compute(iter([h2o]))
        result = next(gen)

        # 3. Expectation
        assert result.get_potential_energy() == TEST_ENERGY_H2O  # type: ignore[no-untyped-call]
        assert result.get_forces().shape == (3, 3)  # type: ignore[no-untyped-call]


def test_uat_02_02_self_healing(
    uat_dft_config: DFTConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Scenario 02-02: Self-Healing Test.
    Verify that the system recovers from a simulated SCF convergence failure.
    """
    # 1. Preparation
    h2o = Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True
    )

    # 2. Action: Run DFTManager with failure
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(uat_dft_config)
        mock_driver_instance = MockDriver.return_value

        # Mock failure on first attempt, success on second
        calc_fail = UATMockCalculator(fail_count=1)
        calc_success = UATMockCalculator(fail_count=0)
        mock_driver_instance.get_calculator.side_effect = [calc_fail, calc_success]

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
