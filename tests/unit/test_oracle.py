from typing import ClassVar
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig


@pytest.fixture
def mock_dft_config() -> DFTConfig:
    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF"},
    )


class MockCalculator(Calculator):
    """Mock ASE calculator that can simulate failure."""
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
            # Simulate SCF failure (which usually raises RuntimeError in ASE)
            msg = "Convergence not achieved"
            raise RuntimeError(msg)

        self.results = {
            "energy": -13.6,
            "forces": np.array([[0.0, 0.0, 0.0]]),
            "stress": np.array([0.0] * 6),
        }


def test_dft_manager_compute_success(mock_dft_config: DFTConfig) -> None:
    """Test successful computation."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Mock QEDriver to return a working calculator
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(mock_dft_config)
        mock_driver_instance = MockDriver.return_value
        mock_driver_instance.get_calculator.return_value = MockCalculator(fail_count=0)

        results = list(manager.compute(iter([atoms])))

        assert len(results) == 1
        assert results[0].get_potential_energy() == -13.6

        # Verify get_calculator was called with correct config
        mock_driver_instance.get_calculator.assert_called_with(atoms, mock_dft_config)


def test_dft_manager_self_healing(mock_dft_config: DFTConfig) -> None:
    """Test self-healing mechanism."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Mock QEDriver to verify parameter changes
    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(mock_dft_config)
        mock_driver_instance = MockDriver.return_value

        # The calculator needs to fail first, then succeed.
        calc_fail = MockCalculator(fail_count=1) # Fails once (attempt 1)
        calc_success = MockCalculator(fail_count=0) # Succeeds (attempt 2)

        mock_driver_instance.get_calculator.side_effect = [calc_fail, calc_success]

        results = list(manager.compute(iter([atoms])))

        assert len(results) == 1
        assert results[0].get_potential_energy() == -13.6

        # Verify calls to get_calculator
        assert mock_driver_instance.get_calculator.call_count == 2

        # First call: original config
        call1_args = mock_driver_instance.get_calculator.call_args_list[0]
        config1 = call1_args[0][1] # second arg is config
        assert config1.mixing_beta == 0.7
        assert config1.smearing_width == 0.1
        assert config1.diagonalization == "david"

        # Second call: updated config (reduced mixing_beta)
        call2_args = mock_driver_instance.get_calculator.call_args_list[1]
        config2 = call2_args[0][1]

        assert config2.mixing_beta < 0.7


def test_dft_manager_fatal_error(mock_dft_config: DFTConfig) -> None:
    """Test fatal error after exhausting retries."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    with patch("pyacemaker.core.oracle.QEDriver") as MockDriver:
        manager = DFTManager(mock_dft_config)
        mock_driver_instance = MockDriver.return_value
        # Always fail
        mock_driver_instance.get_calculator.return_value = MockCalculator(fail_count=100)

        with pytest.raises(RuntimeError, match="DFT calculation failed"):
            list(manager.compute(iter([atoms])))

        # Verify retries happened (at least > 1)
        assert mock_driver_instance.get_calculator.call_count > 1
