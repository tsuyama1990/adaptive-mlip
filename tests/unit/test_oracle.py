from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.conftest import MockCalculator


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


def test_dft_manager_compute_success(mock_dft_config: DFTConfig) -> None:
    """Test successful computation using dependency injection."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Create Mock Driver
    mock_driver = MagicMock()
    mock_driver.get_calculator.return_value = MockCalculator(fail_count=0)

    # Inject mock driver
    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # Verify generator behavior with next() instead of list()
    generator = manager.compute(iter([atoms]))
    result = next(generator)

    assert result.get_potential_energy() == -13.6  # type: ignore[no-untyped-call]

    # Verify get_calculator was called with correct config
    mock_driver.get_calculator.assert_called_with(atoms, mock_dft_config)


def test_dft_manager_self_healing(mock_dft_config: DFTConfig) -> None:
    """Test self-healing mechanism."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    # Mock Driver
    mock_driver = MagicMock()

    # The calculator needs to fail first, then succeed.
    calc_fail = MockCalculator(fail_count=1) # Fails once (attempt 1)
    calc_success = MockCalculator(fail_count=0) # Succeeds (attempt 2)

    mock_driver.get_calculator.side_effect = [calc_fail, calc_success]

    # Inject mock driver
    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # Note: consume generator to trigger execution
    results = list(manager.compute(iter([atoms])))

    assert len(results) == 1
    assert results[0].get_potential_energy() == -13.6  # type: ignore[no-untyped-call]

    # Verify calls to get_calculator
    assert mock_driver.get_calculator.call_count == 2

    # First call: original config
    call1_args = mock_driver.get_calculator.call_args_list[0]
    config1 = call1_args[0][1] # second arg is config
    assert config1.mixing_beta == 0.7
    assert config1.smearing_width == 0.1
    assert config1.diagonalization == "david"

    # Second call: updated config (reduced mixing_beta)
    call2_args = mock_driver.get_calculator.call_args_list[1]
    config2 = call2_args[0][1]

    assert config2.mixing_beta < 0.7


def test_dft_manager_fatal_error(mock_dft_config: DFTConfig) -> None:
    """Test fatal error after exhausting retries."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    mock_driver = MagicMock()
    # Always fail
    mock_driver.get_calculator.return_value = MockCalculator(fail_count=100)

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # Now raises OracleError
    with pytest.raises(OracleError, match="DFT calculation failed"):
        list(manager.compute(iter([atoms])))

    # Verify retries happened (at least > 1)
    assert mock_driver.get_calculator.call_count > 1


def test_dft_manager_setup_error(mock_dft_config: DFTConfig) -> None:
    """Test handling of CalculatorSetupError."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    mock_driver = MagicMock()
    # Fails with setup error (e.g. missing pseudo file)
    mock_driver.get_calculator.return_value = MockCalculator(setup_error=True)

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    with pytest.raises(OracleError, match="DFT calculation failed"):
        list(manager.compute(iter([atoms])))

    # Should retry even on setup error if it's considered transient or parameter based?
    # Spec says "JobFailedException" (RuntimeError). Implementation catches (RuntimeError, CalculatorSetupError).
    # So it should retry.
    assert mock_driver.get_calculator.call_count > 1


def test_dft_manager_strategies(mock_dft_config: DFTConfig) -> None:
    """Test that strategies are correctly defined."""
    manager = DFTManager(mock_dft_config)
    strategies = manager._get_strategies()

    assert len(strategies) > 0
    assert strategies[0] is None # First attempt is vanilla

    # Test strategy logic (e.g. reduced beta)
    strat_beta = strategies[1]
    assert strat_beta is not None

    config_copy = mock_dft_config.model_copy()
    original_beta = config_copy.mixing_beta
    strat_beta(config_copy)
    # Using default factor 0.5
    assert config_copy.mixing_beta == original_beta * 0.5
