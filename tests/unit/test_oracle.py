from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.conftest import MockCalculator, create_dummy_pseudopotentials
from tests.constants import TEST_ENERGY_GENERIC


@pytest.fixture
def mock_dft_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["H"])

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
    # Mock returns a calculator instance
    calc = MockCalculator(fail_count=0)
    mock_driver.get_calculator.return_value = calc

    # Inject mock driver
    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # Verify generator behavior with next() instead of list()
    generator = manager.compute(iter([atoms]))
    result = next(generator)

    assert result.get_potential_energy() == TEST_ENERGY_GENERIC  # type: ignore[no-untyped-call]

    # Verify get_calculator was called with correct config
    from unittest.mock import ANY
    mock_driver.get_calculator.assert_called_with(atoms, mock_dft_config, directory=ANY)


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

    # Use next() to consume generator one-by-one without materializing list
    gen = manager.compute(iter([atoms]))
    result = next(gen)

    assert result.get_potential_energy() == TEST_ENERGY_GENERIC  # type: ignore[no-untyped-call]

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

    # Check if config was modified (it's a copy)
    # The strategy modifies the copy.
    # Just check if it's different from original default
    assert config2.mixing_beta != 0.7


def test_dft_manager_fatal_error(mock_dft_config: DFTConfig) -> None:
    """Test fatal error after exhausting retries."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    mock_driver = MagicMock()
    # Always fail
    mock_driver.get_calculator.return_value = MockCalculator(fail_count=100)

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # Now raises OracleError
    # Use next() to trigger execution
    gen = manager.compute(iter([atoms]))
    with pytest.raises(OracleError, match="DFT calculation failed"):
        next(gen)

    # Verify retries happened (at least > 1)
    assert mock_driver.get_calculator.call_count > 1


def test_dft_manager_setup_error(mock_dft_config: DFTConfig) -> None:
    """Test handling of CalculatorSetupError."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    mock_driver = MagicMock()
    # Fails with setup error (e.g. missing pseudo file)
    mock_driver.get_calculator.return_value = MockCalculator(setup_error=True)

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    gen = manager.compute(iter([atoms]))
    with pytest.raises(OracleError, match="DFT calculation failed"):
        next(gen)

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

    # Strategy 1: Reduce Beta
    strat_beta = strategies[1]
    assert strat_beta is not None
    config_copy = mock_dft_config.model_copy()
    original_beta = config_copy.mixing_beta
    strat_beta(config_copy)
    assert config_copy.mixing_beta == original_beta * 0.5

    # Strategy 2: Increase Smearing
    strat_smearing = strategies[2]
    assert strat_smearing is not None
    config_copy = mock_dft_config.model_copy()
    original_smearing = config_copy.smearing_width
    strat_smearing(config_copy)
    assert config_copy.smearing_width == original_smearing * 2.0

    # Strategy 3: CG Diagonalization
    strat_cg = strategies[3]
    assert strat_cg is not None
    config_copy = mock_dft_config.model_copy()
    strat_cg(config_copy)
    assert config_copy.diagonalization == "cg"

def test_dft_manager_invalid_input(mock_dft_config: DFTConfig) -> None:
    """Test compute raises TypeError for non-iterator input."""
    manager = DFTManager(mock_dft_config)
    atoms_list = [Atoms("H")]

    # Check that it raises TypeError immediately upon calling next()
    # because generator execution is deferred.
    # We want to ensure it fails.
    gen = manager.compute(atoms_list) # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be an Iterator"):
        next(gen)

def test_dft_manager_empty_iterator(mock_dft_config: DFTConfig) -> None:
    """Test compute handles empty iterator correctly with warning."""
    manager = DFTManager(mock_dft_config)
    empty_iter: Iterator[Atoms] = iter([])

    # Explicit loop without list() materialization for safety
    # Use deque(..., maxlen=0) to consume iterator efficiently
    from collections import deque
    with pytest.warns(UserWarning, match="Oracle received empty iterator"):
        deque(manager.compute(empty_iter), maxlen=0)

def test_dft_manager_embedding(mock_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that embedding is applied when configured."""
    from pyacemaker.core.oracle import DFTManager

    # Configure embedding buffer
    mock_dft_config.embedding_buffer = 5.0

    # Mock embed_cluster
    mock_embed = MagicMock()
    # Return a dummy atoms object
    embedded_atoms = Atoms("H", cell=[20, 20, 20], pbc=True)
    mock_embed.return_value = embedded_atoms

    monkeypatch.setattr("pyacemaker.core.oracle.embed_cluster", mock_embed)

    # Mock Driver
    mock_driver = MagicMock()
    mock_driver.get_calculator.return_value = MockCalculator(fail_count=0)

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    atoms = Atoms("H", positions=[[0, 0, 0]])
    # Must be iterator
    gen = manager.compute(iter([atoms]))
    result = next(gen)

    # Check if embed_cluster was called
    mock_embed.assert_called_once()
    args, kwargs = mock_embed.call_args
    assert args[0] == atoms
    assert kwargs['buffer'] == 5.0

    # Check if result is the embedded one
    # DFTManager.compute yields the result of _compute_single(embedded_atoms)
    # _compute_single returns the atom object passed to it (which is embedded_atoms)
    assert result == embedded_atoms
