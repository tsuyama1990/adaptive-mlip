from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pyacemaker.core.exceptions import OracleError
from pyacemaker.core.oracle import MACEManager, TieredOracle
from pyacemaker.domain_models.workflow import ActiveLearningThresholds


class DummyMaceCalc(Calculator):
    implemented_properties: list[str] = ["energy", "forces"]  # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)  # type: ignore[no-untyped-call]
        self.models = [1, 2] # Dummy to trigger node variance extraction block

    def calculate(self, atoms: Atoms | None = None, properties: list[str] | None = None, system_changes: list[str] = all_changes) -> None:
        if atoms is None:
            return
        super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]
        n_atoms = len(atoms)
        self.results["energy"] = -10.0 * n_atoms
        self.results["forces"] = np.ones((n_atoms, 3)) * 0.1

def test_macemanager_initialization(tmp_path: Path) -> None:
    model_path = tmp_path / "model.model"
    model_path.touch()

    with patch("pyacemaker.utils.security.validate_path_containment", return_value=model_path), \
         patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))
        assert manager.is_initialized

def test_macemanager_initialization_failure(tmp_path: Path) -> None:
    model_path = tmp_path / "model.model"
    model_path.touch()

    with patch("pyacemaker.utils.security.validate_path_containment", return_value=model_path), \
         patch("mace.calculators.mace_mp", side_effect=Exception("Model failed to load")), \
         pytest.raises(OracleError, match="Failed to load MACE model"):
        MACEManager(str(model_path))

def test_macemanager_compute(tmp_path: Path) -> None:
    model_path = tmp_path / "model.model"
    model_path.touch()

    with patch("pyacemaker.utils.security.validate_path_containment", return_value=model_path), \
         patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        structures_iter = manager.compute(iter([atoms]))

        computed_atoms = next(structures_iter)
        assert "energy" in computed_atoms.info
        assert computed_atoms.has("forces")
        assert computed_atoms.has("c_gamma")

        c_gamma = computed_atoms.get_array("c_gamma")
        assert len(c_gamma) == 2
        # np.linalg.norm(np.ones(3) * 0.1) * 0.01 = sqrt(3*0.01) * 0.01 = 0.001732
        assert np.allclose(c_gamma, 0.0017320508)

def test_macemanager_compute_invalid_input(tmp_path: Path) -> None:
    model_path = tmp_path / "model.model"
    model_path.touch()

    with patch("pyacemaker.utils.security.validate_path_containment", return_value=model_path), \
         patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))

        with pytest.raises(TypeError, match="Oracle failed to create iterator"):
            manager.compute([Atoms("H")])  # type: ignore[arg-type]

def test_tiered_oracle_initialization() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3)

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)
    assert oracle.mace == mock_mace
    assert oracle.dft == mock_dft

    with pytest.raises(ValueError, match="MACEManager must be provided"):
        TieredOracle(mace_manager=None, dft_manager=mock_dft, thresholds=thresholds)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="DFTManager cannot be None"):
        TieredOracle(mace_manager=mock_mace, dft_manager=None, thresholds=thresholds)  # type: ignore[arg-type]

def test_tiered_oracle_compute_below_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3)

    atoms_result = Atoms("H")
    atoms_result.new_array("c_gamma", np.array([0.01]))  # type: ignore[no-untyped-call]

    mock_mace.compute.return_value = iter([atoms_result])

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)

    atoms_input = Atoms("H")
    result_iter = oracle.compute(iter([atoms_input]))
    result = next(result_iter)

    assert result == atoms_result
    mock_mace.compute.assert_called_once()
    mock_dft.compute.assert_not_called()

def test_tiered_oracle_compute_above_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3)

    atoms_mace_result = Atoms("H")
    atoms_mace_result.new_array("c_gamma", np.array([0.1]))  # type: ignore[no-untyped-call]

    atoms_dft_result = Atoms("H")

    mock_mace.compute.return_value = iter([atoms_mace_result])
    mock_dft.compute.return_value = iter([atoms_dft_result])

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)

    atoms_input = Atoms("H")
    result_iter = oracle.compute(iter([atoms_input]))
    result = next(result_iter)

    assert result == atoms_dft_result
    assert result.has("c_gamma")
    assert np.array_equal(result.get_array("c_gamma"), np.array([0.1]))

    mock_mace.compute.assert_called_once()
    mock_dft.compute.assert_called_once()

def test_tiered_oracle_compute_invalid_input() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3)

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        oracle.compute([Atoms("H")])  # type: ignore[arg-type]
