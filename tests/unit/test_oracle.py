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
        self.models = [1, 2]  # Dummy to trigger node variance extraction block

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        if atoms is None:
            return
        super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]
        n_atoms = len(atoms)
        self.results["energy"] = -10.0 * n_atoms
        self.results["forces"] = np.ones((n_atoms, 3)) * 0.1


def get_safe_test_model_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    pot_dir = tmp_path / "potentials"
    monkeypatch.setattr("pyacemaker.domain_models.defaults.DEFAULT_POTENTIALS_DIR", str(pot_dir))
    pot_dir.mkdir(parents=True, exist_ok=True)
    return pot_dir / "model.model"


def test_macemanager_initialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = get_safe_test_model_path(monkeypatch, tmp_path)
    model_path.touch()

    with patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))
        assert manager.is_initialized

    model_path.unlink()


def test_macemanager_initialization_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = get_safe_test_model_path(monkeypatch, tmp_path)
    model_path.touch()

    with (
        patch("mace.calculators.mace_mp", side_effect=Exception("Model failed to load")),
        pytest.raises(OracleError, match="Failed to load MACE model"),
    ):
        MACEManager(str(model_path))

    model_path.unlink()


def test_macemanager_compute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = get_safe_test_model_path(monkeypatch, tmp_path)
    model_path.touch()

    with patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))

        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
        structures_iter = manager.compute(iter([atoms]))

        computed_atoms = next(structures_iter)
        assert "energy" in computed_atoms.info
        assert computed_atoms.has("forces")  # type: ignore[no-untyped-call]
        assert computed_atoms.has("c_gamma")  # type: ignore[no-untyped-call]

        c_gamma = computed_atoms.get_array("c_gamma")  # type: ignore[no-untyped-call]
        assert len(c_gamma) == 2
        # np.linalg.norm(np.ones(3) * 0.1) * 0.01 = sqrt(3*0.01) * 0.01 = 0.001732
        assert np.allclose(c_gamma, 0.0017320508)
        assert np.all(c_gamma >= 0.0), "Uncertainty must be non-negative"

        # Edge case: zero forces
        class ZeroMaceCalc(DummyMaceCalc):
            def calculate(self, atoms: Atoms | None = None, properties: list[str] | None = None, system_changes: list[str] = all_changes) -> None:
                if atoms is None:
                    return
                super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]
                n_atoms = len(atoms)
                self.results["forces"] = np.zeros((n_atoms, 3))

        manager_zero = MACEManager(str(model_path), calculator=ZeroMaceCalc())
        computed_atoms_zero = next(manager_zero.compute(iter([Atoms("H")])))
        assert np.allclose(computed_atoms_zero.get_array("c_gamma"), 0.0)  # type: ignore[no-untyped-call]

        # Edge case: huge forces
        class HugeMaceCalc(DummyMaceCalc):
            def calculate(self, atoms: Atoms | None = None, properties: list[str] | None = None, system_changes: list[str] = all_changes) -> None:
                if atoms is None:
                    return
                super().calculate(atoms, properties, system_changes)  # type: ignore[no-untyped-call]
                n_atoms = len(atoms)
                self.results["forces"] = np.ones((n_atoms, 3)) * 1e6

        manager_huge = MACEManager(str(model_path), calculator=HugeMaceCalc())
        computed_atoms_huge = next(manager_huge.compute(iter([Atoms("H")])))
        huge_gamma = computed_atoms_huge.get_array("c_gamma")[0]  # type: ignore[no-untyped-call]
        assert huge_gamma > 1000.0, "Huge forces should result in large uncertainty metric proxy"

    model_path.unlink()


def test_macemanager_compute_invalid_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_path = get_safe_test_model_path(monkeypatch, tmp_path)
    model_path.touch()

    with patch("mace.calculators.mace_mp", return_value=DummyMaceCalc()):
        manager = MACEManager(str(model_path))

        with pytest.raises(TypeError, match="Oracle failed to create iterator"):
            manager.compute([Atoms("H")])  # type: ignore[arg-type]

    model_path.unlink()


def test_tiered_oracle_initialization() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

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
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

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

def test_tiered_oracle_compute_boundary_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    # Exact boundary edge case
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3)

    atoms_mace_result = Atoms("H")
    atoms_mace_result.new_array("c_gamma", np.array([0.05]))  # type: ignore[no-untyped-call]

    mock_mace.compute.return_value = iter([atoms_mace_result])
    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)

    result = next(oracle.compute(iter([Atoms("H")])))

    # Should NOT fall back to DFT because it is <= threshold (not strictly >)
    assert result == atoms_mace_result
    mock_dft.compute.assert_not_called()


def test_tiered_oracle_compute_above_threshold() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

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
    assert result.has("c_gamma")  # type: ignore[no-untyped-call]
    assert np.array_equal(result.get_array("c_gamma"), np.array([0.1]))  # type: ignore[no-untyped-call]

    mock_mace.compute.assert_called_once()
    mock_dft.compute.assert_called_once()


def test_tiered_oracle_compute_invalid_input() -> None:
    mock_mace = MagicMock()
    mock_dft = MagicMock()
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    oracle = TieredOracle(mace_manager=mock_mace, dft_manager=mock_dft, thresholds=thresholds)
    with pytest.raises(TypeError, match="Oracle failed to create iterator"):
        oracle.compute([Atoms("H")])  # type: ignore[arg-type]
