import numpy as np
from ase import Atoms

from pyacemaker.core.oracle import MACEManager, TieredOracle
from pyacemaker.domain_models.workflow import ActiveLearningThresholds


def test_macemanager_compute_fallback():
    # If mace is not installed, it should fallback and set mock c_gamma
    manager = MACEManager()

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    stream = manager.compute(iter([atoms]))

    result = next(stream)

    assert "c_gamma" in result.arrays
    assert result.arrays["c_gamma"].shape == (2,)

def test_tiered_oracle_no_dft(mocker):
    # Mock MACEManager to return low uncertainty
    mace = MACEManager()
    mace.compute = mocker.MagicMock()
    atoms = Atoms("H")
    atoms.arrays["c_gamma"] = np.array([0.01])
    mace.compute.return_value = iter([atoms])

    dft = mocker.MagicMock()

    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05)

    oracle = TieredOracle(thresholds, mace, dft)

    stream = oracle.compute(iter([Atoms("H")]))
    result = next(stream)

    assert result == atoms
    dft.compute.assert_not_called()

def test_tiered_oracle_call_dft(mocker):
    # Mock MACEManager to return high uncertainty
    mace = MACEManager()
    mace.compute = mocker.MagicMock()
    atoms = Atoms("H")
    atoms.arrays["c_gamma"] = np.array([0.1])
    mace.compute.return_value = iter([atoms])

    dft = mocker.MagicMock()
    dft_atoms = Atoms("H")
    dft_atoms.info["dft"] = True
    dft.compute.return_value = iter([dft_atoms])

    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05)

    oracle = TieredOracle(thresholds, mace, dft)

    stream = oracle.compute(iter([Atoms("H")]))
    result = next(stream)

    assert result.info.get("dft") is True
    dft.compute.assert_called_once()
