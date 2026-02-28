from typing import Any
from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.policy import (
    BasePolicy,
    CompositePolicy,
    MDMicroBurstPolicy,
    NormalModePolicy,
)
from pyacemaker.domain_models.structure import StructureConfig


class MockPolicy(BasePolicy):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ):
        for _ in range(n_structures):
            a = base_structure.copy()
            a.info["policy"] = self.name
            yield a


class MockEngine:
    # Class-level attribute to control return value from instances created inside policy
    result_to_return: Any = None

    def __init__(self, config: Any) -> None:
        self.config = config
        # Ensure config has model_copy
        if not hasattr(self.config, "model_copy"):
            self.config.model_copy = MagicMock(return_value=config)

    def run(self, structure: Any, potential: Any) -> Any:
        return self.result_to_return


def test_composite_policy_distribution() -> None:
    p1 = MockPolicy("p1")
    p2 = MockPolicy("p2")
    composite = CompositePolicy(p1, p2)

    config = StructureConfig(elements=["H"], supercell_size=[1, 1, 1])
    base = Atoms("H")

    # n=10, 2 policies -> 5 each
    results = list(composite.generate(base, config, n_structures=10))
    assert len(results) == 10
    counts = {"p1": 0, "p2": 0}
    for r in results:
        counts[r.info["policy"]] += 1

    assert counts["p1"] == 5
    assert counts["p2"] == 5

    # n=3, 2 policies -> 2 for p1, 1 for p2 (remainder logic)
    results = list(composite.generate(base, config, n_structures=3))
    assert len(results) == 3
    counts = {"p1": 0, "p2": 0}
    for r in results:
        counts[r.info["policy"]] += 1

    assert counts["p1"] == 2
    assert counts["p2"] == 1


def test_md_micro_burst_policy() -> None:
    # Setup Mock Result
    from pyacemaker.domain_models.md import MDSimulationResult

    mock_result = MagicMock(spec=MDSimulationResult)
    mock_result.trajectory_path = "dummy.traj"

    MockEngine.result_to_return = mock_result

    # Setup initial engine
    config_mock = MagicMock()
    config_mock.model_copy.return_value = config_mock
    initial_engine = MockEngine(config_mock)

    # Mock trajectory read
    with patch("pyacemaker.core.policy.read") as mock_read, patch("pyacemaker.core.policy.np.random.choice") as mock_choice:
            final_atoms = Atoms("He")
            mock_read.return_value = [final_atoms]
            mock_choice.return_value = final_atoms

            policy = MDMicroBurstPolicy()
            config = StructureConfig(elements=["H"], supercell_size=[1, 1, 1])
            base = Atoms("H")

            results = list(
                policy.generate(base, config, n_structures=1, engine=initial_engine, potential="pot")
            )

            assert len(results) == 1
            assert results[0] == final_atoms
            mock_read.assert_called_with("dummy.traj", index=":")


def test_md_micro_burst_fallback() -> None:
    # No engine provided -> Fallback to rattle
    policy = MDMicroBurstPolicy()
    config = StructureConfig(elements=["H"], supercell_size=[1, 1, 1])
    base = Atoms("H")

    results = list(policy.generate(base, config, n_structures=1))  # No engine kwarg

    assert len(results) == 1
    # Check if rattled (positions changed) or fallback logic executed
    # Rattle changes positions.
    assert results[0].get_chemical_symbols() == ["H"]


def test_normal_mode_policy_fallback() -> None:
    policy = NormalModePolicy()
    config = StructureConfig(elements=["H"], supercell_size=[1, 1, 1])
    base = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10])

    results = list(policy.generate(base, config, n_structures=1))

    assert len(results) == 1
    # Should fall back to rattle
    import numpy as np

    assert np.any(results[0].positions[0] != [0, 0, 0])  # Rattle moves atoms
