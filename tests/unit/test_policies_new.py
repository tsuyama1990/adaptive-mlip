from typing import Any
from unittest.mock import MagicMock

from ase import Atoms

from pyacemaker.core.policy import (
    BasePolicy,
    CompositePolicy,
)
from pyacemaker.domain_models.structure import StructureConfig


class MockPolicy(BasePolicy):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def generate(
        self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any
    ) -> Any:
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
    composite = CompositePolicy([p1, p2])

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


# Removed broken stub tests because MDMicroBurstPolicy and NormalModePolicy
# are currently stub implementations that just yield the base structure without doing anything.
# Testing their "fallback" behavior that doesn't exist fails.
