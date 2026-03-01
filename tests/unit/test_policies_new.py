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

    def generate(self, base_structure: Atoms, config: StructureConfig, n_structures: int = 1, **kwargs: Any):
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
    composite = CompositePolicy(policies=[p1, p2])

    config = StructureConfig(elements=["H"], supercell_size=[1,1,1])
    base = Atoms("H")

    # With n=10, distribution yields len 10
    # Because SafeBasePolicy fallback works now
    results = list(composite.generate(base_structure=base, config=config, n_structures=10))
    assert len(results) == 10


def test_md_micro_burst_policy() -> None:
    # Setup Mock Result
    policy = MDMicroBurstPolicy()
    config = StructureConfig(elements=["H"], supercell_size=[1,1,1])
    base = Atoms("H")

    results = list(policy.generate(base_structure=base, config=config, n_structures=1))
    assert len(results) == 1
    results = list(policy.generate(base_structure=base, config=config, n_structures=1))
    assert len(results) == 1

def test_mock():
    pass


def test_normal_mode_policy_fallback() -> None:
    policy = NormalModePolicy()
    config = StructureConfig(elements=["H"], supercell_size=[1,1,1])
    base = Atoms("H", positions=[[0,0,0]], cell=[10,10,10])

    results = list(policy.generate(base_structure=base, config=config, n_structures=1))
    assert len(results) == 1
