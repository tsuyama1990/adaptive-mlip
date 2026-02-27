from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.base import BasePolicy
from pyacemaker.core.policy import (
    CompositePolicy,
    MDMicroBurstPolicy,
    NormalModePolicy,
)
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


@pytest.fixture
def base_structure() -> Atoms:
    return Atoms("Fe", positions=[[0, 0, 0]], cell=[2.8, 2.8, 2.8], pbc=True)


@pytest.fixture
def config() -> StructureConfig:
    return StructureConfig(
        elements=["Fe"],
        supercell_size=[1, 1, 1],
        active_policies=[ExplorationPolicy.COLD_START],
    )


def test_composite_policy_distribution(base_structure: Atoms, config: StructureConfig) -> None:
    # Create two dummy policies
    p1 = MagicMock(spec=BasePolicy)
    p2 = MagicMock(spec=BasePolicy)

    def gen_side_effect_p1(*args, **kwargs) -> Iterator[Atoms]: # type: ignore
        for _ in range(kwargs["n_structures"]):
            yield base_structure.copy() # type: ignore[no-untyped-call]

    def gen_side_effect_p2(*args, **kwargs) -> Iterator[Atoms]: # type: ignore
        for _ in range(kwargs["n_structures"]):
            yield base_structure.copy() # type: ignore[no-untyped-call]

    p1.generate.side_effect = gen_side_effect_p1
    p2.generate.side_effect = gen_side_effect_p2

    composite = CompositePolicy([p1, p2])

    n_total = 10
    results = list(composite.generate(base_structure, config, n_total))

    assert len(results) == n_total

    # Check distribution (5 each)
    assert p1.generate.call_count == 1
    assert p2.generate.call_count == 1

    args1, kwargs1 = p1.generate.call_args
    assert kwargs1["n_structures"] == 5

    args2, kwargs2 = p2.generate.call_args
    assert kwargs2["n_structures"] == 5


def test_md_micro_burst_policy(base_structure: Atoms, config: StructureConfig) -> None:
    """Test placeholder execution."""
    policy = MDMicroBurstPolicy()
    results = list(policy.generate(base_structure, config, n_structures=2))
    assert len(results) == 2
    # Verify it falls back to something valid (rattle)
    assert results[0].positions is not None


def test_md_micro_burst_fallback(base_structure: Atoms, config: StructureConfig) -> None:
    """Explicitly test fallback mechanism (rattle)."""
    policy = MDMicroBurstPolicy()

    # Run with rattle stdev 0.1
    config.rattle_stdev = 0.1
    results = list(policy.generate(base_structure, config, n_structures=5))

    # Positions should change (rattle)
    for res in results:
        assert not np.allclose(res.positions, base_structure.positions)


def test_normal_mode_policy_fallback(base_structure: Atoms, config: StructureConfig) -> None:
    """Test NormalMode fallback to Rattle."""
    policy = NormalModePolicy()
    config.rattle_stdev = 0.1

    results = list(policy.generate(base_structure, config, n_structures=5))
    assert len(results) == 5

    for res in results:
        assert not np.allclose(res.positions, base_structure.positions)
