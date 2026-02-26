from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import LocalGenerationStrategy, StructureConfig


def test_generate_local_rattle():
    config = StructureConfig(
        elements=["H"],
        supercell_size=[1,1,1],
        local_generation_strategy=LocalGenerationStrategy.RANDOM_DISPLACEMENT
    )
    generator = StructureGenerator(config)
    base = Atoms("H", positions=[[0,0,0]], cell=[10,10,10])

    candidates = list(generator.generate_local(base, n_candidates=5))
    assert len(candidates) == 5
    # Check simple property: positions changed
    assert any(c.positions[0].tolist() != [0,0,0] for c in candidates)

def test_generate_local_md_burst():
    # Mock PolicyFactory to return a Mock Policy, so we don't depend on Engine logic here (tested in policy test)
    # OR mock Engine passed to generator.

    config = StructureConfig(
        elements=["H"],
        supercell_size=[1,1,1],
        local_generation_strategy=LocalGenerationStrategy.MD_MICRO_BURST
    )
    generator = StructureGenerator(config)
    base = Atoms("H", positions=[[0,0,0]], cell=[10,10,10])

    mock_engine = MagicMock()

    # We need to ensure MDMicroBurstPolicy is used and engine is passed.
    # Since we use real classes, we can check behavior.

    # MDMicroBurstPolicy logic relies on engine.
    # If we pass mock engine, it should call engine.run/config.model_copy

    # Mock engine.config.model_copy
    mock_engine.config.model_copy.return_value = MagicMock()

    # Mock engine.run to return result with trajectory
    from pyacemaker.domain_models.md import MDSimulationResult
    mock_result = MagicMock(spec=MDSimulationResult)
    mock_result.trajectory_path = "dummy.traj"

    # Mock type(engine)(config)
    class MockEngineClass:
        def __init__(self, config) -> None:
            self.config = config
        def run(self, s, p):
            return mock_result

    real_mock_engine = MockEngineClass(MagicMock())
    real_mock_engine.config.model_copy.return_value = MagicMock()

    with patch("pyacemaker.core.policy.read") as mock_read:
        mock_read.return_value = Atoms("He")

        candidates = list(generator.generate_local(base, n_candidates=1, engine=real_mock_engine, potential="pot"))

        assert len(candidates) == 1
        assert candidates[0].get_chemical_symbols() == ["He"]
