import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        hybrid_params={"lj/cut": "..."}
    )

def test_lammps_engine_run(mock_md_config: MDConfig) -> None:
    engine = LammpsEngine(mock_md_config)

    atoms = Atoms("H")
    result = engine.run(atoms, "potential.yace")

    assert isinstance(result, dict)
    assert "energy" in result
    assert "forces" in result
    assert "halted" in result
