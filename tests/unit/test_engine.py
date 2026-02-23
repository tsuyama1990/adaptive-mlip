from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_lammps_engine_run(mock_md_config: MDConfig) -> None:
    engine = LammpsEngine(mock_md_config)

    atoms = Atoms("H")
    result = engine.run(atoms, "potential.yace")

    assert isinstance(result, dict)
    assert "energy" in result
    assert "forces" in result
    assert "halted" in result
