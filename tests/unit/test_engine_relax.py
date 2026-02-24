from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


@pytest.fixture
def mock_driver_relax() -> Any:
    with patch("pyacemaker.core.engine.LammpsDriver") as mock:
        yield mock

def test_lammps_engine_relax(mock_md_config: MDConfig, mock_driver_relax: Any, tmp_path: Path) -> None:
    # Setup mock driver
    driver_class = mock_driver_relax
    driver_instance = driver_class.return_value

    # Mock return of get_atoms
    relaxed_atoms = Atoms("He", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    driver_instance.get_atoms.return_value = relaxed_atoms

    engine = LammpsEngine(mock_md_config)
    initial_atoms = Atoms("He", positions=[[0.1, 0.1, 0.1]], cell=[10, 10, 10], pbc=True)

    # Create dummy potential file
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    # Call relax
    result_atoms = engine.relax(initial_atoms, pot_path)

    # Verify result
    assert result_atoms == relaxed_atoms
    assert result_atoms.get_chemical_symbols() == ["He"]

    # Verify script content
    driver_instance.run_file.assert_called_once()
    script_path = Path(driver_instance.run_file.call_args[0][0])
    script = script_path.read_text()

    assert "minimize" in script
    assert "min_style cg" in script
    assert "read_data" in script
    assert "pair_coeff" in script
    # Ensure no MD commands
    assert "fix npt" not in script
    assert "velocity all create" not in script

def test_lammps_engine_relax_missing_potential(mock_md_config: MDConfig, mock_driver_relax: Any) -> None:
    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H")

    with pytest.raises(FileNotFoundError):
        engine.relax(atoms, "nonexistent.yace")

def test_lammps_engine_relax_driver_fail(mock_md_config: MDConfig, mock_driver_relax: Any, tmp_path: Path) -> None:
    driver_instance = mock_driver_relax.return_value
    driver_instance.run_file.side_effect = RuntimeError("Minimization failed")

    engine = LammpsEngine(mock_md_config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    with pytest.raises(RuntimeError, match="Minimization failed"):
        engine.relax(atoms, pot_path)
