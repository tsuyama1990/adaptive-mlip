from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        temperature=300.0,
        timestep=0.001,
        n_steps=100,
        potential_path=None
    )

@patch("pyacemaker.core.engine.LammpsDriver")
def test_engine_integration_mocked(mock_driver_cls: MagicMock, mock_md_config: MDConfig, tmp_path: Path) -> None:
    """
    Test LammpsEngine with mocked driver.
    """
    mock_driver = mock_driver_cls.return_value
    mock_driver.extract_variable.return_value = -100.0 # Energy
    mock_driver.get_forces.return_value = MagicMock(tolist=lambda: [[0.0, 0.0, 0.0]])
    mock_driver.get_stress.return_value = MagicMock(tolist=lambda: [0.0]*6)

    engine = LammpsEngine(mock_md_config)

    # Create dummy potential
    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    # Mock file manager
    with patch.object(engine.file_manager, "prepare_workspace") as mock_prep:
        mock_ctx = MagicMock()
        mock_ctx.name = str(tmp_path)
        mock_prep.return_value = (mock_ctx, tmp_path/"data.lmp", tmp_path/"dump.lammpstrj", tmp_path/"log.lammps", ["Fe"])

        # Must have a cell to pass validation if we validate structure
        atoms = Atoms("Fe", positions=[[0,0,0]], cell=[10,10,10], pbc=True)
        result = engine.run(atoms, pot_path)

        assert result.energy == -100.0
        mock_driver.run_file.assert_called()
