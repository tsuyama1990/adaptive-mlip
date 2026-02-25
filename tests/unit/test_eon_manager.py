import subprocess
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.core.eon_manager import EONManager
from pyacemaker.domain_models.eon import EONConfig


class TestEONManager:
    @pytest.fixture
    def config(self, tmp_path):
        pot_path = tmp_path / "pot.yace"
        pot_path.touch()
        return EONConfig(
            enabled=True,
            eon_executable="eonclient",
            potential_path=pot_path,
            temperature=300.0,
            akmc_steps=100
        )

    @pytest.fixture
    def manager(self, config):
        return EONManager(config)

    def test_write_driver(self, manager, tmp_path):
        driver_path = manager._write_driver(tmp_path, manager.config.potential_path)

        assert driver_path.exists()
        content = driver_path.read_text()
        assert "from pyacemaker.interfaces.eon_driver import run_driver" in content
        assert str(manager.config.potential_path.absolute()) in content

    @patch("pyacemaker.interfaces.process.SubprocessRunner.run")
    def test_run_success(self, mock_run, manager, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)

        structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
        result = manager.run(tmp_path, manager.config.potential_path, structure)

        assert result["halted"] is False
        assert (tmp_path / "reactant.con").exists()
        assert (tmp_path / "config.ini").exists()
        assert (tmp_path / "pace_driver.py").exists()

    @patch("pyacemaker.interfaces.process.SubprocessRunner.run")
    def test_run_halted(self, mock_run, manager, tmp_path):
        # Raise CalledProcessError with code 100
        mock_run.side_effect = subprocess.CalledProcessError(100, cmd="eonclient")

        structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
        result = manager.run(tmp_path, manager.config.potential_path, structure)

        assert result["halted"] is True
        assert result["halt_structure_path"] == str(tmp_path / "bad_structure.cfg")

    @patch("pyacemaker.interfaces.process.SubprocessRunner.run")
    def test_run_failure(self, mock_run, manager, tmp_path):
        mock_run.side_effect = subprocess.CalledProcessError(1, cmd="eonclient", stderr="Error")

        structure = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])

        with pytest.raises(RuntimeError, match="EON execution failed"):
            manager.run(tmp_path, manager.config.potential_path, structure)
