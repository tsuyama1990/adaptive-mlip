import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario
from pyacemaker.interfaces.eon_driver import EONWrapper


@pytest.fixture
def integration_config():
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        mock_conf = MagicMock(spec=PyAceConfig)
        mock_conf.scenario = ScenarioConfig(
            name="fept_mgo",
            enabled=True,
            parameters={"num_depositions": 2, "fe_pt_ratio": 0.5, "write_intermediate_files": True}
        )
        mock_conf.eon = EONConfig(potential_path=path, enabled=True)
        # We need a valid MDConfig mostly for instantiation if not mocked
        mock_conf.md = MagicMock()
        yield mock_conf

def test_fept_mgo_integration(integration_config):
    # Setup mocks for heavy lifting
    mock_engine = MagicMock()
    # relax returns a copy
    mock_engine.relax.side_effect = lambda atoms, pot: atoms.copy()

    # We use real EONWrapper but mock the runner to avoid executing eonclient
    mock_runner = MagicMock()
    mock_runner.run.return_value.stdout = "EON simulation mocked output"

    wrapper = EONWrapper(integration_config.eon, runner=mock_runner)

    scenario = FePtMgoScenario(integration_config, engine=mock_engine, eon_wrapper=wrapper)

    # Run in temp dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            scenario.run()

            # Verify files
            assert Path("mgo_surface.xyz").exists()
            assert Path("deposited.xyz").exists()

            eon_work = Path("eon_work")
            assert eon_work.exists()
            assert (eon_work / "config.ini").exists()
            assert (eon_work / "pace_driver.py").exists()
            assert (eon_work / "pos.con").exists()

            # Verify config content thoroughly
            config_content = (eon_work / "config.ini").read_text()
            assert "[Main]" in config_content
            assert "job = akmc" in config_content
            assert "potential = command_line" in config_content
            assert "pace_driver.py" in config_content
            assert "supercell = [1, 1, 1]" in config_content

            # Verify driver script content
            driver_content = (eon_work / "pace_driver.py").read_text()
            assert "PACE_POTENTIAL_PATH" in driver_content
            assert "from ase.calculators.lammpsrun import LAMMPS" in driver_content
            assert "os.environ.get" in driver_content

            # Verify runner call
            assert mock_runner.run.called
            # Check command and environment
            args, kwargs = mock_runner.run.call_args
            cmd = args[0]
            assert cmd[0] == "eonclient"
            env = kwargs.get("env")
            assert env is not None
            assert "PACE_POTENTIAL_PATH" in env
            assert env["PACE_POTENTIAL_PATH"] == str(integration_config.eon.potential_path)

        finally:
            os.chdir(cwd)
