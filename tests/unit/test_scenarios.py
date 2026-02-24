from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario


@pytest.fixture
def mock_config():
    # Create a minimal config mock
    config = MagicMock(spec=PyAceConfig)
    config.scenario = ScenarioConfig(name="fept_mgo")
    config.eon = EONConfig(enabled=True, temperature=500.0)
    config.md = MagicMock(spec=MDConfig)
    config.project_name = "test_project"
    return config

def test_scenario_init(mock_config):
    # Need to set md config attributes used in LammpsEngine
    mock_config.md.neighbor_skin = 2.0
    mock_config.md.timestep = 0.001
    mock_config.md.hybrid_potential = False
    mock_config.md.fix_halt = False

    scenario = FePtMgoScenario(mock_config)
    assert scenario.config == mock_config
    assert scenario.name == "fept_mgo"

@patch("pyacemaker.scenarios.fept_mgo.LammpsEngine")
@patch("pyacemaker.scenarios.fept_mgo.EONWrapper")
def test_scenario_run(mock_eon, mock_engine, mock_config):
    scenario = FePtMgoScenario(mock_config)

    # Mock internal steps
    scenario.generate_surface = MagicMock()
    scenario.deposit_atoms = MagicMock()
    scenario.run_akmc = MagicMock()
    scenario.visualize = MagicMock()

    scenario.run()

    scenario.generate_surface.assert_called_once()
    scenario.deposit_atoms.assert_called_once()
    scenario.run_akmc.assert_called_once()
    scenario.visualize.assert_called_once()

def test_generate_surface_step(mock_config):
    scenario = FePtMgoScenario(mock_config)

    # Use context manager to mock ase.build functions
    # Note: 'bulk' and 'surface' are imported directly in fept_mgo.py
    # so we must patch them there.
    with patch("pyacemaker.scenarios.fept_mgo.bulk") as mock_bulk, \
         patch("pyacemaker.scenarios.fept_mgo.surface") as mock_surface, \
         patch("pyacemaker.scenarios.fept_mgo.write") as mock_write:

        mock_slab = MagicMock()
        mock_surface.return_value = mock_slab
        mock_slab.repeat.return_value = mock_slab

        result = scenario.generate_surface()

        mock_bulk.assert_called()
        mock_surface.assert_called()
        mock_write.assert_called()
        assert result == mock_slab
