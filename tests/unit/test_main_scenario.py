import contextlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.main import main


@pytest.fixture
def mock_deps():
    with patch("pyacemaker.main.load_config") as mock_load, \
         patch("pyacemaker.main.setup_logger") as mock_logger, \
         patch("pyacemaker.main.Orchestrator") as mock_orch, \
         patch("pyacemaker.main.FePtMgoScenario") as mock_scenario:

        config = MagicMock(spec=PyAceConfig)
        config.project_name = "test"
        config.scenario = ScenarioConfig(name="fept_mgo")
        config.logging = MagicMock()
        mock_load.return_value = config

        yield mock_load, mock_logger, mock_orch, mock_scenario

def test_main_with_scenario_flag(mock_deps):
    mock_load, _, mock_orch, mock_scenario = mock_deps

    with patch.object(sys, 'argv', ["main.py", "--config", "config.yaml", "--scenario", "fept_mgo"]), \
         contextlib.suppress(SystemExit):
        main()

    mock_scenario.assert_called()
    mock_orch.assert_not_called()

def test_main_without_scenario_flag(mock_deps):
    mock_load, _, mock_orch, mock_scenario = mock_deps

    with patch.object(sys, 'argv', ["main.py", "--config", "config.yaml"]), \
         contextlib.suppress(SystemExit):
        main()

    # Standard behavior
    mock_orch.assert_called()
    mock_orch.return_value.run.assert_called()
    mock_scenario.assert_not_called()
