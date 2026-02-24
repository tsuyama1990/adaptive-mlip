from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock(spec=PyAceConfig)
    config.scenario = ScenarioConfig(
        name="fept_mgo",
        parameters={"num_depositions": 10, "temperature": 600.0},
        enabled=True
    )
    config.md = MagicMock(spec=MDConfig)
    config.md.potential_path = Path("pot.yace")
    config.eon = MagicMock(spec=EONConfig)
    config.eon.potential_path = Path("pot.yace")
    config.eon.enabled = True
    return config


@pytest.fixture
def mock_md_engine() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_eon_wrapper() -> MagicMock:
    return MagicMock()


def test_fept_mgo_initialization(mock_config: MagicMock) -> None:
    scenario = FePtMgoScenario(mock_config)
    assert scenario.config == mock_config
    assert scenario.name == "fept_mgo"


def test_fept_mgo_run(
    mock_config: MagicMock, mock_md_engine: MagicMock, mock_eon_wrapper: MagicMock
) -> None:
    with patch("pyacemaker.scenarios.fept_mgo.LammpsEngine", return_value=mock_md_engine), \
         patch("pyacemaker.scenarios.fept_mgo.EONWrapper", return_value=mock_eon_wrapper), \
         patch("pyacemaker.scenarios.fept_mgo.Atoms"), \
         patch("pyacemaker.scenarios.fept_mgo.surface") as mock_surface, \
         patch("pyacemaker.scenarios.fept_mgo.write"):

        scenario = FePtMgoScenario(mock_config)
        scenario.run()

        # Check if steps were executed
        # 1. Surface generation
        mock_surface.assert_called()

        # 2. Deposition (MD Engine calls)
        assert mock_md_engine.relax.call_count >= 1

        # 3. EON Run
        mock_eon_wrapper.run.assert_called()


def test_fept_mgo_run_missing_config() -> None:
    config = MagicMock(spec=PyAceConfig)
    config.scenario = None  # No scenario config provided

    scenario = FePtMgoScenario(config)
    with pytest.raises(ValueError, match="Scenario configuration is missing"):
        scenario.run()
