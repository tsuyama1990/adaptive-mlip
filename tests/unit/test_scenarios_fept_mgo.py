from pathlib import Path
from unittest.mock import MagicMock, patch

import tempfile
import pytest
from ase import Atoms

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario


@pytest.fixture
def mock_config():
    # Mock minimal config for scenario
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        mock_conf = MagicMock(spec=PyAceConfig)
        mock_conf.scenario = ScenarioConfig(
            name="fept_mgo",
            enabled=True,
            parameters={"num_depositions": 2, "fe_pt_ratio": 0.5}
        )
        mock_conf.eon = EONConfig(potential_path=path)
        mock_conf.md = MagicMock() # Mock MD config
        yield mock_conf


def test_fept_init(mock_config):
    scenario = FePtMgoScenario(mock_config)
    assert scenario.name == "fept_mgo"
    assert scenario.eon_wrapper is None  # Should be None initially or mocked


def test_fept_generate_surface(mock_config):
    scenario = FePtMgoScenario(mock_config)
    surface = scenario._generate_surface()
    assert isinstance(surface, Atoms)
    assert len(surface) > 0
    # MgO rocksalt structure check
    syms = surface.get_chemical_symbols()
    assert "Mg" in syms and "O" in syms


def test_fept_deposit_atoms(mock_config):
    mock_engine = MagicMock()
    # Configure engine.relax to return a modified structure (simulating relaxation)
    def mock_relax(atoms, pot):
        atoms_copy = atoms.copy()
        # Just return copy, maybe shift an atom slightly
        atoms_copy.positions[0] += 0.1
        return atoms_copy

    mock_engine.relax.side_effect = mock_relax

    scenario = FePtMgoScenario(mock_config, engine=mock_engine)

    initial_slab = scenario._generate_surface()
    initial_count = len(initial_slab)

    deposited = scenario._deposit_atoms(initial_slab)

    # Check atoms added
    expected_additions = mock_config.scenario.parameters["num_depositions"]
    assert len(deposited) == initial_count + expected_additions

    # Check relaxation called
    assert mock_engine.relax.call_count == expected_additions


def test_fept_run_flow(mock_config):
    mock_config.eon.enabled = True # Enable EON for this test
    mock_engine = MagicMock()
    mock_engine.relax.side_effect = lambda atoms, pot: atoms.copy()

    mock_eon = MagicMock()

    scenario = FePtMgoScenario(mock_config, engine=mock_engine, eon_wrapper=mock_eon)

    # Mock write to avoid file creation in CWD during test
    with patch("pyacemaker.scenarios.fept_mgo.write") as mock_write:
        scenario.run()

    # Verify sequence
    assert mock_write.call_count >= 3 # surface, deposited, eon structure
    assert mock_eon.generate_config.called
    assert mock_eon.run.called
