from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tempfile
from ase import Atoms

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario, DepositionManager


@pytest.fixture
def mock_config():
    # Mock minimal config for scenario
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        mock_conf = MagicMock(spec=PyAceConfig)
        mock_conf.scenario = ScenarioConfig(
            name="fept_mgo",
            enabled=True,
            parameters={"num_depositions": 2, "fe_pt_ratio": 0.5, "random_seed": 42}
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


def test_fept_deposit_atoms_deterministic(mock_config):
    mock_engine = MagicMock()
    # Configure engine.relax to return a modified structure (simulating relaxation)
    def mock_relax(atoms, pot):
        atoms_copy = atoms.copy()
        atoms_copy.positions[0] += 0.1
        return atoms_copy

    mock_engine.relax.side_effect = mock_relax

    scenario1 = FePtMgoScenario(mock_config, engine=mock_engine)
    slab1 = scenario1._generate_surface()
    # Manually instantiate DepositionManager as it's now decoupled
    pot_path1 = scenario1._get_potential_path()
    dm1 = DepositionManager(mock_engine, scenario1.params, pot_path1)
    deposited1 = dm1.deposit(slab1)

    # Run again with same seed
    scenario2 = FePtMgoScenario(mock_config, engine=mock_engine)
    slab2 = scenario2._generate_surface()
    pot_path2 = scenario2._get_potential_path()
    dm2 = DepositionManager(mock_engine, scenario2.params, pot_path2)
    deposited2 = dm2.deposit(slab2)

    # Check positions match (determinism)
    import numpy as np
    assert np.allclose(deposited1.get_positions(), deposited2.get_positions())
    assert deposited1.get_chemical_symbols() == deposited2.get_chemical_symbols()


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


def test_fept_relaxation_failure(mock_config):
    # Test relaxation failure handling via DepositionManager
    mock_engine = MagicMock()
    mock_engine.relax.side_effect = Exception("Relaxation failed")

    scenario = FePtMgoScenario(mock_config, engine=mock_engine)
    slab = scenario._generate_surface()
    dm = DepositionManager(mock_engine, scenario.params, Path("pot"))

    with pytest.raises(RuntimeError) as excinfo:
        dm.deposit(slab)

    assert "Deposition failed" in str(excinfo.value)

def test_deposition_manager_error(mock_config):
    mock_engine = MagicMock()
    # relaxation fails
    mock_engine.relax.side_effect = Exception("Fail")

    scenario = FePtMgoScenario(mock_config, engine=mock_engine)
    dm = DepositionManager(mock_engine, scenario.params, Path("pot"))
    slab = scenario._generate_surface()

    with pytest.raises(RuntimeError) as excinfo:
        dm.deposit(slab)
    assert "Deposition failed" in str(excinfo.value)
