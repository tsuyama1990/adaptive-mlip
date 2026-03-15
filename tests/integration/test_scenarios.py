import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.domain_models.scenario import ScenarioConfig
from pyacemaker.interfaces.eon_driver import EONWrapper
from pyacemaker.main import app
from pyacemaker.scenarios.fept_mgo import FePtMgoScenario


@pytest.fixture
def integration_config() -> Generator[PyAceConfig, None, None]:
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        mock_conf = MagicMock(spec=PyAceConfig)
        mock_conf.scenario = ScenarioConfig(
            name="fept_mgo",
            enabled=True,
            parameters={"num_depositions": 2, "fe_pt_ratio": 0.5, "write_intermediate_files": True},
        )
        mock_conf.eon = EONConfig(potential_path=path, enabled=True)
        # We need a valid MDConfig mostly for instantiation if not mocked
        mock_conf.md = MagicMock()
        yield mock_conf


def test_fept_mgo_integration(integration_config: PyAceConfig) -> None:
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
        cwd = Path.cwd()
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


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_api_compile_intent_success(client: TestClient) -> None:
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Pt",
        "nodes": [
            {
                "id": "node_001",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Pt",
                    "lattice_constant": 3.92,
                },
            },
            {
                "id": "node_002",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"},
            },
            {
                "id": "node_003",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
        ],
        "edges": [{"source": "node_001", "target": "node_002"}, {"source": "node_002", "target": "node_003"}],
    }
    response = client.post("/api/v1/intent/compile", json=payload)
    assert response.status_code == 200

def test_api_compile_intent_invalid_slider(client: TestClient) -> None:
    payload = {"accuracy_speed_slider": 15, "target_material": "Pt", "nodes": [], "edges": []}
    response = client.post("/api/v1/intent/compile", json=payload)
    assert response.status_code == 422
    assert "accuracy_speed_slider" in response.text

def test_spatial_region_compiler_integration() -> None:
    from pyacemaker.domain_models.compiler import SemanticCompiler
    from pyacemaker.domain_models.gui_schema import SpatialAction, SpatialRegion
    from pyacemaker.domain_models.scenario import (
        ActiveLearningData,
        DagNode,
        Edge,
        InitialStructureData,
        IntentRequest,
        MaceTrainingData,
        NodeType,
    )

    # Create spatial regions
    freeze_region = SpatialRegion(
        x_min=0.0, x_max=10.0,
        y_min=0.0, y_max=10.0,
        z_min=0.0, z_max=5.0,
        action=SpatialAction.ACTION_FREEZE
    )
    thermostat_region = SpatialRegion(
        x_min=0.0, x_max=10.0,
        y_min=0.0, y_max=10.0,
        z_min=5.0, z_max=15.0,
        action=SpatialAction.ACTION_LANGEVIN_THERMOSTAT
    )

    node_init = DagNode(
        id="n1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE,
            chemical_symbol="Pt",
            lattice_constant=3.92,
            spatial_regions=[freeze_region, thermostat_region]
        )
    )
    node_train = DagNode(
        id="n2",
        type=NodeType.MACE_TRAINING,
        data=MaceTrainingData(type=NodeType.MACE_TRAINING)
    )
    node_loop = DagNode(
        id="n3",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP)
    )
    edge1 = Edge(source="n1", target="n2")
    edge2 = Edge(source="n2", target="n3")

    intent = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Pt",
        nodes=[node_init, node_train, node_loop],
        edges=[edge1, edge2]
    )

    config = SemanticCompiler.compile(intent)

    # Assert MDConfig contains the generated LAMMPS strings
    assert config.md.spatial_tags_commands is not None
    commands = " ".join(config.md.spatial_tags_commands)
    assert "region reg_1 block 0.0 10.0 0.0 10.0 0.0 5.0" in commands
    assert "group group_1 region reg_1" in commands
    assert "fix fix_1 group_1 setforce 0.0 0.0 0.0" in commands

    assert "region reg_2 block 0.0 10.0 0.0 10.0 5.0 15.0" in commands
    assert "group group_2 region reg_2" in commands
    assert "langevin" in commands

    # Assert ignore tags are propagated
    assert config.workflow.loop_strategy.thresholds.ignore_tags == [1]
