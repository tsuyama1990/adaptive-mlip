from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.main import app
from tests.conftest import create_test_config_dict


def test_scenario_01_a_successful_intent_payload() -> None:
    """
    SCENARIO-01-A [Priority: High] - Successful Intent Payload Processing
    """
    client = TestClient(app)
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
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
        ],
        "edges": [{"source": "node_001", "target": "node_002"}],
    }
    response = client.post("/api/v1/intent/compile", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["node_count"] == 2


def test_scenario_01_b_strict_rejection_advanced_params() -> None:
    """
    SCENARIO-01-B [Priority: High] - Strict Rejection of Advanced Parameters
    """
    client = TestClient(app)
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Pt",
        "lammps_command_string": "fix 1 all nve",
        "learning_rate": 0.001,
        "nodes": [],
        "edges": [],
    }
    response = client.post("/api/v1/intent/compile", json=payload)
    assert response.status_code == 422
    assert "lammps_command_string" in response.text
    assert "learning_rate" in response.text


def test_scenario_01_c_rejection_out_of_bounds_and_invalid_types() -> None:
    """
    Scenario 3: Rejection of Out-of-Bounds Intent Parameters and Invalid Types
    """
    client = TestClient(app)
    # Test out of bounds
    payload_oob = {"accuracy_speed_slider": 15, "target_material": "Pt", "nodes": [], "edges": []}
    resp1 = client.post("/api/v1/intent/compile", json=payload_oob)
    assert resp1.status_code == 400
    assert "Slider must be an integer between" in resp1.text

    # Test invalid string instead of int
    payload_type = {
        "accuracy_speed_slider": "high",
        "target_material": "Pt",
        "nodes": [],
        "edges": [],
    }
    resp2 = client.post("/api/v1/intent/compile", json=payload_type)
    assert resp2.status_code == 422
    assert "accuracy_speed_slider" in resp2.text

    # Test invalid Node type
    payload_node = {
        "accuracy_speed_slider": 5,
        "target_material": "Pt",
        "nodes": [
            {
                "id": "node_001",
                "type": "QUANTUM_MAGIC_NODE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Pt",
                    "lattice_constant": 3.92,
                },
            }
        ],
        "edges": [],
    }
    resp3 = client.post("/api/v1/intent/compile", json=payload_node)
    assert resp3.status_code == 422
    assert "type" in resp3.text


def test_scenario_01_01_hello_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario 01-01: "Hello Config"
    Objective: Verify that the system can load a configuration file and initialize.
    """
    # 1. Preparation
    monkeypatch.chdir(tmp_path)
    # Create dummy pseudo files
    (tmp_path / "H.UPF").touch()
    (tmp_path / "O.UPF").touch()

    config_file = tmp_path / "config.yaml"
    # Create valid config manually as before
    path = config_file
    config_content = """
project_name: UAT_Project
structure:
    elements: [H, O]
    supercell_size: [1, 1, 1]
dft:
    code: qe
    functional: PBE
    kpoints_density: 0.04
    encut: 500.0
    pseudopotentials:
        H: H.UPF
        O: O.UPF
training:
    potential_type: ace
    cutoff_radius: 5.0
    max_basis_size: 500
md:
    temperature: 300.0
    pressure: 0.0
    timestep: 0.001
    n_steps: 1000
    uncertainty_threshold: 0.1
    check_interval: 50
workflow:
    max_iterations: 10
    state_file_path: uat_state.json
"""
    path.write_text(config_content)

    # 2. Action
    from pyacemaker.main import main

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(config=str(config_file), dry_run=True, scenario=None),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_scenario_01_02_guardrails_check_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Scenario 01-02: "Guardrails Check" (Temperature)
    Objective: Verify that the system rejects invalid physical parameters (negative temperature).
    """
    # 1. Preparation
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    # We use Pydantic model directly validation
    config_dict = create_test_config_dict()
    config_dict["md"]["temperature"] = -50.0

    # 2. Action & 3. Expectation
    # Pydantic raises ValidationError
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PyAceConfig.model_validate(config_dict)


def test_scenario_01_02_guardrails_check_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Scenario 01-02: "Guardrails Check" (Cutoff)
    Objective: Verify that the system rejects invalid physical parameters (negative cutoff).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    config_dict = create_test_config_dict()
    config_dict["training"]["cutoff_radius"] = -1.0
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PyAceConfig.model_validate(config_dict)
