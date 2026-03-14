from fastapi.testclient import TestClient

from pyacemaker.main import app

client = TestClient(app)


def test_compile_intent_success() -> None:
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Al",
        "nodes": [
            {
                "id": "node1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Al",
                    "lattice_constant": 4.0,
                },
            },
            {
                "id": "node2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
        ],
        "edges": [{"source": "node1", "target": "node2"}],
    }

    response = client.post("/api/v1/intent/compile", json=payload)

    assert response.status_code == 200
    config_dict = response.json()
    assert config_dict["project_name"] == "intent_driven_project"
    assert config_dict["structure"]["elements"] == ["Al"]
    assert "quantum_espresso" in config_dict["dft"]["code"]


def test_compile_intent_branching_rejection() -> None:
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Al",
        "nodes": [
            {
                "id": "node1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Al",
                    "lattice_constant": 4.0,
                },
            },
            {
                "id": "node2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
            {
                "id": "node3",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
        ],
        "edges": [{"source": "node1", "target": "node2"}, {"source": "node1", "target": "node3"}],
    }

    response = client.post("/api/v1/intent/compile", json=payload)

    assert response.status_code == 400
    assert "Parallel active learning loops" in response.json()["detail"]
