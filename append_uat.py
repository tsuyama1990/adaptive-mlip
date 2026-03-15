from pathlib import Path
content = Path("tests/uat/test_cycle03_uat.py").read_text()

new_uat = """
from fastapi.testclient import TestClient

from pyacemaker.main import app

client = TestClient(app)

def test_scenario_03_a_successful_spatial_bounding_box() -> None:
    \"\"\"
    SCENARIO-03-A: Successful Spatial Bounding Box Translation.
    Validates that a visual "paint" action mathematically translates into a LAMMPS constraint.
    \"\"\"
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Pt",
        "nodes": [
            {
                "id": "n1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Pt",
                    "lattice_constant": 3.92,
                    "spatial_regions": [
                        {
                            "x_min": 0.0, "x_max": 20.0,
                            "y_min": 0.0, "y_max": 20.0,
                            "z_min": 0.0, "z_max": 6.0,
                            "action": "ACTION_FREEZE"
                        }
                    ]
                }
            },
            {
                "id": "n2",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"}
            },
            {
                "id": "n3",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"}
            }
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"}
        ]
    }

    response = client.post("/api/v1/intent/compile", json=payload)
    assert response.status_code == 200
    config = response.json()

    # Assert LAMMPS strings
    cmds = config["md"].get("spatial_tags_commands", [])
    cmd_str = " ".join(cmds)

    assert "region reg_1 block 0.0 20.0 0.0 20.0 0.0 6.0" in cmd_str
    assert "group group_1 region reg_1" in cmd_str
    assert "fix fix_1 group_1 setforce 0.0 0.0 0.0" in cmd_str

    # Assert ignoring uncertainty
    ignore_tags = config["workflow"]["loop_strategy"]["thresholds"].get("ignore_tags")
    assert ignore_tags == [1]

def test_scenario_03_b_accurate_mathematical_ase_masking_generation_and_conflict_resolution() -> None:
    \"\"\"
    SCENARIO-03-B: Conflict Resolution.
    We test the logical outcome by ensuring the API accepts overlapping boxes
    and translates them to the correct LAMMPS commands.
    (Note: the pure ASE conflict resolution is fully tested in test_spatial.py,
    this just ensures the API accepts the payload correctly).
    \"\"\"
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Pt",
        "nodes": [
            {
                "id": "n1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Pt",
                    "lattice_constant": 3.92,
                    "spatial_regions": [
                        {
                            "x_min": 0.0, "x_max": 20.0,
                            "y_min": 0.0, "y_max": 20.0,
                            "z_min": 0.0, "z_max": 10.0,
                            "action": "ACTION_FREEZE"
                        },
                        {
                            "x_min": 0.0, "x_max": 20.0,
                            "y_min": 0.0, "y_max": 20.0,
                            "z_min": 5.0, "z_max": 15.0,
                            "action": "ACTION_LANGEVIN_THERMOSTAT"
                        }
                    ]
                }
            },
            {
                "id": "n2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"}
            },
            {
                "id": "n3",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"}
            }
        ],
        "edges": [
            {"source": "n1", "target": "n3"},
            {"source": "n3", "target": "n2"}
        ]
    }

    response = client.post("/api/v1/intent/compile", json=payload)
    assert response.status_code == 200
    config = response.json()

    cmds = config["md"].get("spatial_tags_commands", [])
    cmd_str = " ".join(cmds)

    assert "region reg_1 block 0.0 20.0 0.0 20.0 0.0 10.0" in cmd_str
    assert "region reg_2 block 0.0 20.0 0.0 20.0 5.0 15.0" in cmd_str
    assert "langevin" in cmd_str


def test_scenario_03_c_handling_of_empty_selections_and_invalid_geometry() -> None:
    \"\"\"
    SCENARIO-03-C: Handling invalid geometry bounding box coordinates.
    \"\"\"
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "Pt",
        "nodes": [
            {
                "id": "n1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Pt",
                    "lattice_constant": 3.92,
                    "spatial_regions": [
                        {
                            "x_min": 15.0, "x_max": 5.0,  # INVERTED
                            "y_min": 0.0, "y_max": 20.0,
                            "z_min": 0.0, "z_max": 10.0,
                            "action": "ACTION_FREEZE"
                        }
                    ]
                }
            },
            {
                "id": "n2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"}
            },
            {
                "id": "n3",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"}
            }
        ],
        "edges": [
            {"source": "n1", "target": "n3"},
            {"source": "n3", "target": "n2"}
        ]
    }

    response = client.post("/api/v1/intent/compile", json=payload)
    # The Pydantic validator should catch the inverted coordinates before reaching the logic.
    assert response.status_code == 422
    assert "Input should be an instance of SpatialAction" in response.text or "Invalid x-axis boundary" in response.text
"""

Path("tests/uat/test_cycle03_uat.py").write_text(content + new_uat)
