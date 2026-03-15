import json
from pathlib import Path
file_path = Path("tests/integration/test_scenarios.py")
content = file_path.read_text()

# We need to correctly patch the test to add the missing nodes
target = """            "edges": [{"source": "node_001", "target": "node_002"}],
        }"""
new = """            "edges": [
                {"source": "node_001", "target": "node_002"},
                {"source": "node_002", "target": "node_003"}
            ],
        }"""

# I'll just rewrite the test
idx_start = content.find("def test_api_compile_intent_success")
idx_end = content.find("def test_api_compile_intent_invalid")
if idx_start != -1 and idx_end != -1:
    new_test = """def test_api_compile_intent_success(client: TestClient) -> None:
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

"""
    content = content[:idx_start] + new_test + content[idx_end:]
    file_path.write_text(content)
