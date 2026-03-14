import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.scenario import (
    ActiveLearningData,
    DagNode,
    Edge,
    InitialStructureData,
    IntentRequest,
    MaceTrainingData,
    NodeType,
    ScenarioConfig,
)


def test_scenario_config_valid() -> None:
    config = ScenarioConfig(
        name="fept_mgo",
        parameters={"fe_pt_ratio": 0.5, "steps": 100},
        enabled=True,
    )
    assert config.name == "fept_mgo"
    assert config.parameters["fe_pt_ratio"] == 0.5
    assert config.enabled is True


def test_scenario_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ScenarioConfig(name="test", extra_field="forbidden")  # type: ignore[call-arg]


def test_initial_structure_data_valid() -> None:
    data = InitialStructureData(
        type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Pt", lattice_constant=3.92
    )
    assert data.chemical_symbol == "Pt"
    assert data.lattice_constant == 3.92


def test_initial_structure_data_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE,
            chemical_symbol="Pt",
            lattice_constant=3.92,
            extra="forbidden",
        )  # type: ignore[call-arg]


def test_intent_request_valid_dag() -> None:
    req = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Pt",
        nodes=[
            DagNode(
                id="n1",
                type=NodeType.INITIAL_STRUCTURE,
                data=InitialStructureData(
                    type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Pt", lattice_constant=3.9
                ),
            ),
            DagNode(
                id="n2",
                type=NodeType.ACTIVE_LEARNING_LOOP,
                data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
            ),
        ],
        edges=[Edge(source="n1", target="n2")],
    )
    assert req.accuracy_speed_slider == 5
    assert req.target_material == "Pt"
    assert len(req.nodes) == 2


def test_intent_request_invalid_slider() -> None:
    with pytest.raises(ValidationError, match="accuracy_speed_slider"):
        IntentRequest(
            accuracy_speed_slider=11,  # Out of bounds
            target_material="Pt",
            nodes=[],
            edges=[],
        )
    with pytest.raises(ValidationError, match="accuracy_speed_slider"):
        IntentRequest(
            accuracy_speed_slider=0,  # Out of bounds
            target_material="Pt",
            nodes=[],
            edges=[],
        )


def test_intent_request_cycle_detection() -> None:
    with pytest.raises(ValidationError, match="cycle"):
        IntentRequest(
            accuracy_speed_slider=5,
            target_material="Pt",
            nodes=[
                DagNode(
                    id="n1",
                    type=NodeType.INITIAL_STRUCTURE,
                    data=InitialStructureData(
                        type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Pt", lattice_constant=3.9
                    ),
                ),
                DagNode(
                    id="n2",
                    type=NodeType.ACTIVE_LEARNING_LOOP,
                    data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
                ),
                DagNode(
                    id="n3",
                    type=NodeType.MACE_TRAINING,
                    data=MaceTrainingData(type=NodeType.MACE_TRAINING),
                ),
            ],
            edges=[
                Edge(source="n1", target="n2"),
                Edge(source="n2", target="n3"),
                Edge(source="n3", target="n1"),  # Cycle!
            ],
        )


def test_intent_request_invalid_edge_nodes() -> None:
    with pytest.raises(ValidationError, match="not found in nodes"):
        IntentRequest(
            accuracy_speed_slider=5,
            target_material="Pt",
            nodes=[
                DagNode(
                    id="n1",
                    type=NodeType.INITIAL_STRUCTURE,
                    data=InitialStructureData(
                        type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Pt", lattice_constant=3.9
                    ),
                ),
            ],
            edges=[
                Edge(source="n1", target="n2"),  # n2 doesn't exist
            ],
        )


def test_intent_request_invalid_node_type() -> None:
    with pytest.raises(ValidationError):
        IntentRequest(
            accuracy_speed_slider=5,
            target_material="Pt",
            nodes=[
                DagNode(id="n1", type="MAGIC_NODE", data={"type": "MAGIC_NODE"}),  # type: ignore[arg-type]
            ],
            edges=[],
        )
