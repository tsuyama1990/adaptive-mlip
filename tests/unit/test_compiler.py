import pytest

from pyacemaker.core.exceptions import CompilerError
from pyacemaker.domain_models.compiler import SemanticCompiler
from pyacemaker.domain_models.scenario import (
    ActiveLearningData,
    DagNode,
    Edge,
    InitialStructureData,
    IntentRequest,
    MaceTrainingData,
    NodeType,
)


def create_mock_intent(
    nodes: list[DagNode], edges: list[Edge], target_material: str = "Al", slider: int = 5
) -> IntentRequest:
    return IntentRequest(
        accuracy_speed_slider=slider,
        target_material=target_material,
        nodes=nodes,
        edges=edges,
    )


def test_compiler_topological_sort_success() -> None:
    node1 = DagNode(
        id="n1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Al", lattice_constant=4.0
        ),
    )
    node2 = DagNode(
        id="n2", type=NodeType.MACE_TRAINING, data=MaceTrainingData(type=NodeType.MACE_TRAINING)
    )
    node3 = DagNode(
        id="n3",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )

    # Valid linear graph
    edges = [Edge(source="n1", target="n2"), Edge(source="n2", target="n3")]
    intent = create_mock_intent([node1, node2, node3], edges)

    config = SemanticCompiler.compile(intent)

    assert config is not None
    assert config.project_name == "intent_driven_project"
    assert config.structure.elements == ["Al"]
    assert config.md.timestep > 0.0
    assert config.training.potential_type == "mace"
    assert config.workflow.max_iterations > 0


def test_compiler_invalid_sequence() -> None:
    node1 = DagNode(
        id="n1",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )
    node2 = DagNode(
        id="n2",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Al", lattice_constant=4.0
        ),
    )

    edges = [Edge(source="n1", target="n2")]
    intent = create_mock_intent([node1, node2], edges)

    with pytest.raises(CompilerError, match="INITIAL_STRUCTURE node must precede"):
        SemanticCompiler.compile(intent)


def test_compiler_branching_rejection() -> None:
    node1 = DagNode(
        id="n1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Al", lattice_constant=4.0
        ),
    )
    node2 = DagNode(
        id="n2",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )
    node3 = DagNode(
        id="n3",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )

    edges = [Edge(source="n1", target="n2"), Edge(source="n1", target="n3")]
    intent = create_mock_intent([node1, node2, node3], edges)

    with pytest.raises(CompilerError, match="Parallel active learning loops"):
        SemanticCompiler.compile(intent)


def test_compiler_intelligent_defaults() -> None:
    node1 = DagNode(
        id="n1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Al", lattice_constant=4.0
        ),
    )
    node2 = DagNode(
        id="n2",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )
    node3 = DagNode(
        id="n3", type=NodeType.MACE_TRAINING, data=MaceTrainingData(type=NodeType.MACE_TRAINING)
    )

    edges = [Edge(source="n1", target="n3"), Edge(source="n3", target="n2")]
    intent = create_mock_intent([node1, node2, node3], edges, slider=8)

    config = SemanticCompiler.compile(intent)

    # Check physical defaults injection
    assert config.dft.encut >= 30.0
    assert config.dft.kpoints_density > 0.0
    assert config.md.temperature == 300.0
    assert config.md.n_steps > 0
    assert config.md.timestep <= 2.0
    assert config.workflow.loop_strategy.thresholds.threshold_call_dft > 0.0
