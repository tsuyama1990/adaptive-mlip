from enum import StrEnum
from typing import Annotated, Literal

import networkx as nx
from ase.data import chemical_symbols
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyacemaker.domain_models.gui_schema import SpatialRegion


class NodeType(StrEnum):
    INITIAL_STRUCTURE = "INITIAL_STRUCTURE"
    MACE_TRAINING = "MACE_TRAINING"
    ACTIVE_LEARNING_LOOP = "ACTIVE_LEARNING_LOOP"
    EON_TRANSITION_SEARCH = "EON_TRANSITION_SEARCH"


class InitialStructureData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)
    type: Literal[NodeType.INITIAL_STRUCTURE] = Field(NodeType.INITIAL_STRUCTURE)
    chemical_symbol: str = Field(..., description="The chemical symbol")
    lattice_constant: float = Field(..., description="The lattice constant")
    spatial_regions: list[SpatialRegion] | None = Field(default=None, description="Optional list of spatial regions to apply constraints.")


class MaceTrainingData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal[NodeType.MACE_TRAINING] = Field(NodeType.MACE_TRAINING)


class ActiveLearningData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal[NodeType.ACTIVE_LEARNING_LOOP] = Field(NodeType.ACTIVE_LEARNING_LOOP)


class EonTransitionData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal[NodeType.EON_TRANSITION_SEARCH] = Field(NodeType.EON_TRANSITION_SEARCH)


AnyNodeData = Annotated[
    InitialStructureData | MaceTrainingData | ActiveLearningData | EonTransitionData,
    Field(discriminator="type"),
]


class DagNode(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)
    id: str = Field(..., description="Node UUID")
    type: NodeType = Field(..., description="Type of the node")
    data: AnyNodeData = Field(..., description="Node data")


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)
    accuracy_speed_slider: int = Field(..., ge=1, le=10, description="Accuracy vs speed tradeoff")
    target_material: str = Field(..., description="Target material for the intent")
    nodes: list[DagNode] = Field(..., description="List of nodes in the DAG")
    edges: list[Edge] = Field(..., description="List of edges in the DAG")

    @field_validator("target_material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        if v not in chemical_symbols:
            msg = f"Target material {v} is not a valid chemical symbol"
            raise ValueError(msg)
        return v

    def _validate_nodes_exist(self) -> None:
        nodes = {node.id: node for node in self.nodes}
        for edge in self.edges:
            if edge.source not in nodes:
                msg = f"Edge source {edge.source} not found in nodes"
                raise ValueError(msg)
            if edge.target not in nodes:
                msg = f"Edge target {edge.target} not found in nodes"
                raise ValueError(msg)

    def _check_cycles(self) -> None:
        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node.id)
        for edge in self.edges:
            graph.add_edge(edge.source, edge.target)

        if not nx.is_directed_acyclic_graph(graph):
            msg = "Graph contains a cycle and is not a valid DAG"
            raise ValueError(msg)

    @model_validator(mode="after")
    def validate_dag(self) -> "IntentRequest":
        self._validate_nodes_exist()
        self._check_cycles()
        return self


class ScenarioConfig(BaseModel):
    """Configuration for specific scenarios."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(..., description="Name of the scenario to run")
    parameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict, description="Scenario-specific parameters"
    )
    enabled: bool = Field(default=False, description="Whether to run this scenario")
