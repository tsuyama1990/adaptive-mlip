from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NodeType(StrEnum):
    INITIAL_STRUCTURE = "INITIAL_STRUCTURE"
    MACE_TRAINING = "MACE_TRAINING"
    ACTIVE_LEARNING_LOOP = "ACTIVE_LEARNING_LOOP"
    EON_TRANSITION_SEARCH = "EON_TRANSITION_SEARCH"


class InitialStructureData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    chemical_symbol: str = Field(..., description="The chemical symbol")
    lattice_constant: float = Field(..., description="The lattice constant")


class ActiveLearningData(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DagNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., description="Node UUID")
    type: NodeType = Field(..., description="Type of the node")
    data: dict[str, Any] = Field(default_factory=dict, description="Node data")


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(..., description="Source node id")
    target: str = Field(..., description="Target node id")


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    accuracy_speed_slider: int = Field(..., ge=1, le=10, description="Accuracy vs speed tradeoff")
    target_material: str = Field(..., description="Target material for the intent")
    nodes: list[DagNode] = Field(..., description="List of nodes in the DAG")
    edges: list[Edge] = Field(..., description="List of edges in the DAG")

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
        nodes = {node.id: node for node in self.nodes}
        adj: dict[str, list[str]] = {node.id: [] for node in self.nodes}
        for edge in self.edges:
            adj[edge.source].append(edge.target)

        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            if node_id in stack:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            stack.add(node_id)

            for neighbor in adj.get(node_id, []):
                if dfs(neighbor):
                    return True

            stack.remove(node_id)
            return False

        for node_id in nodes:
            if dfs(node_id):
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
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Scenario-specific parameters"
    )
    enabled: bool = Field(False, description="Whether to run this scenario")
