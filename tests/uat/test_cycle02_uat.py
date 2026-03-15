from pathlib import Path
from typing import Any

import pytest
from ase import Atoms

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
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
from tests.conftest import MockCalculator
from tests.constants import TEST_ENERGY_H2O


@pytest.fixture
def uat_dft_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DFTConfig:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "H.UPF").touch()
    (tmp_path / "O.UPF").touch()

    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.7,
        smearing_type="mv",
        smearing_width=0.1,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF", "O": "O.UPF"},
    )


class DummyFuture:
    def __init__(self, result_value: Any, exception: Any = None) -> None:
        self._result_value = result_value
        self._exception = exception

    def result(self, timeout: float | None = None) -> Any:
        return self._result_value, self._exception


class DummyExecutor:
    def __init__(self, max_workers: int) -> None:
        pass

    def __enter__(self) -> "DummyExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> DummyFuture:
        try:
            res, exc = fn(*args, **kwargs)
            return DummyFuture(res, exc)
        except Exception as e:
            return DummyFuture(None, e)


def test_uat_02_01_single_point_calculation(
    uat_dft_config: DFTConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Scenario 02-01: Single Point Calculation.
    Verify that the system can run a simple DFT calculation (mocked).
    """
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummyExecutor)
    # 1. Preparation: H2O molecule
    h2o = Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True
    )

    # 2. Action: Run DFTManager with mocked driver via dependency injection
    from unittest.mock import MagicMock

    mock_driver_instance = MagicMock()
    mock_driver_instance.get_calculator.side_effect = lambda atoms, config, **kwargs: (
        MockCalculator(fail_count=0, test_energy=TEST_ENERGY_H2O)
    )

    manager = DFTManager(uat_dft_config, driver=mock_driver_instance)

    # Use explicit iteration
    gen = manager.compute(iter([h2o]))
    result = next(gen)

    # 3. Expectation
    assert result.get_potential_energy() == TEST_ENERGY_H2O  # type: ignore[no-untyped-call]
    assert result.get_forces().shape == (3, 3)  # type: ignore[no-untyped-call]


def test_uat_02_02_self_healing(
    uat_dft_config: DFTConfig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Scenario 02-02: Self-Healing Test.
    Verify that the system recovers from a simulated SCF convergence failure.
    """
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", DummyExecutor)
    # 1. Preparation
    h2o = Atoms(
        "H2O", positions=[[0, 0, 0], [0, 0, 0.96], [0, 0.96, 0]], cell=[10, 10, 10], pbc=True
    )

    # 2. Action: Run DFTManager with failure via dependency injection
    from unittest.mock import MagicMock

    mock_driver_instance = MagicMock()

    # Mock failure on first attempt, success on second
    calc_fail = MockCalculator(fail_count=1, test_energy=TEST_ENERGY_H2O)
    calc_success = MockCalculator(fail_count=0, test_energy=TEST_ENERGY_H2O)

    mock_driver_instance.get_calculator.side_effect = [calc_fail, calc_success]

    manager = DFTManager(uat_dft_config, driver=mock_driver_instance)

    gen = manager.compute(iter([h2o]))
    result = next(gen)

    # 3. Expectation
    assert result.get_potential_energy() == TEST_ENERGY_H2O  # type: ignore[no-untyped-call]

    # Verify that get_calculator was called twice (original + retry)
    assert mock_driver_instance.get_calculator.call_count == 2

    # Verify second call had reduced mixing_beta
    # First call: original (0.7)
    # Second call: reduced (0.35)
    args, _ = mock_driver_instance.get_calculator.call_args  # Last call
    final_config = args[1]
    assert final_config.mixing_beta < 0.7
    assert final_config.mixing_beta == 0.35


def test_uat_02_a_successful_translation() -> None:
    """
    SCENARIO-02-A: Successful DAG to WorkflowConfig Translation
    """
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

    intent = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Al",
        nodes=[node1, node2, node3],
        edges=[Edge(source="n1", target="n2"), Edge(source="n2", target="n3")],
    )

    config = SemanticCompiler.compile(intent)
    assert config.structure.elements == ["Al"]
    assert config.training.potential_type == "mace"
    assert config.workflow.loop_strategy.use_tiered_oracle is True


def test_uat_02_b_intelligent_defaults() -> None:
    """
    SCENARIO-02-B: Intelligent Default Parameter Injection
    """
    node1 = DagNode(
        id="n1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE, chemical_symbol="W", lattice_constant=3.16
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

    intent = IntentRequest(
        accuracy_speed_slider=1,
        target_material="W",
        nodes=[node1, node2, node3],
        edges=[Edge(source="n1", target="n3"), Edge(source="n3", target="n2")],
    )

    config = SemanticCompiler.compile(intent)

    assert config.md.timestep == 2.0  # W is heavy > 50.0
    assert config.dft.encut == 42.0  # 40.0 + (1 * 2.0)
    assert (
        config.workflow.loop_strategy.thresholds.threshold_call_dft > 0.1
    )  # slider=1 triggers high threshold


def test_uat_02_c_logical_rejection() -> None:
    """
    SCENARIO-02-C: Logical Workflow Validation Rejection
    """
    from pyacemaker.core.exceptions import CompilerError

    # 1. Invalid sequence (Active learning before Structure)
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
    node3 = DagNode(
        id="n3", type=NodeType.MACE_TRAINING, data=MaceTrainingData(type=NodeType.MACE_TRAINING)
    )

    intent1 = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Al",
        nodes=[node1, node2, node3],
        edges=[Edge(source="n1", target="n3"), Edge(source="n3", target="n2")],
    )

    with pytest.raises(CompilerError, match="INITIAL_STRUCTURE node must precede"):
        SemanticCompiler.compile(intent1)

    # 2. Branching error (Parallel active learning loops)
    node1b = DagNode(
        id="n1",
        type=NodeType.INITIAL_STRUCTURE,
        data=InitialStructureData(
            type=NodeType.INITIAL_STRUCTURE, chemical_symbol="Al", lattice_constant=4.0
        ),
    )
    node2b = DagNode(
        id="n2",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )
    node3b = DagNode(
        id="n3",
        type=NodeType.ACTIVE_LEARNING_LOOP,
        data=ActiveLearningData(type=NodeType.ACTIVE_LEARNING_LOOP),
    )
    node4b = DagNode(
        id="n4", type=NodeType.MACE_TRAINING, data=MaceTrainingData(type=NodeType.MACE_TRAINING)
    )

    intent2 = IntentRequest(
        accuracy_speed_slider=5,
        target_material="Al",
        nodes=[node1b, node4b, node2b, node3b],
        edges=[
            Edge(source="n1", target="n4"),
            Edge(source="n4", target="n2"),
            Edge(source="n4", target="n3"),
        ],
    )

    with pytest.raises(CompilerError, match="Parallel active learning loops"):
        SemanticCompiler.compile(intent2)
