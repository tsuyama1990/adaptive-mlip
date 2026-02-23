from pathlib import Path
import pytest
from pydantic import ValidationError

from pyacemaker.domain_models import (
    DFTConfig,
    LoggingConfig,
    MDConfig,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    WorkflowConfig,
)
from pyacemaker.domain_models.structure import ExplorationPolicy


def test_structure_config_valid() -> None:
    config = StructureConfig(elements=["Fe", "Pt"], supercell_size=[2, 2, 2])
    assert config.elements == ["Fe", "Pt"]
    assert config.supercell_size == [2, 2, 2]
    # Default policy
    assert config.policy_name == ExplorationPolicy.COLD_START


def test_structure_config_invalid_element() -> None:
    with pytest.raises(ValueError, match="Invalid chemical symbol"):
        StructureConfig(elements=["Xy"], supercell_size=[1, 1, 1])  # Xy is not an element


def test_structure_config_duplicates() -> None:
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        StructureConfig(elements=["Fe", "Fe"], supercell_size=[1, 1, 1])


def test_structure_config_invalid_supercell() -> None:
    with pytest.raises(ValidationError):
        StructureConfig(elements=["Fe"], supercell_size=[1, 1])  # Too short


def test_structure_config_policy() -> None:
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[1,1,1],
        policy_name="random_rattle",
        rattle_stdev=0.2
    )
    assert config.policy_name == ExplorationPolicy.RANDOM_RATTLE
    assert config.rattle_stdev == 0.2


def test_dft_config_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    config = DFTConfig(
        code="quantum_espresso",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"Fe": "Fe.UPF"},
    )
    assert config.encut == 500.0


def test_training_config_valid() -> None:
    config = TrainingConfig(potential_type="ace", cutoff_radius=5.0, max_basis_size=500)
    assert config.cutoff_radius == 5.0


def test_md_config_valid() -> None:
    config = MDConfig(temperature=1000.0, pressure=0.0, timestep=0.001, n_steps=1000)
    assert config.temperature == 1000.0


def test_workflow_config_valid() -> None:
    config = WorkflowConfig(
        max_iterations=10,
        state_file_path="custom_state.json",
        active_learning_dir="my_al_dir",
        potentials_dir="my_pots",
    )
    assert config.max_iterations == 10
    assert config.state_file_path == "custom_state.json"
    assert config.active_learning_dir == "my_al_dir"
    assert config.potentials_dir == "my_pots"


def test_workflow_config_default() -> None:
    config = WorkflowConfig(max_iterations=10)
    assert config.state_file_path == "state.json"
    assert config.active_learning_dir == "active_learning"
    assert config.potentials_dir == "potentials"
    assert config.checkpoint_interval == 1


def test_logging_config_valid() -> None:
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.log_file == "pyacemaker.log"


def test_pyace_config_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Al.UPF").touch()

    structure = StructureConfig(elements=["Al"], supercell_size=[1, 1, 1])
    dft = DFTConfig(
        code="qe",
        functional="PBE",
        kpoints_density=0.04,
        encut=400.0,
        pseudopotentials={"Al": "Al.UPF"},
    )
    training = TrainingConfig(potential_type="ace", cutoff_radius=4.5, max_basis_size=500)
    md = MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=1000)
    workflow = WorkflowConfig(max_iterations=10)
    logging = LoggingConfig()

    config = PyAceConfig(
        project_name="TestProject",
        structure=structure,
        dft=dft,
        training=training,
        md=md,
        workflow=workflow,
        logging=logging,
    )
    assert config.project_name == "TestProject"
    assert config.structure.elements == ["Al"]
