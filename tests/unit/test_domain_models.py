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


def test_structure_config_valid() -> None:
    config = StructureConfig(elements=["Fe", "Pt"], supercell_size=[2, 2, 2])
    assert config.elements == ["Fe", "Pt"]
    assert config.supercell_size == [2, 2, 2]


def test_structure_config_invalid_element() -> None:
    with pytest.raises(ValueError, match="Invalid chemical symbol"):
        StructureConfig(elements=["Xy"], supercell_size=[1, 1, 1])  # Xy is not an element


def test_structure_config_invalid_supercell() -> None:
    with pytest.raises(ValidationError):
        StructureConfig(elements=["Fe"], supercell_size=[1, 1])  # Too short


def test_dft_config_valid() -> None:
    config = DFTConfig(code="quantum_espresso", functional="PBE", kpoints_density=0.04, encut=500.0)
    assert config.encut == 500.0


def test_dft_config_invalid_encut() -> None:
    with pytest.raises(ValidationError):
        DFTConfig(
            code="qe", functional="PBE", kpoints_density=0.04, encut=-100.0
        )  # Must be positive


def test_dft_config_missing_field() -> None:
    with pytest.raises(ValidationError):
        DFTConfig(code="qe", functional="PBE", kpoints_density=0.04)  # type: ignore[call-arg]
        # Missing encut


def test_training_config_valid() -> None:
    config = TrainingConfig(potential_type="ace", cutoff_radius=5.0, max_basis_size=500)
    assert config.cutoff_radius == 5.0


def test_training_config_invalid_cutoff() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(
            potential_type="ace", cutoff_radius=-2.0, max_basis_size=500
        )  # Must be positive


def test_md_config_valid() -> None:
    config = MDConfig(temperature=1000.0, pressure=0.0, timestep=0.001, n_steps=1000)
    assert config.temperature == 1000.0


def test_md_config_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        MDConfig(temperature=-100.0, pressure=0.0, timestep=0.001, n_steps=1000)  # Must be positive


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


def test_logging_config_valid() -> None:
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.log_file == "pyacemaker.log"


def test_logging_config_invalid_level() -> None:
    with pytest.raises(ValidationError):
        LoggingConfig(level="VERBOSE")  # Invalid level


def test_pyace_config_valid() -> None:
    structure = StructureConfig(elements=["Al"], supercell_size=[1, 1, 1])
    dft = DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0)
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
    assert config.logging.level == "INFO"


def test_pyace_config_missing_field() -> None:
    structure = StructureConfig(elements=["Al"], supercell_size=[1, 1, 1])
    dft = DFTConfig(code="qe", functional="PBE", kpoints_density=0.04, encut=400.0)
    training = TrainingConfig(potential_type="ace", cutoff_radius=4.5, max_basis_size=500)
    md = MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=1000)
    # Missing workflow

    with pytest.raises(ValidationError):
        PyAceConfig(  # type: ignore[call-arg]
            project_name="TestProject", structure=structure, dft=dft, training=training, md=md
        )
