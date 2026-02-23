import pytest
from pydantic import ValidationError

from pyacemaker.domain_models import (
    DFTConfig,
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


def test_structure_config_invalid_supercell() -> None:
    with pytest.raises(ValidationError):
        StructureConfig(elements=["Fe"], supercell_size=[1, 1])  # Too short


def test_dft_config_valid() -> None:
    config = DFTConfig(encut=500.0)
    assert config.encut == 500.0
    assert config.kpoints_density == 0.04


def test_dft_config_invalid_encut() -> None:
    with pytest.raises(ValidationError):
        DFTConfig(encut=-100.0)  # Must be positive


def test_training_config_valid() -> None:
    config = TrainingConfig(cutoff_radius=5.0)
    assert config.cutoff_radius == 5.0


def test_training_config_invalid_cutoff() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(cutoff_radius=-2.0)  # Must be positive


def test_md_config_valid() -> None:
    config = MDConfig(temperature=1000.0)
    assert config.temperature == 1000.0


def test_md_config_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        MDConfig(temperature=-100.0)  # Must be positive


def test_workflow_config_valid() -> None:
    config = WorkflowConfig()
    assert config.max_iterations == 10


def test_pyace_config_valid() -> None:
    structure = StructureConfig(elements=["Al"], supercell_size=[1, 1, 1])
    dft = DFTConfig(encut=400.0)
    training = TrainingConfig(cutoff_radius=4.5)
    md = MDConfig(temperature=300.0)
    workflow = WorkflowConfig()

    config = PyAceConfig(
        project_name="TestProject",
        structure=structure,
        dft=dft,
        training=training,
        md=md,
        workflow=workflow,
    )
    assert config.project_name == "TestProject"
    assert config.structure.elements == ["Al"]


def test_pyace_config_missing_field() -> None:
    structure = StructureConfig(elements=["Al"], supercell_size=[1, 1, 1])
    dft = DFTConfig(encut=400.0)
    training = TrainingConfig(cutoff_radius=4.5)
    md = MDConfig(temperature=300.0)
    # Missing workflow

    with pytest.raises(ValidationError):
        PyAceConfig(  # type: ignore[call-arg]
            project_name="TestProject", structure=structure, dft=dft, training=training, md=md
        )
