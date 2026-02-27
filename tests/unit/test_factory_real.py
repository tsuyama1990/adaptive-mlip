from pyacemaker.core.active_set import ActiveSetSelector
from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.generator import StructureGenerator
from pyacemaker.core.oracle import DFTManager
from pyacemaker.core.trainer import PacemakerTrainer
from pyacemaker.core.validator import Validator
from pyacemaker.domain_models import PyAceConfig
from pyacemaker.factory import ModuleFactory


def test_create_modules_standard(
    mock_dft_config, mock_structure_config, mock_training_config, mock_md_config
) -> None:
    """
    Verify that ModuleFactory creates real instances (not mocks) given a valid config.
    This integration test ensures that the factory logic matches the interface expectations.
    """
    config = PyAceConfig(
        project_name="test_proj",
        structure=mock_structure_config,
        dft=mock_dft_config,
        training=mock_training_config,
        md=mock_md_config,
        workflow={
            "max_iterations": 1,
            "state_file_path": "state.json",
            "active_learning_dir": "active_learning",
            "potentials_dir": "potentials",
            "data_dir": "data",
            "n_candidates": 10
        }
    )

    # Since lammps is mocked in conftest, LammpsEngine should initialize without binary error.
    modules = ModuleFactory.create_modules(config)

    assert len(modules) == 6
    gen, oracle, trainer, engine, selector, validator = modules

    assert isinstance(gen, StructureGenerator)
    assert isinstance(oracle, DFTManager)
    assert isinstance(trainer, PacemakerTrainer)
    assert isinstance(engine, LammpsEngine)
    assert isinstance(selector, ActiveSetSelector)
    assert isinstance(validator, Validator)

def test_create_modules_distillation(
    mock_dft_config, mock_structure_config, mock_training_config, mock_md_config
) -> None:
    """Verify factory behavior when distillation is enabled (currently returns standard modules)."""
    config = PyAceConfig(
        project_name="test_proj",
        structure=mock_structure_config,
        dft=mock_dft_config,
        training=mock_training_config,
        md=mock_md_config,
        workflow={
            "max_iterations": 1,
            "state_file_path": "state.json",
            "active_learning_dir": "active_learning",
            "potentials_dir": "potentials",
            "data_dir": "data",
            "n_candidates": 10
        },
        distillation={"enable_mace_distillation": True}
    )

    modules = ModuleFactory.create_modules(config)
    assert len(modules) == 6
    gen, *_ = modules
    assert isinstance(gen, StructureGenerator)
