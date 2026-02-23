from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models import (
    DFTConfig,
    MDConfig,
    PyAceConfig,
    StructureConfig,
    TrainingConfig,
    WorkflowConfig,
)
from pyacemaker.factory import ModuleFactory


def test_factory_creates_structure_generator() -> None:
    config = PyAceConfig(
        project_name="TestProject",
        structure=StructureConfig(elements=["Fe"], supercell_size=[1, 1, 1]),
        dft=DFTConfig(
            code="qe",
            functional="pbe",
            kpoints_density=0.1,
            encut=500.0,
            pseudopotentials={"Fe": "Fe.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ),
        training=TrainingConfig(potential_type="linear", cutoff_radius=5.0, max_basis_size=100),
        md=MDConfig(temperature=300, pressure=1.0, timestep=0.001, n_steps=1000),
        workflow=WorkflowConfig(max_iterations=10),
    )

    generator, oracle, trainer, engine = ModuleFactory.create_modules(config)

    # Verify Generator is correctly instantiated
    assert isinstance(generator, StructureGenerator)
    # Verify configuration injection
    assert generator.config == config.structure
