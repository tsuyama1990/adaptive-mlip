from io import StringIO
from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import MDConfig


def test_lammps_generator_order() -> None:
    from pyacemaker.domain_models.workflow import WorkflowConfig

    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path="state.json",
        data_dir="data",
        active_learning_dir="al",
        potentials_dir="pots",
        otf={"fix_halt": True},
    )
    generator = LammpsScriptGenerator(config, workflow_config)

    # Use StringIO as buffer
    buffer = StringIO()
    generator.write_script(
        buffer,
        potential_path=Path("pot.yace"),
        data_file=Path("data.lmp"),
        dump_file=Path("dump.lammps"),
        elements=["Fe"],
    )

    script = buffer.getvalue()

    lines = script.splitlines()
    run_idx = -1
    dump_idx = -1
    thermo_idx = -1

    for i, line in enumerate(lines):
        if line.startswith("run"):
            run_idx = i
        if line.startswith("dump"):
            dump_idx = i
        if line.startswith("thermo"):
            thermo_idx = i

    assert run_idx != -1, "run command not found"
    assert dump_idx != -1, "dump command not found"
    assert thermo_idx != -1, "thermo command not found"

    # dump and thermo must be BEFORE run
    assert dump_idx < run_idx, f"dump command is after run: dump={dump_idx}, run={run_idx}"
    assert thermo_idx < run_idx, f"thermo command is after run: thermo={thermo_idx}, run={run_idx}"


def test_lammps_generator_gamma_column() -> None:
    from pyacemaker.domain_models.workflow import WorkflowConfig

    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path="state.json",
        data_dir="data",
        active_learning_dir="al",
        potentials_dir="pots",
        otf={"fix_halt": True},
    )
    generator = LammpsScriptGenerator(config, workflow_config)

    buffer = StringIO()
    generator.write_script(
        buffer,
        potential_path=Path("pot.yace"),
        data_file=Path("data.lmp"),
        dump_file=Path("dump.lammps"),
        elements=["Fe"],
    )
    script = buffer.getvalue()

    dump_line = next(line for line in script.splitlines() if line.startswith("dump"))
    assert "c_gamma" in dump_line, "c_gamma not found in dump command"
