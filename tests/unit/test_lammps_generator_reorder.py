from io import StringIO
from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import MDConfig


def test_lammps_generator_order(tmp_path: Path) -> None:
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True,
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    # Use StringIO as buffer
    buffer = StringIO()
    generator.write_script(
        buffer,
        potential_path=pot_path,
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


def test_lammps_generator_gamma_column(tmp_path: Path) -> None:
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True,
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    buffer = StringIO()
    generator.write_script(
        buffer,
        potential_path=pot_path,
        data_file=Path("data.lmp"),
        dump_file=Path("dump.lammps"),
        elements=["Fe"],
    )
    script = buffer.getvalue()

    dump_line = next(line for line in script.splitlines() if line.startswith("dump"))
    assert "c_gamma" in dump_line, "c_gamma not found in dump command"


def test_lammps_generator_resume_script(tmp_path: Path) -> None:
    config = MDConfig(
        temperature=300,
        pressure=0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True,
        soft_start_steps=50,
        soft_start_langevin_damp=0.2,
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch()

    buffer = StringIO()
    generator.write_script_resume(
        buffer,
        potential_path=pot_path,
        restart_file=Path("restart.lammps"),
        dump_file=Path("dump.lammps"),
        elements=["Fe"],
        resume_step=500,
        override_n_steps=200,
    )
    script = buffer.getvalue()

    assert "read_restart /app/restart.lammps" in script
    assert "python eval_wrapper invoke here" in script
    assert "fix soft_langevin all langevin 300.0 300.0 0.2" in script
    assert "run 50" in script  # the soft start run
    assert "unfix main_ensemble" in script
    assert "unfix soft_langevin" in script
    assert "run 150" in script  # override_n_steps (200) - soft_start_steps (50) = 150
