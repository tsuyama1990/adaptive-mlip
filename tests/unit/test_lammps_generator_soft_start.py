from io import StringIO
from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import MDConfig


def test_generator_soft_start_and_restart(tmp_path: Path) -> None:
    """Tests script generation for soft start (resume) and restart parameters."""
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        soft_start_steps=15,
        fix_halt=True
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"
    restart_file = tmp_path / "lammps.restart"
    read_restart = tmp_path / "lammps.read.restart"
    eval_dir = tmp_path / "eval_dir"
    eval_dir.mkdir()
    # Path utils expect parents to exist, so let's touch the file
    evaluator_script_path = eval_dir / "evaluator_script.py"
    evaluator_script_path.touch()

    buffer = StringIO()
    generator.write_script(
        buffer,
        pot_path,
        data_file,
        dump_file,
        ["Al"],
        use_fix_invoke=True,
        thresholds={"threshold_call_dft": 0.5, "threshold_add_train": 0.2, "smooth_steps": 3},
        resume_from_step=100,
        restart_file=restart_file,
        read_restart=read_restart,
        eval_dir=eval_dir
    )
    script = buffer.getvalue()

    # Read restart
    assert f"read_restart {read_restart!s}" in script

    # Write restart (only once at end if using read_restart)
    # The normal write_restart should be there, but the periodic restart might be skipped or present depending on logic.
    assert f"write_restart {restart_file!s}" in script

    # Soft Start Langevin Thermostat
    assert "fix soft_start all langevin" in script
    assert "run 15" in script
    assert "unfix soft_start" in script
    assert "unfix nve" in script

    # Calculate remaining steps: 1000 - 100 (resume) - 15 (soft_start) = 885
    assert "run 885" in script

    # Check evaluator_script generation
    evaluator_script_path = eval_dir / "evaluator_script.py"
    assert evaluator_script_path.exists()
    eval_script = evaluator_script_path.read_text()
    assert "evaluator = TwoTierEvaluator(0.5, 0.2, 3)" in eval_script
    assert f"python invoke_evaluator invoke lammps_invoke_evaluator file {evaluator_script_path!s}" in script
