import shlex
from io import StringIO
from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import HybridParams, MDConfig


def test_generator_pure_pace(tmp_path: Path) -> None:
    """Tests script generation with pure PACE."""
    from pyacemaker.domain_models.workflow import WorkflowConfig

    config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=1000, hybrid_potential=False
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path=str(tmp_path / "state.json"),
        data_dir=str(tmp_path / "data"),
        active_learning_dir=str(tmp_path / "al"),
        potentials_dir=str(tmp_path / "pots"),
    )
    generator = LammpsScriptGenerator(config, workflow_config)

    pot_path = tmp_path / "potential.yace"
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"

    buffer = StringIO()
    generator.write_script(buffer, pot_path, data_file, dump_file, ["Al"])
    script = buffer.getvalue()

    assert "pair_style pace" in script
    assert "pair_style hybrid" not in script

    # Expected quoted path (shlex.quote might use single quotes)
    expected_pot = shlex.quote(str(pot_path))
    assert f"pair_coeff * * pace {expected_pot} Al" in script


def test_generator_hybrid_potential(tmp_path: Path) -> None:
    """Tests script generation with hybrid potential."""
    from pyacemaker.domain_models.workflow import WorkflowConfig

    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        hybrid_params=HybridParams(zbl_cut_inner=1.0, zbl_cut_outer=1.5),
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path=str(tmp_path / "state.json"),
        data_dir=str(tmp_path / "data"),
        active_learning_dir=str(tmp_path / "al"),
        potentials_dir=str(tmp_path / "pots"),
    )
    generator = LammpsScriptGenerator(config, workflow_config)

    pot_path = tmp_path / "potential.yace"
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"

    buffer = StringIO()
    generator.write_script(buffer, pot_path, data_file, dump_file, ["H", "He"])
    script = buffer.getvalue()

    assert "pair_style hybrid/overlay" in script

    expected_pot = shlex.quote(str(pot_path))
    assert f"pair_coeff * * pace {expected_pot} H He" in script

    # ZBL check
    assert "pair_coeff 1 1 zbl 1 1" in script
    assert "pair_coeff 1 2 zbl 1 2" in script
    assert "pair_coeff 2 2 zbl 2 2" in script


def test_generator_watchdog(tmp_path: Path) -> None:
    """Tests generation of watchdog commands."""
    from pyacemaker.domain_models.workflow import WorkflowConfig

    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
    )
    workflow_config = WorkflowConfig(
        max_iterations=1,
        state_file_path=str(tmp_path / "state.json"),
        data_dir=str(tmp_path / "data"),
        active_learning_dir=str(tmp_path / "al"),
        potentials_dir=str(tmp_path / "pots"),
        loop_strategy={
            "thresholds": {"threshold_call_dft": 5.0, "threshold_add_train": 2.0, "smooth_steps": 3}
        },
        otf={"fix_halt": True, "check_interval": 10},
    )
    generator = LammpsScriptGenerator(config, workflow_config)

    pot_path = tmp_path / "potential.yace"
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"

    buffer = StringIO()
    generator.write_script(buffer, pot_path, data_file, dump_file, ["Al"])
    script = buffer.getvalue()

    expected_pot = shlex.quote(str(pot_path))
    assert f"compute gamma all pace {expected_pot}" in script
    assert "compute max_gamma all reduce max c_gamma" in script
    assert "fix halt_check all halt 10 v_max_g > 5.0 error continue" in script
