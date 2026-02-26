import shlex
from io import StringIO
from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import HybridParams, MDConfig


def test_generator_pure_pace(tmp_path: Path) -> None:
    """Tests script generation with pure PACE."""
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=False
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch() # Create dummy file
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
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        hybrid_potential=True,
        hybrid_params=HybridParams(zbl_cut_inner=1.0, zbl_cut_outer=1.5)
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch() # Create dummy file
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
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        n_steps=1000,
        fix_halt=True,
        uncertainty_threshold=5.0,
        check_interval=10
    )
    generator = LammpsScriptGenerator(config)

    pot_path = tmp_path / "potential.yace"
    pot_path.touch() # Create dummy file
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"

    buffer = StringIO()
    generator.write_script(buffer, pot_path, data_file, dump_file, ["Al"])
    script = buffer.getvalue()

    expected_pot = shlex.quote(str(pot_path))
    assert f"compute gamma all pace {expected_pot}" in script
    assert "compute max_gamma all reduce max c_gamma" in script
    assert "fix halt_check all halt 10 v_max_g > 5.0 error continue" in script
