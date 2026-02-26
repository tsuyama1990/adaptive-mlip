from io import StringIO
from pathlib import Path
from unittest.mock import patch

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import (
    HybridParams,
    MCConfig,
    MDConfig,
    MDRampingConfig,
)


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
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"

    buffer = StringIO()
    generator.write_script(buffer, pot_path, data_file, dump_file, ["H", "He"])
    script = buffer.getvalue()

    assert "pair_style hybrid/overlay" in script
    assert f'pair_coeff * * pace "{pot_path}" H He' in script

    # ZBL checks
    assert "pair_coeff 1 1 zbl 1 1" in script
    assert "pair_coeff 1 2 zbl 1 2" in script
    assert "pair_coeff 2 2 zbl 2 2" in script
    assert "1.0 1.5" in script # cutoffs


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
    data_file = tmp_path / "data.lmp"
    dump_file = tmp_path / "dump.lammpstrj"

    buffer = StringIO()
    generator.write_script(buffer, pot_path, data_file, dump_file, ["Al"])
    script = buffer.getvalue()

    assert "pair_style pace" in script
    assert "pair_style hybrid" not in script
    assert f'pair_coeff * * pace "{pot_path}" Al' in script


def test_generator_damping(tmp_path: Path) -> None:
    """Tests damping parameter calculation."""
    config = MDConfig(
        temperature=300.0,
        pressure=1.0,
        timestep=0.002,
        n_steps=1000,
        tdamp_factor=50.0,
        pdamp_factor=500.0
    )
    generator = LammpsScriptGenerator(config)

    buffer = StringIO()
    # Mock validate_path_safe to avoid error on fake paths
    with patch("pyacemaker.core.lammps_generator.validate_path_safe", side_effect=lambda x: x):
        generator.write_script(buffer, Path("pot"), Path("dat"), Path("dump"), ["H"])
    script = buffer.getvalue()

    # tdamp = 50 * 0.002 = 0.1
    # pdamp = 500 * 0.002 = 1.0
    assert "temp 300.0 300.0 0.1" in script
    assert "iso 1.0 1.0 1.0" in script


def test_generator_atom_style() -> None:
    """Tests atom_style configuration."""
    config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=100,
        atom_style="charge"
    )
    generator = LammpsScriptGenerator(config)
    buffer = StringIO()
    # Mock validate_path_safe to avoid error on fake paths
    with patch("pyacemaker.core.lammps_generator.validate_path_safe", side_effect=lambda x: x):
        generator.write_script(buffer, Path("pot"), Path("dat"), Path("dump"), ["H"])
    script = buffer.getvalue()

    # Check for correct atom_style command
    # Note: If MDConfig's atom_style is an Enum, f-string uses value if StrEnum, or Member repr if Enum.
    # We will fix implementation to use .value, so expect "atom_style charge".
    assert "atom_style charge" in script


def test_generator_minimization() -> None:
    """Tests minimization script generation."""
    config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=100
    )
    generator = LammpsScriptGenerator(config)
    buffer = StringIO()

    with patch("pyacemaker.core.lammps_generator.validate_path_safe", side_effect=lambda x: x):
        generator.write_minimization_script(buffer, Path("pot"), Path("dat"), ["H"])
    script = buffer.getvalue()

    assert "min_style cg" in script
    # Defaults in MDConfig are now 1.0e-4 1.0e-6 100 1000
    # Note: floating point format might vary slightly depending on default representation
    # "0.0001" or "1e-04".
    # MDConfig defaults are floats.
    assert "minimize 0.0001 1e-06 10000 10000" in script or "minimize 1e-04 1e-06 10000 10000" in script


def test_generator_mc_commands() -> None:
    """Tests generation of Monte Carlo swap commands."""
    mc = MCConfig(swap_freq=50, swap_prob=0.5, seed=999)
    config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=100, mc=mc
    )
    generator = LammpsScriptGenerator(config)
    buffer = StringIO()

    with patch("pyacemaker.core.lammps_generator.validate_path_safe", side_effect=lambda x: x):
        generator.write_script(buffer, Path("pot"), Path("dat"), Path("dump"), ["H", "He"])
    script = buffer.getvalue()

    # Expected: fix swap all atom/swap 50 1 999 300.0 ke no types 1 2
    # But wait, T needs to be defined. The swap command syntax:
    # fix ID group-ID atom/swap N X seed T [options]
    # N = swap every N steps
    # X = number of swaps per attempt (1 usually)
    # seed = random seed
    # T = temperature (or variable)
    assert "fix mc_swap all atom/swap 50 1 999 300.0 ke no types 1 2" in script or "fix mc_swap all atom/swap 50 1 999 300.0" in script


def test_generator_ramping_commands() -> None:
    """Tests generation of ramping commands."""
    ramping = MDRampingConfig(temp_start=100.0, temp_end=500.0, press_start=1.0, press_end=10.0)
    config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=100, ramping=ramping
    )
    generator = LammpsScriptGenerator(config)
    buffer = StringIO()

    with patch("pyacemaker.core.lammps_generator.validate_path_safe", side_effect=lambda x: x):
        generator.write_script(buffer, Path("pot"), Path("dat"), Path("dump"), ["H"])
    script = buffer.getvalue()

    # fix npt temp Tstart Tend Tdamp iso Pstart Pend Pdamp
    # Tdamp = 100 * 0.001 = 0.1
    # Pdamp = 1000 * 0.001 = 1.0
    assert "temp 100.0 500.0 0.1" in script
    assert "iso 1.0 10.0 1.0" in script
