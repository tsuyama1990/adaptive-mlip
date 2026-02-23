from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import HybridParams, MDConfig


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

    script = generator.generate(pot_path, data_file, dump_file, ["H", "He"])

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

    script = generator.generate(pot_path, data_file, dump_file, ["Al"])

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

    script = generator.generate(Path("pot"), Path("dat"), Path("dump"), ["H"])

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
    script = generator.generate(Path("pot"), Path("dat"), Path("dump"), ["H"])
    assert "atom_style charge" in script
