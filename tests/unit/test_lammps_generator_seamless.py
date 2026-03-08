import io
from pathlib import Path

from pyacemaker.core.lammps_generator import LammpsScriptGenerator
from pyacemaker.domain_models.md import MDConfig


def test_script_generator_resume() -> None:
    config = MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=1000)
    gen = LammpsScriptGenerator(config)

    buf = io.StringIO()
    gen.write_script(buf, Path("pot.yace"), Path("data"), Path("dump"), ["H"], resume_from_step=500)
    script = buf.getvalue()

    # Check that soft start langevin is generated instead of standard velocity creation
    assert "velocity all create" not in script
    assert "fix langevin all langevin" in script
    assert "run 100" in script # Soft start steps
    assert "run 500" in script # Remaining steps (1000 - 500)
    assert "fix python_invoke" in script

def test_script_generator_no_resume() -> None:
    config = MDConfig(temperature=300.0, pressure=0.0, timestep=0.001, n_steps=1000)
    gen = LammpsScriptGenerator(config)

    buf = io.StringIO()
    gen.write_script(buf, Path("pot.yace"), Path("data"), Path("dump"), ["H"], resume_from_step=None)
    script = buf.getvalue()

    # Check standard start
    assert "velocity all create" in script
    assert "fix npt all npt temp" in script
    assert "run 1000" in script
    assert "fix python_invoke" in script
