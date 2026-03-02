import tempfile
from pathlib import Path

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig


def test_lammps_engine_resume():
    config = MDConfig(temperature=300, pressure=0, timestep=0.001, n_steps=100)
    engine = LammpsEngine(config=config)

    # Check initial state
    assert engine._restart_file_path is None

    # Just test that the _write_resume_script generates the soft start properly
    with tempfile.TemporaryDirectory() as td:
        dummy_file = Path(td) / "dummy.lmp"
        with dummy_file.open("w") as f:
            engine._write_resume_script(
                f, Path("dummy.yace"), Path("in.restart"), Path("out.restart"), Path("dump.xyz")
            )

        with dummy_file.open("r") as f:
            content = f.read()
            assert "read_restart in.restart" in content
            assert "fix soft_start all langevin 300.0 300.0 10.0" in content
            assert "unfix soft_start" in content
