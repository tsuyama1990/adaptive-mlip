from pathlib import Path
from ase import Atoms
from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.md import MDConfig

mock_md_config = MDConfig(
    temperature=300.0, pressure=1.0, timestep=0.001, n_steps=1000,
    thermo_freq=10, dump_freq=100, minimize=False,
    hybrid_potential=False, fix_halt=False, uncertainty_threshold=5.0,
    check_interval=10, soft_start_steps=0
)
engine = LammpsEngine(mock_md_config)
atoms = Atoms("He", positions=[[0.1, 0.1, 0.1]], cell=[10, 10, 10], pbc=True)
p = Path("/tmp/potential.yace")
p.touch()

ctx, data_file, dump_file, log_file, elements, potential_path = engine._prepare_simulation_env(atoms, p)
import io
buffer = io.StringIO()
engine.generator.write_minimization_script(buffer, potential_path, data_file, elements)
print(buffer.getvalue())
