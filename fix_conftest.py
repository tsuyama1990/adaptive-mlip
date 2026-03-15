import re

with open('tests/conftest.py', 'r') as f:
    data = f.read()

# Update mock_training_config
old_train = 'TrainingConfig(potential_type="mace", cutoff_radius=5.0, max_basis_size=8)'
new_train = 'TrainingConfig(potential_type="mace", cutoff_radius=5.0, max_basis_size=8, elements=["Al"], max_iterations=1000, batch_size=8, delta_learning=False, active_set_optimization=False, active_set_size=None)'
data = data.replace(old_train, new_train)

# Update mock_md_config
old_md = 'MDConfig(temperature=300.0, pressure=0.0, timestep=1.0, n_steps=100)'
new_md = 'MDConfig(temperature=300.0, pressure=0.0, timestep=1.0, n_steps=100, units="metal", atom_style="atomic", thermo_freq=100, dump_freq=100, minimize=False, neighbor_skin=2.0, tdamp_factor=100.0, pdamp_factor=1000.0, hybrid_potential=False, fix_halt=False, uncertainty_threshold=5.0, check_interval=10, ramping=None, mc=None, soft_start_steps=0, soft_start_langevin_damp=0.1)'
data = data.replace(old_md, new_md)

with open('tests/conftest.py', 'w') as f:
    f.write(data)
