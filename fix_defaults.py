
with open('src/pyacemaker/domain_models/defaults.py') as f:
    content = f.read()

additions = """
DEFAULT_MC_SEED = 12345
DEFAULT_EON_SEED = 12345
LAMMPS_VELOCITY_SEED = 12345
"""

if 'DEFAULT_MC_SEED' not in content:
    with open('src/pyacemaker/domain_models/defaults.py', 'a') as f:
        f.write(additions)
