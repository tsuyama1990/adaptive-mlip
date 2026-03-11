# Patch test_lammps_generator.py
with open("tests/unit/test_lammps_generator.py") as f:
    content = f.read()

content = content.replace(
    'assert "fix halt_check all halt 10 v_max_g > 5.0 error continue" in script',
    'assert "fix python_invoke all python/invoke 10 post_force invoke_evaluator" in script',
)

with open("tests/unit/test_lammps_generator.py", "w") as f:
    f.write(content)

# Patch test_engine.py
with open("tests/unit/test_engine.py") as f:
    content = f.read()

content = content.replace('assert "fix halt" in script', 'assert "fix python_invoke" in script')

with open("tests/unit/test_engine.py", "w") as f:
    f.write(content)
