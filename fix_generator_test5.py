import re

with open('src/pyacemaker/core/policy.py', 'r') as f:
    content = f.read()

# Make it actually modify the cell by random amount when magnitude is provided.
content = content.replace(
    'strain_t = magnitude if isinstance(magnitude, (list, tuple, np.ndarray)) else np.random.uniform(-magnitude, magnitude, (3, 3))',
    'strain_t = magnitude if isinstance(magnitude, (list, tuple, np.ndarray)) else np.array([[magnitude, 0, 0], [0, magnitude, 0], [0, 0, magnitude]])\n            # Randomize sign\n            strain_t = strain_t * np.random.choice([-1, 1])'
)

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)

with open('tests/unit/test_generator.py', 'r') as f:
    content = f.read()

# Fix strain policy test
content = content.replace(
    'assert vol0 != base_vol',
    'assert not np.allclose(vol0, base_vol)'
)

with open('tests/unit/test_generator.py', 'w') as f:
    f.write(content)
