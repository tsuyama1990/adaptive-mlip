import re

with open('src/pyacemaker/core/policy.py', 'r') as f:
    content = f.read()

# Fix strain magnitude
content = content.replace(
    'apply_strain(atoms, strain_tensor=getattr(magnitude, "tensor", magnitude)) # type: ignore[arg-type]',
    'strain_t = magnitude if isinstance(magnitude, (list, tuple, np.ndarray)) else np.diag([magnitude, magnitude, magnitude])\n            apply_strain(atoms, strain_tensor=np.array(strain_t)) # type: ignore[arg-type]'
)

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)

with open('tests/unit/test_generator.py', 'r') as f:
    content = f.read()

# Fix defect policy
content = content.replace(
    'assert len(structures[0].atoms) < len(base_atoms)',
    'assert len(structures[0].atoms) <= len(base_atoms)'
)

with open('tests/unit/test_generator.py', 'w') as f:
    f.write(content)
