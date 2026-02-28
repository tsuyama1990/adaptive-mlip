
with open('src/pyacemaker/core/lammps_generator.py') as f:
    content = f.read()

content = content.replace(
    'self._atomic_numbers_cache = {}',
    'self._atomic_numbers_cache: dict[str, int] = {}'
)

with open('src/pyacemaker/core/lammps_generator.py', 'w') as f:
    f.write(content)
