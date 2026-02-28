
with open('src/pyacemaker/modules/mock_oracle.py') as f:
    content = f.read()

content = content.replace(
    'self.calc = LennardJones(sigma=sigma, epsilon=epsilon)',
    'self.calc = LennardJones(sigma=sigma, epsilon=epsilon) # type: ignore[no-untyped-call]'
)
content = content.replace(
    'energy = float(atoms.get_potential_energy())  # type: ignore[no-untyped-call]',
    'energy = float(atoms.get_potential_energy())'
)
content = content.replace(
    'forces = atoms.get_forces()  # type: ignore[no-untyped-call]',
    'forces = atoms.get_forces()'
)
content = content.replace(
    'stress = atoms.get_stress()  # type: ignore[no-untyped-call]',
    'stress = atoms.get_stress()'
)

with open('src/pyacemaker/modules/mock_oracle.py', 'w') as f:
    f.write(content)
