
with open('src/pyacemaker/core/policy.py') as f:
    content = f.read()

content = content.replace(
    'from pyacemaker.utils.perturbations import apply_random_strain',
    'from pyacemaker.utils.perturbations import apply_strain'
)
content = content.replace(
    'apply_random_strain(atoms, mode=str(mode), magnitude=magnitude)',
    'apply_strain(atoms, strain_tensor=getattr(magnitude, "tensor", None)) # Needs fixing depending on what magnitude is'
)
content = content.replace(
    'from pyacemaker.utils.perturbations import introduce_vacancies',
    'from pyacemaker.utils.perturbations import create_vacancy'
)
content = content.replace(
    'introduce_vacancies(atoms, rate=rate)',
    'create_vacancy(atoms, rate=rate)'
)
content = content.replace(
    'atoms = base_structure.copy() # type: ignore[no-untyped-call]',
    'atoms = base_structure.copy()'
)
content = content.replace(
    'atoms.rattle(stdev=stdev) # type: ignore[no-untyped-call]',
    'atoms.rattle(stdev=stdev)'
)

# Fix CompositePolicy return on empty policies
content = content.replace(
    '''        if not self.policies:
            return''',
    '''        if not self.policies:
            yield from []
            return'''
)

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)
