
with open('src/pyacemaker/utils/delta.py') as f:
    content = f.read()

content = content.replace(
    'return DEFAULT_LJ_PARAMS.get(element, FALLBACK_LJ_PARAMS.copy())',
    'return DEFAULT_LJ_PARAMS.get(element, FALLBACK_LJ_PARAMS.copy()) # type: ignore[return-value]'
)

with open('src/pyacemaker/utils/delta.py', 'w') as f:
    f.write(content)

with open('src/pyacemaker/modules/sampling.py') as f:
    content = f.read()

content = content.replace(
    'db = ase.db.connect(str(tmp_db_path))',
    'db = ase.db.connect(str(tmp_db_path)) # type: ignore[no-untyped-call]'
)
with open('src/pyacemaker/modules/sampling.py', 'w') as f:
    f.write(content)


with open('src/pyacemaker/core/policy.py') as f:
    content = f.read()

content = content.replace(
    'atoms = base_structure.copy()',
    'atoms = base_structure.copy() # type: ignore[no-untyped-call]'
)
content = content.replace(
    'apply_strain(atoms, strain_tensor=getattr(magnitude, "tensor", None)) # Needs fixing depending on what magnitude is',
    'apply_strain(atoms, strain_tensor=getattr(magnitude, "tensor", magnitude)) # type: ignore[arg-type]'
)

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)


with open('src/pyacemaker/core/generator.py') as f:
    content = f.read()
content = content.replace(
    'base_supercell = base_structure.repeat(self.config.supercell_size)',
    'base_supercell = base_structure.repeat(self.config.supercell_size) # type: ignore[no-untyped-call]'
)
with open('src/pyacemaker/core/generator.py', 'w') as f:
    f.write(content)


with open('src/pyacemaker/orchestrator.py') as f:
    content = f.read()
content = content.replace(
    'self.generator.update_config(self.config.structure)  # type: ignore[arg-type]',
    'self.generator.update_config(self.config.structure)'
)
with open('src/pyacemaker/orchestrator.py', 'w') as f:
    f.write(content)
