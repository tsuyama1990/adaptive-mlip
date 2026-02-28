
with open('src/pyacemaker/core/generator.py') as f:
    content = f.read()

content = content.replace(
    'base_supercell = base_structure.repeat(self.config.supercell_size)  # type: ignore[no-untyped-call]',
    'base_supercell = base_structure.repeat(self.config.supercell_size)'
)

with open('src/pyacemaker/core/generator.py', 'w') as f:
    f.write(content)
