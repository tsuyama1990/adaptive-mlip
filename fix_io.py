
with open('src/pyacemaker/utils/io.py') as f:
    content = f.read()

content = content.replace(
    'symbols = set()',
    'symbols: set[str] = set()'
)
content = content.replace(
    'new_syms = set(atoms.get_chemical_symbols())',
    'new_syms = set(atoms.get_chemical_symbols()) # type: ignore[no-untyped-call]'
)
content = content.replace(
    'cell = atoms.get_cell()',
    'cell = atoms.get_cell() # type: ignore[no-untyped-call]'
)
content = content.replace(
    'pos = atoms.get_positions() # (N, 3)',
    'pos = atoms.get_positions() # type: ignore[no-untyped-call]'
)
content = content.replace(
    'symbols = atoms.get_chemical_symbols() # List of strings (N)',
    'symbols = atoms.get_chemical_symbols() # type: ignore[no-untyped-call]'
)

with open('src/pyacemaker/utils/io.py', 'w') as f:
    f.write(content)
