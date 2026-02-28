
with open('src/pyacemaker/modules/sampling.py') as f:
    content = f.read()

content = content.replace(
    'from pyacemaker.logger import get_logger',
    'import logging'
)
content = content.replace(
    'logger = get_logger()',
    'logger = logging.getLogger(__name__)'
)
content = content.replace(
    'db = ase.db.connect(str(tmp_db_path)) # type: ignore[no-untyped-call]',
    'db = ase.db.connect(str(tmp_db_path))'
)
content = content.replace(
    'db.write(atoms) # type: ignore[no-untyped-call]',
    'db.write(atoms)'
)
content = content.replace(
    'for row in db.select(): # type: ignore[no-untyped-call]',
    'for row in db.select():'
)
content = content.replace(
    'if row.id in selected_set: # type: ignore[attr-defined]',
    'if row.id in selected_set:'
)
content = content.replace(
    'atoms = row.toatoms() # type: ignore[no-untyped-call]',
    'atoms = row.toatoms()'
)

with open('src/pyacemaker/modules/sampling.py', 'w') as f:
    f.write(content)
