
with open('src/pyacemaker/core/base.py') as f:
    content = f.read()

content = content.replace(
    'def generate(self, **kwargs: Any) -> None:',
    'def generate(self, base_structure: Atoms, config: Any, n_structures: int, **kwargs: Any) -> Iterator[Atoms]:'
)

with open('src/pyacemaker/core/base.py', 'w') as f:
    f.write(content)
