import re

with open('tests/unit/test_sampling_sqlite.py', 'r') as f:
    content = f.read()

content = content.replace(
    '    def compute(self, atoms_list: list[Atoms]) -> np.ndarray:\n        return np.random.rand(len(atoms_list), 10)',
    '    def compute(self, atoms_list: list[Atoms], batch_size: int = 100) -> np.ndarray:\n        return np.random.rand(len(atoms_list), 10)'
)

with open('tests/unit/test_sampling_sqlite.py', 'w') as f:
    f.write(content)
