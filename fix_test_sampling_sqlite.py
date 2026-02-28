import re

with open('tests/unit/test_sampling_sqlite.py', 'r') as f:
    content = f.read()

content = content.replace(
    'class MockDescriptorCalculator:\n        def __init__(self, config: object) -> None: pass\n        def compute(self, atoms_list: list[Atoms]) -> np.ndarray:\n            return np.random.rand(len(atoms_list), 5)',
    'class MockDescriptorCalculator:\n        def __init__(self, config: object) -> None: pass\n        def compute(self, atoms_list: list[Atoms], batch_size: int = 100) -> np.ndarray:\n            return np.random.rand(len(atoms_list), 5)'
)

with open('tests/unit/test_sampling_sqlite.py', 'w') as f:
    f.write(content)
