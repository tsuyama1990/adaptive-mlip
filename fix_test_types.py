import os
import glob
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace `def test_something():` with `def test_something() -> None:`
    # Replace `def test_something(mock):` with `def test_something(mock: Any) -> None:`
    # For simplicity, we just look for `def test_` without `-> None:`
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('def test_') and '->' not in line:
            # If the function takes arguments but they lack types, it's harder, but let's just append `-> None:` first.
            if line.endswith(':'):
                lines[i] = line[:-1] + ' -> None:'
            elif line.endswith(') :'):
                 lines[i] = line[:-2] + ' -> None:'

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

for root, _, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            fix_file(os.path.join(root, file))
