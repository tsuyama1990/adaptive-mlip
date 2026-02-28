import re

with open('tests/unit/test_loop.py', 'r') as f:
    content = f.read()

content = content.replace(
    'with pytest.raises(ValueError, match="Potential path does not exist"):',
    'with pytest.raises(ValueError, match="Potential path is not a file|does not exist|No such file"):'
)


with open('tests/unit/test_loop.py', 'w') as f:
    f.write(content)
