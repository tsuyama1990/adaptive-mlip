import re

with open("tests/conftest.py", "r") as f:
    content = f.read()

content = content.replace('"ramping": False,', '"ramping": None,')
content = content.replace('"mc": False,', '"mc": None,')

with open("tests/conftest.py", "w") as f:
    f.write(content)
