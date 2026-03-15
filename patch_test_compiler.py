import re

with open("tests/integration/test_compiler.py", "r") as f:
    content = f.read()

# Fix the test failure: custom_initialization_commands missing on md_config because Pydantic models might be strict or something else.
# The error was "AttributeError: 'MDConfig' object has no attribute 'custom_initialization_commands'"
# Ah, I added `custom_initialization_commands` to `MDConfig` in `md.py` but the error implies it's missing! Let's check `md.py`
