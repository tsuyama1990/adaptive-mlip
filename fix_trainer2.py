import re

with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

# Fix the import since Any is no longer used
content = content.replace("from typing import Any", "")

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)
