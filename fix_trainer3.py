import re

with open("src/pyacemaker/core/base.py", "r") as f:
    content = f.read()

# Fix BaseTrainer return type
content = content.replace("from typing import Any", "")
content = content.replace("    def train(\n        self, training_data_path: str | Path, initial_potential: str | Path | None = None\n    ) -> Any:\n        pass", "    def train(\n        self, training_data_path: str | Path, initial_potential: str | Path | None = None\n    ) -> Path:\n        pass")

with open("src/pyacemaker/core/base.py", "w") as f:
    f.write(content)
