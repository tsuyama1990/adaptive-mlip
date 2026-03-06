import re
from pathlib import Path

# Add typing Any back to base.py
p = Path("src/pyacemaker/core/base.py")
content = p.read_text()
content = "from typing import Any\n" + content
p.write_text(content)

# Fix BaseTrainer return type again correctly this time
content = content.replace("    def train(\n        self,\n        training_data_path: str | Path,\n        initial_potential: str | Path | None = None\n    ) -> Any:", "    def train(\n        self,\n        training_data_path: str | Path,\n        initial_potential: str | Path | None = None\n    ) -> Path:")
p.write_text(content)
