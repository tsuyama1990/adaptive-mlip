import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Let's fix FakeTrainer completely
fake_classes = """
class FakeTrainer(BaseTrainer):
    def __init__(self, output_pot: Path):
        self.output_pot = output_pot

    def train(self, training_data_path: str | Path, initial_potential: str | Path | None = None) -> Any:
        return self.output_pot

    def incremental_train(self, new_data_path: str | Path, strategy_config: Any, initial_potential: str | Path | None = None) -> Any:
        return self.output_pot
"""

if "def incremental_train" not in content:
    content = re.sub(r'class FakeTrainer\(BaseTrainer\):\n    def __init__\(self, output_pot: Path\):\n        self\.output_pot = output_pot\n\n    def train\(self, training_data_path: str \| Path, initial_potential: str \| Path \| None = None\) -> Any:\n        return self\.output_pot', fake_classes, content, flags=re.DOTALL)

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
