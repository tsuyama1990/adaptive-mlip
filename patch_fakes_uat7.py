import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Oh, the base trainer is imported as `from pyacemaker.core.base import BaseEngine, BaseGenerator, BaseTrainer`.
# So it has `BaseTrainer` available.
# But `incremental_train` was added to `BaseTrainer` with `strategy_config` missing from the FakeTrainer signature, wait...
# The FakeTrainer definition I injected:
# `def incremental_train(self, new_data_path: str | Path, strategy_config: Any, initial_potential: str | Path | None = None) -> Any:`
# Did I get a TypeError because I didn't actually replace it properly?
# Let's just fix FakeTrainer using regex.

fake_classes = """
class FakeTrainer(BaseTrainer):
    def __init__(self, output_pot: Path):
        self.output_pot = output_pot

    def train(self, training_data_path: str | Path, initial_potential: str | Path | None = None) -> Any:
        return self.output_pot

    def incremental_train(self, new_data_path: str | Path, strategy_config: Any, initial_potential: str | Path | None = None) -> Any:
        return self.output_pot
"""

content = re.sub(r'class FakeTrainer\(BaseTrainer\):\n.*?def train\(self, training_data_path: str \| Path, initial_potential: str \| Path \| None = None\) -> Any:\n        return self\.output_pot\n', fake_classes, content, flags=re.DOTALL)

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
