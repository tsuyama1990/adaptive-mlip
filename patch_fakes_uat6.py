import sys
import re

with open("tests/e2e/test_orchestrator_refinement.py", "r") as f:
    content = f.read()

# Oh the fake trainer is failing to instantiate in test_orchestrator_refinement_extraction_failure
# because it was using an old FakeTrainer. The fix we applied earlier was only for test_orchestrator_refinement_logic, or maybe it didn't apply properly because of strict string matching.

fake_classes = """
class FakeTrainer(BaseTrainer):
    def __init__(self, output_pot: Path):
        self.output_pot = output_pot

    def train(self, training_data_path: str | Path, initial_potential: str | Path | None = None) -> Any:
        return self.output_pot

    def incremental_train(self, new_data_path: str | Path, strategy_config: Any, initial_potential: str | Path | None = None) -> Any:
        return self.output_pot
"""

content = re.sub(r'class FakeTrainer\(BaseTrainer\):\n.*?return self\.output_pot\n', fake_classes, content, flags=re.DOTALL)

with open("tests/e2e/test_orchestrator_refinement.py", "w") as f:
    f.write(content)
