import sys
import re

with open("src/pyacemaker/core/base.py", "r") as f:
    content = f.read()

if "def incremental_train(" not in content:
    old_train = """    @abstractmethod
    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:"""

    new_train = """    @abstractmethod
    def incremental_train(
        self,
        new_data_path: str | Path,
        strategy_config: Any,
        initial_potential: str | Path | None = None,
    ) -> Any:
        \"\"\"
        Mixes a replay buffer with the new active learning data and runs incremental delta learning.
        \"\"\"

    @abstractmethod
    def train(
        self, training_data_path: str | Path, initial_potential: str | Path | None = None
    ) -> Any:"""

    content = content.replace(old_train, new_train)
    with open("src/pyacemaker/core/base.py", "w") as f:
        f.write(content)
