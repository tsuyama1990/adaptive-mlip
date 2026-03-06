from pathlib import Path
import re

p = Path("src/pyacemaker/core/trainer.py")
content = p.read_text()

# Fix static method self issue
content = content.replace(
    """    @staticmethod
    def _validate_training_data(data_path: Path) -> None:""",
    """    def _validate_training_data(self, data_path: Path) -> None:"""
)

# Fix exception messages
content = content.replace(
    'raise TrainerError("Environment variable PACE_TRAIN_EXECUTABLE is not set.")',
    'msg = "Environment variable PACE_TRAIN_EXECUTABLE is not set."\n            raise TrainerError(msg)'
)

content = content.replace(
    'raise TrainerError("PACE_TRAIN_EXECUTABLE contains invalid characters.")',
    'msg = "PACE_TRAIN_EXECUTABLE contains invalid characters."\n            raise TrainerError(msg)'
)

p.write_text(content)
