import re

with open("src/pyacemaker/core/trainer.py", "r") as f:
    content = f.read()

# 1. Type Safety
content = content.replace("def train(\n        self,\n        training_data_path: str | Path,\n        initial_potential: str | Path | None = None\n    ) -> Any:", "def train(\n        self,\n        training_data_path: str | Path,\n        initial_potential: str | Path | None = None\n    ) -> Path:")

# 2. Static Method
content = content.replace("    def _validate_training_data(self, data_path: Path) -> None:", "    @staticmethod\n    def _validate_training_data(data_path: Path) -> None:")

# 3. Hardcoded extensions (Use domain models)
content = content.replace('if data_path.suffix not in {".pckl", ".xyz", ".extxyz", ".gzip"}:', 'from pyacemaker.domain_models.defaults import SUPPORTED_TRAINING_FORMATS\n        if data_path.suffix not in SUPPORTED_TRAINING_FORMATS:')

# 4. Security
content = content.replace('cmd.extend(["--initial_potential", str(initial_path)])', 'from pyacemaker.utils.path import validate_path_safe\n            safe_initial_path = validate_path_safe(initial_path)\n            cmd.extend(["--initial_potential", str(safe_initial_path)])')

# 5. Executable
content = content.replace('if not shutil.which("pace_train"):', 'import os\n        pace_train_exe = os.environ.get("PACE_TRAIN_EXECUTABLE", "pace_train")\n        if not shutil.which(pace_train_exe):')
content = content.replace('msg = "Executable \'pace_train\' not found in PATH."', 'msg = f"Executable \'{pace_train_exe}\' not found in PATH."')
content = content.replace('cmd = ["pace_train", str(input_yaml_path)]', 'cmd = [pace_train_exe, str(input_yaml_path)]')

with open("src/pyacemaker/core/trainer.py", "w") as f:
    f.write(content)

# Update defaults.py
with open("src/pyacemaker/domain_models/defaults.py", "r") as f:
    defaults_content = f.read()

if "SUPPORTED_TRAINING_FORMATS" not in defaults_content:
    defaults_content += '\nSUPPORTED_TRAINING_FORMATS: Final[set[str]] = {".pckl", ".xyz", ".extxyz", ".gzip"}\n'

with open("src/pyacemaker/domain_models/defaults.py", "w") as f:
    f.write(defaults_content)
