from pathlib import Path
import re

# 1. Clean up old script files
for f in Path(".").glob("fix_*.py"):
    if f.name != "fix_trainer_sec.py":
        f.unlink()

# 2. Fix trainer.py security issues
p = Path("src/pyacemaker/core/trainer.py")
content = p.read_text()

# We'll explicitly validate PACE_TRAIN_EXECUTABLE against safe characters and enforce it exists
# We will use the existing DANGEROUS_PATH_CHARS or LAMMPS_SAFE_CMD_PATTERN concepts
replacement = """        pace_train_exe = os.environ.get("PACE_TRAIN_EXECUTABLE")
        if not pace_train_exe:
            raise TrainerError("Environment variable PACE_TRAIN_EXECUTABLE is not set.")

        import re
        if not re.match(r"^[a-zA-Z0-9_\\-\\.\\/]+$", pace_train_exe):
            raise TrainerError("PACE_TRAIN_EXECUTABLE contains invalid characters.")

        if not shutil.which(pace_train_exe):"""

content = re.sub(
    r'        pace_train_exe = os.environ.get\("PACE_TRAIN_EXECUTABLE", "pace_train"\)\n        if not shutil.which\(pace_train_exe\):',
    replacement,
    content
)

# 3. Ensure extensions are part of TrainingConfig, not defaults
p_train_config = Path("src/pyacemaker/domain_models/training.py")
config_content = p_train_config.read_text()
if "supported_formats: list[str]" not in config_content:
    config_content = config_content.replace(
        'max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")',
        'max_basis_size: PositiveInt = Field(default=1000, description="Maximum basis size")\n    supported_formats: list[str] = Field(default_factory=lambda: [".pckl", ".xyz", ".extxyz", ".gzip"], description="Allowed training data formats")'
    )
    p_train_config.write_text(config_content)

# Update trainer.py to use this config field
content = content.replace(
    'from pyacemaker.domain_models.defaults import SUPPORTED_TRAINING_FORMATS\n        if data_path.suffix not in SUPPORTED_TRAINING_FORMATS:',
    'if data_path.suffix not in self.config.supported_formats:'
)
p.write_text(content)
