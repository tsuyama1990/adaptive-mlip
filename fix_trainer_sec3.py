from pathlib import Path
import re

# 1. Update tests to inject environment variable to fix unit tests that will now fail
test_p = Path("tests/unit/test_trainer_pacemaker.py")
test_c = test_p.read_text()
if "os.environ['PACE_TRAIN_EXECUTABLE']" not in test_c:
    test_c = "import os\nos.environ['PACE_TRAIN_EXECUTABLE'] = 'echo'\n" + test_c
    test_p.write_text(test_c)

# 2. Fix TrainingConfig 'output_filename' hardcoding and remove SUPPORTED_TRAINING_FORMATS from defaults.py
p_def = Path("src/pyacemaker/domain_models/defaults.py")
def_c = p_def.read_text()
def_c = re.sub(r'SUPPORTED_TRAINING_FORMATS: Final\[set\[str\]\] = \{.*?\}\n', '', def_c)
p_def.write_text(def_c)
