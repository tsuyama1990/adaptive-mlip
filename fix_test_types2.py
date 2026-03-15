import re
import sys

def add_types(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find def test_xyz(args):
    # and replace args with `mock: Any, tmp_path: Path` etc. It's complex. Let's just fix it manually if it fails MyPy.
    # Actually, ruff or mypy --install-types might do it? No.
    # I'll just manually fix them.

import os
os.system('uv run mypy tests/unit/test_process.py tests/unit/test_eon_driver.py tests/unit/test_domain_models_validation.py')
