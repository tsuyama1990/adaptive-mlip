import re

with open("tests/uat/test_cycle08_uat.py", "r") as f:
    content = f.read()

# Mock subprocess.run for FinetuneManager in cycle08 tests
search = """    # 1. Finetune MACE
    finetune_mgr = FinetuneManager()"""

replace = """    # 1. Finetune MACE
    finetune_mgr = FinetuneManager()
    from unittest.mock import patch
    with patch("pyacemaker.utils.process.subprocess.run") as mock_run, patch("pyacemaker.utils.process.shutil.which") as mock_which:
        mock_which.return_value = "/bin/mace_run_train"
        mock_run.return_value.returncode = 0"""

content = content.replace(search, replace)

# Make sure we un-indent correctly around `finetune` call
search2 = """    dataset_path = tmp_path / "dataset.xyz"
    dataset_path.touch()
    awakened_model = finetune_mgr.finetune(dataset_path)"""

replace2 = """        dataset_path = tmp_path / "dataset.xyz"
        dataset_path.touch()
        awakened_model = finetune_mgr.finetune(dataset_path)"""
content = content.replace(search2, replace2)

with open("tests/uat/test_cycle08_uat.py", "w") as f:
    f.write(content)
