with open("tests/uat/test_cycle08_uat.py", "r") as f:
    content = f.read()

search = """    with patch("pyacemaker.utils.process.subprocess.run") as mock_run, patch("pyacemaker.utils.process.shutil.which") as mock_which:
        mock_which.return_value = "/bin/mace_run_train"
        mock_run.return_value.returncode = 0"""

replace = """    with patch("pyacemaker.utils.process.subprocess.run") as mock_run, patch("shutil.which") as mock_which:
        mock_which.return_value = "/bin/mace_run_train"
        mock_run.return_value.returncode = 0"""

content = content.replace(search, replace)

with open("tests/uat/test_cycle08_uat.py", "w") as f:
    f.write(content)
