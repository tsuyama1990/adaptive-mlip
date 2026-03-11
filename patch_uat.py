with open("tests/uat/test_cycle08_uat.py") as f:
    content = f.read()

# Add a dummy atom to dataset.xyz so it's not empty
content = content.replace(
    "dataset_path.touch()", "dataset_path.write_text('1\\n\\nFe 0.0 0.0 0.0\\n')"
)

with open("tests/uat/test_cycle08_uat.py", "w") as f:
    f.write(content)

with open("tests/uat/test_cycle06_uat.py") as f:
    content = f.read()

# In test_cycle06_uat.py
content = content.replace(
    "with patch('pyacemaker.factory.DFTManager'):\n        with patch('pyacemaker.factory.Validator'):",
    "with patch('pyacemaker.factory.DFTManager'):\n        with patch('pyacemaker.factory.Validator'):\n            with patch('pyacemaker.factory.MACEManager'):",
)
with open("tests/uat/test_cycle06_uat.py", "w") as f:
    f.write(content)
