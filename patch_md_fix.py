import re

with open("src/pyacemaker/domain_models/md.py", "r") as f:
    content = f.read()

# Replace Enum field declaration for atom_style to use_enum_values=True
if "use_enum_values=True" not in content:
    content = content.replace('model_config = ConfigDict(extra="forbid", strict=True)', 'model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)')

with open("src/pyacemaker/domain_models/md.py", "w") as f:
    f.write(content)
