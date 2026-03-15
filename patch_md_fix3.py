with open("src/pyacemaker/domain_models/md.py", "r") as f:
    content = f.read()

# Replace model_config to use_enum_values=True if it wasn't replaced
if "use_enum_values=True" not in content:
    content = content.replace('model_config = ConfigDict(extra="forbid", strict=True)', 'model_config = ConfigDict(extra="forbid", strict=True, use_enum_values=True)')

# Specifically for AtomStyle
if "class AtomStyle(StrEnum):" in content:
    pass

with open("src/pyacemaker/domain_models/md.py", "w") as f:
    f.write(content)
