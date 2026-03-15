with open("src/pyacemaker/domain_models/md.py", "r") as f:
    content = f.read()

# We might need a `@field_validator("atom_style", mode="before")` to handle the AtomStyle parsing from tests directly passing strings.
# Or just let it be since Pydantic does string->enum automatically without use_enum_values=True if it is passed in properly. Wait, use_enum_values=True makes it dict-serializable but Pydantic 2 strictly enforces Enum objects natively when reading unless we do `mode="before"`.
content = content.replace("model_config = ConfigDict(extra=\"forbid\", strict=True, use_enum_values=True)", "model_config = ConfigDict(extra=\"forbid\", strict=False)")

with open("src/pyacemaker/domain_models/md.py", "w") as f:
    f.write(content)
