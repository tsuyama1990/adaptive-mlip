from pathlib import Path

content = Path("src/pyacemaker/domain_models/defaults.py").read_text()
if "DEFAULT_TAG_ASSIGNMENT_STRATEGY" not in content:
    content += "\nDEFAULT_TAG_ASSIGNMENT_STRATEGY = 'priority'\n"
    Path("src/pyacemaker/domain_models/defaults.py").write_text(content)
