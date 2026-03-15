import re

with open("src/pyacemaker/domain_models/workflow.py", "r") as f:
    content = f.read()

# Make sure ignored_atoms exists in ActiveLearningThresholds
replacement = """    smooth_steps: int = Field(
        default=3,
        description="Consecutive steps required to exceed threshold to exclude thermal noise",
    )
    ignored_atoms: list[int] = Field(
        default_factory=list,
        description="List of atom indices (1-based for LAMMPS) to ignore during variance calculation",
    )"""

content = re.sub(
    r"    smooth_steps: int = Field\(\n        default=3,\n        description=\"Consecutive steps required to exceed threshold to exclude thermal noise\",\n    \)",
    replacement,
    content
)

with open("src/pyacemaker/domain_models/workflow.py", "w") as f:
    f.write(content)
