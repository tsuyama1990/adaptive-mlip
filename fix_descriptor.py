with open('src/pyacemaker/domain_models/active_learning.py', 'r') as f:
    content = f.read()

content = content.replace(
    'sigma: float | None = Field(None, gt=0.0, description="Gaussian smearing")',
    'sigma: float | None = Field(None, gt=0.0, description="Gaussian smearing")\n    sparse: bool = Field(False, description="Whether to use sparse descriptor matrix")'
)

with open('src/pyacemaker/domain_models/active_learning.py', 'w') as f:
    f.write(content)
