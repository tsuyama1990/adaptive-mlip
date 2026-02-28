import re

with open('src/pyacemaker/core/policy.py', 'r') as f:
    content = f.read()

content = content.replace(
    'from pyacemaker.utils.perturbations import apply_strain',
    'import numpy as np\n        from pyacemaker.utils.perturbations import apply_strain'
)

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)
