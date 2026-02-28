import re

with open('src/pyacemaker/domain_models/active_learning.py', 'r') as f:
    content = f.read()

content = content.replace(
    'model_config = ConfigDict(extra="forbid")',
    'model_config = ConfigDict(extra="allow")'
)

with open('src/pyacemaker/domain_models/active_learning.py', 'w') as f:
    f.write(content)
