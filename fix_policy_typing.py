
with open('src/pyacemaker/core/policy.py') as f:
    content = f.read()

if 'from typing import Any' not in content:
    content = content.replace('from collections.abc import Iterator', 'from collections.abc import Iterator\nfrom typing import Any')

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)
