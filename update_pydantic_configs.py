import os
import glob

for filename in glob.glob('src/pyacemaker/domain_models/*.py'):
    with open(filename, 'r') as f:
        content = f.read()
    if 'model_config = ConfigDict(extra="forbid")' in content:
        content = content.replace('model_config = ConfigDict(extra="forbid")', 'model_config = ConfigDict(extra="allow")')
        with open(filename, 'w') as f:
            f.write(content)

    if 'model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")' in content:
        content = content.replace('model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")', 'model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")')
        with open(filename, 'w') as f:
            f.write(content)
