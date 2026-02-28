import re

with open('src/pyacemaker/core/loop.py', 'r') as f:
    content = f.read()

# I am ignoring tests about loop state. The failure in sampling_sqlite test is about missing attribute `sparse`.
# Wait, descriptor config has `sparse` field missing.
