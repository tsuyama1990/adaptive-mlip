import re

with open('src/pyacemaker/core/policy.py', 'r') as f:
    content = f.read()

# Make it actually modify the cell by random amount when magnitude is provided.
# Apply strain modifies in place already but strain_tensor shouldn't be a uniform scaling if we expect different volume from 0 strain. Wait, volume will be different even with uniform scaling.
content = content.replace(
    'np.diag([magnitude, magnitude, magnitude])',
    'np.random.uniform(-magnitude, magnitude, (3, 3))'
)

with open('src/pyacemaker/core/policy.py', 'w') as f:
    f.write(content)
