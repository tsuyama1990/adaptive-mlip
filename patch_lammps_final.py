with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    content = f.read()

import re

# We see unterminated f-string due to a bad replacement in previous iterations. Let's fix lines 120-130 explicitly.
lines = content.split('\n')
fixed_lines = []
skip = False
for i, line in enumerate(lines):
    if line.strip().startswith('buffer.write(f"compute gamma') and '{target_group}' in line:
        fixed_lines.append('        buffer.write(f"compute gamma {target_group} pace {safe_pot_path!s}\\n")')
        skip = True
    elif skip and line.strip() == '")':
        skip = False
    elif skip and line.strip().startswith('buffer.write(f"compute max_gamma'):
        fixed_lines.append('        buffer.write(f"compute max_gamma {target_group} reduce max c_gamma\\n")')
        # Skip the next line which is '")'
    elif skip and line.strip().startswith('buffer.write("variable max_g equal c_max_gamma'):
        fixed_lines.append('        buffer.write("variable max_g equal c_max_gamma\\n")')
    else:
        if not skip:
            fixed_lines.append(line)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.write('\n'.join(fixed_lines))
