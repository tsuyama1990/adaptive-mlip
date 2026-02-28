with open('src/pyacemaker/orchestrator.py') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if i == 413:
        lines[i] = '            candidates_ase_gen,\n'

with open('src/pyacemaker/orchestrator.py', 'w') as f:
    f.writelines(lines)
