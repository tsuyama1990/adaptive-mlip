with open('src/pyacemaker/orchestrator.py') as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if line.strip() == 'self.generator.update_config(self.config.structure) # type: ignore[arg-type]':
        lines[i] = ' ' * (len(line) - len(line.lstrip())) + 'self.generator.update_config(self.config.structure)\n'
    elif 'type: ignore' in line and 'update_config' in line:
        lines[i] = line.replace(' # type: ignore[arg-type]', '')
with open('src/pyacemaker/orchestrator.py', 'w') as f:
    f.writelines(lines)
