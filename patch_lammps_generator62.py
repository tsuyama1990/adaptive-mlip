with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'buffer.write(f"compute gamma all pace {safe_pot_path!s}\\n")' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + "has_active_atoms = hasattr(self.config, 'custom_initialization_commands') and self.config.custom_initialization_commands and any('active_atoms' in cmd for cmd in self.config.custom_initialization_commands)\n")
        new_lines.append(indent + "target_group = 'active_atoms' if has_active_atoms else 'all'\n")
        new_lines.append(indent + 'buffer.write(f"compute gamma {target_group} pace {safe_pot_path!s}\\n")\n')
    elif 'buffer.write("compute max_gamma all reduce max c_gamma\\n")' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + 'buffer.write(f"compute max_gamma {target_group} reduce max c_gamma\\n")\n')
    else:
        new_lines.append(line)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.writelines(new_lines)
