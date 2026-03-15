with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'buffer.write("compute max_gamma all reduce max c_gamma\\n")' in line:
        pass
    elif 'buffer.write(f"compute gamma all pace {safe_pot_path!s}\\n")' in line:
        new_lines.append("        has_active_atoms = hasattr(self.config, 'custom_initialization_commands') and any('active_atoms' in cmd for cmd in self.config.custom_initialization_commands)\n")
        new_lines.append("        target_group = 'active_atoms' if has_active_atoms else 'all'\n")
        new_lines.append("        buffer.write(f\"compute gamma {target_group} pace {safe_pot_path!s}\\n\")\n")
        new_lines.append("        buffer.write(f\"compute max_gamma {target_group} reduce max c_gamma\\n\")\n")
    else:
        new_lines.append(line)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.writelines(new_lines)
