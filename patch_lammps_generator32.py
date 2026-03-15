with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == 'buffer.write(f"compute gamma all pace {safe_pot_path!s}\\n")':
        new_lines.append("        has_active = hasattr(self.config, 'custom_initialization_commands') and any('active_atoms' in cmd for cmd in self.config.custom_initialization_commands)\n")
        new_lines.append("        tg = 'active_atoms' if has_active else 'all'\n")
        new_lines.append("        buffer.write(f\"compute gamma {tg} pace {safe_pot_path!s}\\n\")\n")
    elif line.strip() == 'buffer.write("compute max_gamma all reduce max c_gamma\\n")':
        new_lines.append("        buffer.write(f\"compute max_gamma {tg} reduce max c_gamma\\n\")\n")
    else:
        new_lines.append(line)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.writelines(new_lines)
