import re

with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    content = f.read()

target = '        buffer.write(f"compute gamma all pace {safe_pot_path!s}\\n")\n'
target += '        buffer.write("compute max_gamma all reduce max c_gamma\\n")\n'
target += '        buffer.write("variable max_g equal c_max_gamma\\n")'

custom = '        has_active_atoms = hasattr(self.config, "custom_initialization_commands") and any("active_atoms" in cmd for cmd in self.config.custom_initialization_commands)\n'
custom += '        target_group = "active_atoms" if has_active_atoms else "all"\n'
custom += '        buffer.write(f"compute gamma {target_group} pace {safe_pot_path!s}\\n")\n'
custom += '        buffer.write(f"compute max_gamma {target_group} reduce max c_gamma\\n")\n'
custom += '        buffer.write("variable max_g equal c_max_gamma\\n")'

# Only replace the target, ensuring we don't mess up indentations elsewhere
new_content = content.replace(target, custom)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.write(new_content)
