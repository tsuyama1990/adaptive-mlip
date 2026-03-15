import re

with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    content = f.read()

target = """        safe_pot_path = validate_path_safe(potential_path)
        buffer.write(f"compute gamma all pace {safe_pot_path!s}\\n")
        buffer.write("compute max_gamma all reduce max c_gamma\\n")
        buffer.write("variable max_g equal c_max_gamma\\n")"""

replacement = """        safe_pot_path = validate_path_safe(potential_path)

        has_active_atoms = hasattr(self.config, 'custom_initialization_commands') and any("active_atoms" in cmd for cmd in self.config.custom_initialization_commands)
        target_group = "active_atoms" if has_active_atoms else "all"

        buffer.write(f"compute gamma {target_group} pace {safe_pot_path!s}\\n")
        buffer.write(f"compute max_gamma {target_group} reduce max c_gamma\\n")
        buffer.write("variable max_g equal c_max_gamma\\n")"""

content = content.replace(target, replacement)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.write(content)
