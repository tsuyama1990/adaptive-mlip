import re

with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    content = f.read()

target = r'        safe_pot_path = validate_path_safe\(potential_path\)\n        buffer\.write\(f"compute gamma all pace \{safe_pot_path!s\}\\n"\)\n        buffer\.write\("compute max_gamma all reduce max c_gamma\\n"\)\n        buffer\.write\("variable max_g equal c_max_gamma\\n"\)'

custom = """        safe_pot_path = validate_path_safe(potential_path)
        has_active = hasattr(self.config, 'custom_initialization_commands') and self.config.custom_initialization_commands and any('active_atoms' in cmd for cmd in self.config.custom_initialization_commands)
        tg = 'active_atoms' if has_active else 'all'
        buffer.write(f"compute gamma {tg} pace {safe_pot_path!s}\\n")
        buffer.write(f"compute max_gamma {tg} reduce max c_gamma\\n")
        buffer.write("variable max_g equal c_max_gamma\\n")"""

content = re.sub(target, custom, content)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.write(content)
