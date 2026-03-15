with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    content = f.read()

target1 = '        buffer.write(f"compute gamma all pace {safe_pot_path!s}\\n")'
target2 = '        buffer.write("compute max_gamma all reduce max c_gamma\\n")'

custom1 = """        has_active = hasattr(self.config, 'custom_initialization_commands') and any('active_atoms' in cmd for cmd in self.config.custom_initialization_commands)
        tg = 'active_atoms' if has_active else 'all'
        buffer.write(f"compute gamma {tg} pace {safe_pot_path!s}\\n")"""

custom2 = '        buffer.write(f"compute max_gamma {tg} reduce max c_gamma\\n")'

content = content.replace(target1, custom1)
content = content.replace(target2, custom2)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.write(content)
