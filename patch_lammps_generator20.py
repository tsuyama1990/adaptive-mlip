import re

with open("src/pyacemaker/core/lammps_generator.py", "r") as f:
    content = f.read()

# Update `_gen_watchdog` to use `active_atoms` if spatial tagging was generated.
# Actually, if we just generate `group active_atoms union all` unconditionally when there are NO regions, then we can ALWAYS use `active_atoms` in `compute gamma`.
# But older configs might not have `active_atoms` generated. Let's just use it if it's there.
# Instead, let's conditionally write `compute gamma active_atoms pace ...` if `custom_initialization_commands` contains `"active_atoms"`.
# It's cleaner to check `any("active_atoms" in cmd for cmd in getattr(self.config, 'custom_initialization_commands', []))`

generator_replacement = """        safe_pot_path = validate_path_safe(potential_path)

        has_active_atoms = hasattr(self.config, 'custom_initialization_commands') and any("active_atoms" in cmd for cmd in self.config.custom_initialization_commands)
        target_group = "active_atoms" if has_active_atoms else "all"

        buffer.write(f"compute gamma {target_group} pace {safe_pot_path!s}\\n")
        buffer.write(f"compute max_gamma {target_group} reduce max c_gamma\\n")
        buffer.write("variable max_g equal c_max_gamma\\n")"""

content = re.sub(
    r"        safe_pot_path = validate_path_safe\(potential_path\)\n        buffer\.write\(f\"compute gamma all pace \{safe_pot_path!s\}\\n\"\)\n        buffer\.write\(\"compute max_gamma all reduce max c_gamma\\n\"\)\n        buffer\.write\(\"variable max_g equal c_max_gamma\\n\"\)",
    generator_replacement,
    content
)

with open("src/pyacemaker/core/lammps_generator.py", "w") as f:
    f.write(content)
