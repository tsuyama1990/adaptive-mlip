with open("src/pyacemaker/interfaces/lammps_driver.py", "r") as f:
    content = f.read()

search = """    def _validate_command(self, cmd: str) -> None:
        \"\"\"Validates a single command against security rules.\"\"\"
        tokens = cmd.split()
        if not tokens:
            return

        first_token = tokens[0]"""

replace = """    def _validate_command(self, cmd: str) -> None:
        \"\"\"Validates a single command against security rules.\"\"\"
        # Check for dangerous shell metacharacters anywhere in the command
        import re
        if re.search(r"[;&\|`$<>\n\r]", cmd):
            msg = f"Command contains forbidden characters: {cmd}"
            raise ValueError(msg)

        tokens = cmd.split()
        if not tokens:
            return

        first_token = tokens[0]"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/lammps_driver.py", "w") as f:
    f.write(content)
