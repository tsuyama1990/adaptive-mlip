with open("src/pyacemaker/interfaces/lammps_driver.py", "r") as f:
    content = f.read()

# Let's use a simpler string matching instead of complex python-in-python regex literal.
# Since we just want to avoid any of ; & | ` $ < > \n \r
replace = """    def _validate_command(self, cmd: str) -> None:
        \"\"\"Validates a single command against security rules.\"\"\"
        # Check for dangerous shell metacharacters anywhere in the command
        forbidden_chars = [";", "&", "|", "`", "$", "<", ">", "\\n", "\\r"]
        for char in forbidden_chars:
            if char in cmd:
                msg = f"Command contains forbidden characters: {cmd}"
                raise ValueError(msg)

        tokens = cmd.split()
        if not tokens:
            return

        first_token = tokens[0]"""

search = """    def _validate_command(self, cmd: str) -> None:
        \"\"\"Validates a single command against security rules.\"\"\"
        tokens = cmd.split()
        if not tokens:
            return

        first_token = tokens[0]"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/lammps_driver.py", "w") as f:
    f.write(content)
