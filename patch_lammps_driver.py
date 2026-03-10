with open("src/pyacemaker/interfaces/lammps_driver.py", "r") as f:
    content = f.read()

# Remove SAFE_CMD_PATTERN and its usage in _validate_command
search = """    # Whitelist of allowed characters in LAMMPS commands
    SAFE_CMD_PATTERN = re.compile(LAMMPS_SAFE_CMD_PATTERN)

    def __init__(self, cmdargs: list[str] | None = None) -> None:"""

replace = """    def __init__(self, cmdargs: list[str] | None = None) -> None:"""

content = content.replace(search, replace)

search_val = """    def _validate_command(self, cmd: str) -> None:
        \"\"\"Validates a single command against security rules.\"\"\"
        if not self.SAFE_CMD_PATTERN.match(cmd):
            msg = f"Command contains forbidden characters: {cmd}"
            raise ValueError(msg)

        tokens = cmd.split()"""

replace_val = """    def _validate_command(self, cmd: str) -> None:
        \"\"\"Validates a single command against security rules.\"\"\"
        tokens = cmd.split()"""

content = content.replace(search_val, replace_val)

search_list = """            "variable",
            "print",
        }"""

replace_list = """            "variable",
            "print",
            "include",
            "if",
            "jump",
            "label",
            "log",
            "echo",
            "set",
            "group",
            "displace_atoms",
            "write_data",
            "write_restart",
        }"""

content = content.replace(search_list, replace_list)

with open("src/pyacemaker/interfaces/lammps_driver.py", "w") as f:
    f.write(content)
