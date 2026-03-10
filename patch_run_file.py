with open("src/pyacemaker/interfaces/lammps_driver.py", "r") as f:
    content = f.read()

search = """        # Security: Read the entire file into memory to avoid TOCTOU attacks
        # Validate all content atomically before executing anything.
        content = path.read_text(encoding="utf-8")

        commands_to_execute = []
        for line in content.splitlines():
            cmd = line.strip()
            if cmd.startswith("#"):
                continue
            if cmd:
                cmd = cmd.split("#")[0].strip()
                if cmd:
                    self._validate_command(cmd)
                    commands_to_execute.append(cmd)

        # If we reach here, all commands are valid and atomic. Execute them.
        for cmd in commands_to_execute:
            self.lmp.command(cmd)"""

replace = """        # Security: Process the file line by line to maintain O(1) memory overhead.
        # Ensure we parse and validate each line incrementally.
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                cmd = line.strip()
                if cmd.startswith("#"):
                    continue
                if cmd:
                    cmd = cmd.split("#")[0].strip()
                    if cmd:
                        self._validate_command(cmd)
                        self.lmp.command(cmd)"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/lammps_driver.py", "w") as f:
    f.write(content)
