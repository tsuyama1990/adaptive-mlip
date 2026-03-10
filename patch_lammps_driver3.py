with open("src/pyacemaker/interfaces/lammps_driver.py", "r") as f:
    content = f.read()

search = """        # Security: Process the file line by line to maintain O(1) memory overhead.
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

replace = """        # Security: Process the file line by line to maintain O(1) memory overhead.
        # Ensure we parse and validate each line incrementally, considering continuations.
        with path.open("r", encoding="utf-8") as file:
            accumulated_cmd = ""
            for line in file:
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("#"):
                    continue

                # Strip trailing comments
                cmd_part = line_stripped.split("#")[0].strip()
                if not cmd_part:
                    continue

                if cmd_part.endswith("&"):
                    # Accumulate line without the '&'
                    accumulated_cmd += cmd_part[:-1] + " "
                else:
                    accumulated_cmd += cmd_part
                    self._validate_command(accumulated_cmd)
                    self.lmp.command(accumulated_cmd)
                    accumulated_cmd = ""

            # If the file ended with a continuation, execute it
            if accumulated_cmd.strip():
                self._validate_command(accumulated_cmd)
                self.lmp.command(accumulated_cmd)"""

content = content.replace(search, replace)

with open("src/pyacemaker/interfaces/lammps_driver.py", "w") as f:
    f.write(content)
