import re
from typing import TextIO


class ScriptTemplate:
    """
    Abstracts script generation to validate commands explicitly before writing.
    """
    def __init__(self, buffer: TextIO) -> None:
        self.buffer = buffer

    def write(self, command: str) -> None:
        """
        Validates the LAMMPS command string for obvious dangerous injection vectors.
        """
        # Block arbitrary shell escapes in LAMMPS natively
        if "shell " in command and not command.strip().startswith("#"):
            msg = f"Shell commands are explicitly forbidden in script generation: {command}"
            raise ValueError(msg)

        # Ensure no system command separators injected inadvertently
        if re.search(r"[;&|`$]", command):
            msg = f"Invalid shell metacharacters detected in LAMMPS script generation: {command}"
            raise ValueError(msg)

        self.buffer.write(command)
