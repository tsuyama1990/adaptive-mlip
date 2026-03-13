import re
from pathlib import Path

from pyacemaker.domain_models.constants import LAMMPS_SAFE_CMD_PATTERN

SAFE_CMD_PATTERN = re.compile(LAMMPS_SAFE_CMD_PATTERN)

ALLOWED_LAMMPS_COMMANDS = {
    "clear",
    "units",
    "dimension",
    "boundary",
    "atom_style",
    "atom_modify",
    "lattice",
    "region",
    "create_box",
    "create_atoms",
    "read_data",
    "read_restart",
    "mass",
    "velocity",
    "pair_style",
    "pair_coeff",
    "pair_modify",
    "neighbor",
    "neigh_modify",
    "compute",
    "fix",
    "unfix",
    "uncompute",
    "thermo",
    "thermo_style",
    "thermo_modify",
    "dump",
    "dump_modify",
    "undump",
    "timestep",
    "reset_timestep",
    "run",
    "minimize",
    "min_style",
    "min_modify",
    "variable",
    "print",
    "write_restart",
}

BLOCKED_PATTERN = re.compile(r"[;&|\`<>\n\r]")


def validate_lammps_command(cmd: str) -> None:
    """
    Validates a single LAMMPS command string against security rules.

    Args:
        cmd: The LAMMPS command string to validate.

    Raises:
        ValueError: If the command contains forbidden characters, unrecognized commands,
                    or shell injection vectors.
    """
    if not SAFE_CMD_PATTERN.match(cmd):
        msg = f"Command contains forbidden characters: {cmd}"
        raise ValueError(msg)

    if BLOCKED_PATTERN.search(cmd):
        msg = f"Command contains explicitly blocked shell metacharacters: {cmd}"
        raise ValueError(msg)

    tokens = cmd.split()
    if not tokens:
        return

    first_token = tokens[0]

    if first_token not in ALLOWED_LAMMPS_COMMANDS:
        msg = f"Script contains forbidden or unrecognized command: '{first_token}'"
        raise ValueError(msg)

    if "shell" in tokens:
        msg = "Script contains forbidden command 'shell'."
        raise ValueError(msg)


def validate_lammps_script_file(script_path: Path) -> None:
    """
    Validates an entire LAMMPS script file for shell injection vulnerabilities.

    Args:
        script_path: The path to the script file.

    Raises:
        ValueError: If the script exceeds maximum size or contains forbidden commands.
    """
    max_size = 1024 * 1024  # 1MB limit
    if script_path.stat().st_size > max_size:
        msg = f"Script file size exceeds maximum limit of 1MB: {script_path}"
        raise ValueError(msg)

    with script_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                validate_lammps_command(line_str)
            except ValueError as e:
                msg = f"Forbidden command detected in LAMMPS script line {line_idx + 1} ({script_path}): {e}"
                raise ValueError(msg) from e
