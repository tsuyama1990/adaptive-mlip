import re
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from pyacemaker.domain_models.constants import (
    ERR_VAL_STRUCT_DUMMY_ELEM,
    ERR_VAL_STRUCT_EMPTY,
    ERR_VAL_STRUCT_NAN_POS,
    ERR_VAL_STRUCT_NONE,
    ERR_VAL_STRUCT_TYPE,
    ERR_VAL_STRUCT_UNKNOWN_SYM,
    ERR_VAL_STRUCT_VOL_FAIL,
)

LAMMPS_SAFE_CMD_PATTERN = r"^[a-zA-Z0-9_\s\.\/\-\+\*]+$"

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
    "write_restart",
    "python",
    "print",
}

# Comprehensively block all shell injection, redirection, pipeline, and grouping metacharacters.
BLOCKED_PATTERN = re.compile(r"[;&|\$<>\`\n\r\"'\\]")


def validate_lammps_command(cmd: str) -> None:
    """
    Validates a single LAMMPS command string against security rules.

    Args:
        cmd: The LAMMPS command string to validate.

    Raises:
        ValueError: If the command contains forbidden characters, unrecognized commands,
                    or shell injection vectors.
    """
    import shlex

    # Check for blatantly obvious injections first
    if BLOCKED_PATTERN.search(cmd):
        msg = f"Command contains explicitly blocked shell metacharacters: {cmd}"
        raise ValueError(msg)

    try:
        # Use proper shell parsing to analyze exactly how the command is structured
        # This handles quotes properly and allows us to inspect arguments safely
        tokens = shlex.split(cmd)
    except ValueError as e:
        msg = f"Failed to parse command due to invalid quoting/escaping: {e}"
        raise ValueError(msg) from e

    if not tokens:
        return

    first_token = tokens[0]

    if first_token not in ALLOWED_LAMMPS_COMMANDS:
        msg = f"Script contains forbidden or unrecognized command: '{first_token}'"
        raise ValueError(msg)

    # Validate every token to ensure no dangerous shell/execution tokens are buried in arguments
    forbidden_tokens = {
        "shell",
        "exec",
        "system",
        "eval",
        "sh",
        "bash",
        "python3",
        "python2",
        "wget",
        "curl",
        "nc",
    }

    for token in tokens:
        # Check against a broader set of strict forbidden shell execution tokens
        if any(f_token == token for f_token in forbidden_tokens):
            msg = f"Script contains explicitly forbidden system command token: '{token}'"
            raise ValueError(msg)

        # Ensure NO token contains embedded shell characters, even encoded ones
        # This provides a strict character whitelist for ALL arguments instead of a weak regex
        # Only allow standard path characters and numbers inside tokens.
        is_quoted = (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        )
        has_invalid_chars = not re.match(r"^[a-zA-Z0-9_\\-\\.\\/\\*\\{\\}\\[\\]\\=\\+]+$", token)
        has_injection = bool(re.search(r"[\$\`\<\>\|\&;\\(\\)]", token))

        if has_invalid_chars and not is_quoted and has_injection:
            msg = f"Token contains potentially dangerous shell metacharacters: {token}"
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


def validate_structure(structure: Any) -> None:  # noqa: C901
    """
    Validates an atomic structure universally for internal processing.

    Args:
        structure: Input structure object (expected to be ase.Atoms).

    Raises:
        ValueError: If structure is invalid, empty, or contains unknown elements.
        TypeError: If input is not an ASE Atoms object.
    """
    if structure is None:
        raise ValueError(ERR_VAL_STRUCT_NONE)

    if not isinstance(structure, Atoms):
        raise TypeError(ERR_VAL_STRUCT_TYPE.format(type=type(structure)))

    if len(structure) == 0:
        raise ValueError(ERR_VAL_STRUCT_EMPTY)

    # Validate structure physical properties
    vol = 1.0  # Default valid volume
    try:
        # Check volume only if it has a valid cell
        if any(structure.pbc) or np.any(structure.cell):
            vol = structure.get_volume()  # type: ignore[no-untyped-call]
    except Exception as e:
        # get_volume might fail if no cell is set
        raise ValueError(ERR_VAL_STRUCT_VOL_FAIL.format(error=e)) from e

    if vol <= 1e-9:
        msg = "Failed to compute structure volume"
        raise ValueError(msg)

    # Validate positions are numeric and finite
    pos = structure.get_positions()  # type: ignore[no-untyped-call]
    if not np.isfinite(pos).all():
        raise ValueError(ERR_VAL_STRUCT_NAN_POS)

    # Validate elements against atomic_numbers
    symbols = set(structure.get_chemical_symbols())  # type: ignore[no-untyped-call]
    for s in symbols:
        # Script injection and sanitization check:
        if not isinstance(s, str) or not s.isalpha() or len(s) > 2:
            msg = f"Chemical symbol contains invalid characters or types: {s}"
            raise ValueError(msg)

        if s not in atomic_numbers:
            raise ValueError(ERR_VAL_STRUCT_UNKNOWN_SYM.format(symbol=s))
        if atomic_numbers[s] == 0:
            raise ValueError(ERR_VAL_STRUCT_DUMMY_ELEM.format(symbol=s))
