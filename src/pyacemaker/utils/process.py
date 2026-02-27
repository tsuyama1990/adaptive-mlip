import logging
import subprocess
from pathlib import Path

from pyacemaker.domain_models.constants import DANGEROUS_PATH_CHARS

logger = logging.getLogger(__name__)

# List of explicitly allowed commands to prevent execution of arbitrary binaries.
# This whitelist restricts the agent to only the necessary scientific codes.
ALLOWED_COMMANDS = {
    "pace_yaml2yace",
    "pace_train",
    "eonclient",
    "lmp",
    "mpirun",
    "mpiexec",
    "vasp_std",
    "vasp_gam",
    "vasp_ncl",
    "pw.x",
    "cp.x",
    "ph.x"
}

def run_command(
    cmd: list[str],
    cwd: str | None = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Executes a subprocess command safely with logging and strict allowlist validation.

    Args:
        cmd: List of command arguments.
        cwd: Current working directory for execution.
        check: Whether to raise CalledProcessError on non-zero exit code.
        capture_output: Whether to capture stdout/stderr.
        text: Whether to return output as string.

    Returns:
        CompletedProcess object.

    Raises:
        subprocess.CalledProcessError: If command fails and check=True.
        FileNotFoundError: If executable is not found.
        ValueError: If command is not explicitly allowed or contains dangerous characters.
    """
    if not cmd:
        msg = "Empty command list provided."
        raise ValueError(msg)

    # 1. Whitelist the executable
    executable = cmd[0]
    # Handle paths to executables (e.g. /usr/bin/mpirun) by checking the basename
    exe_basename = Path(executable).name
    if exe_basename not in ALLOWED_COMMANDS:
        msg = f"Executable '{exe_basename}' is not in the allowed list of commands."
        raise ValueError(msg)

    # 2. Strict Argument Validation (Blacklist dangerous chars)
    # Check for dangerous shell characters even if shell=False, as a defense-in-depth measure.
    # We use the centralized DANGEROUS_PATH_CHARS constant which includes ; & | ` $ etc.
    for arg in cmd:
        for char in DANGEROUS_PATH_CHARS:
             if char in arg:
                 msg = f"Argument contains potentially dangerous character '{char}': {arg}"
                 raise ValueError(msg)

    # Mask potentially sensitive arguments (basic heuristic)
    safe_cmd = []
    for arg in cmd:
        if len(arg) > 100:  # Truncate very long args
            safe_cmd.append(f"{arg[:20]}...[TRUNCATED]")
        else:
            safe_cmd.append(arg)

    safe_cmd_str = " ".join(safe_cmd)
    logger.debug(f"Running command: {safe_cmd_str}")

    try:
        return subprocess.run(  # noqa: S603
            cmd,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=text,
            shell=False,  # Enforce security
        )
    except subprocess.CalledProcessError as e:
        logger.exception(f"Command failed: {safe_cmd_str}. Exit code: {e.returncode}. Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.exception(f"Executable not found: {cmd[0]}")
        raise
