import logging
import os
import re
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def run_command(
    cmd: list[str],
    cwd: str | None = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    """
    Executes a subprocess command safely with logging and error handling.

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
    """
    # Strict Argument Validation (Allowlist)
    # We allow alphanumeric, dot, dash, underscore, slash, equal, comma, colon.
    # This covers paths, simple options, and numbers.
    # Special characters (like &, ;, |, $) which are dangerous in shells are rejected.

    # Check if command exists and has execute permissions
    if not cmd:
        msg = "Command list cannot be empty."
        raise ValueError(msg)

    executable = cmd[0]
    full_path = shutil.which(executable)
    if not full_path or not os.access(full_path, os.X_OK):
        msg = f"Command not found or not executable: {executable}"
        raise FileNotFoundError(msg)

    # Add additional validation to ensure the executable is in a trusted directory.
    # We resolve the real path to prevent symlink traversal to untrusted locations.
    real_path = os.path.realpath(full_path)
    from pathlib import Path
    trusted_dirs = [
        "/bin", "/usr/bin", "/usr/local/bin", "/opt/conda/bin",
        str(Path("~/.local/bin").expanduser()), str(Path("~/.cargo/bin").expanduser()),
        # Allow current virtual environment bin directory
        str(Path(sys.executable).parent) if hasattr(sys, 'executable') else ""
    ]
    # Check if the executable is within one of the trusted directories
    is_trusted = any(real_path.startswith(os.path.realpath(td) + os.sep) for td in trusted_dirs if td)
    if not is_trusted:
        msg = f"Executable is not in a trusted directory: {real_path}"
        raise PermissionError(msg)

    # Comprehensive allowlist approach for argument characters.
    # We only allow alphanumeric characters and a strict set of safe punctuation.
    allowed_pattern = re.compile(r"^[a-zA-Z0-9_./\-+:=,@]+$")
    for arg in cmd:
        if not allowed_pattern.match(arg):
            msg = f"Argument contains non-allowlisted characters: {arg}"
            raise ValueError(msg)

    # Mask potentially sensitive arguments (basic heuristic)
    # We redact arguments that look like they might be sensitive keys or very long strings
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
        logger.exception(
            f"Command failed: {safe_cmd_str}. Exit code: {e.returncode}. Stderr: {e.stderr}"
        )
        raise
    except FileNotFoundError:
        logger.exception(f"Executable not found: {cmd[0]}")
        raise
