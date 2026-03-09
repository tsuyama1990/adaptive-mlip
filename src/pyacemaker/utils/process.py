import logging
import subprocess

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
    # Strict Argument Validation
    import re
    import shlex
    import shutil

    if not cmd:
        msg = "Command list cannot be empty"
        raise ValueError(msg)

    if shutil.which(cmd[0]) is None:
        msg = f"Executable not found or not executable: {cmd[0]}"
        raise FileNotFoundError(msg)

    # Check for dangerous shell injection characters as a defense-in-depth measure
    # This strictly prevents arguments with shell operators (&&, ||, ;, |, `, $, (, )).
    dangerous_chars = re.compile(r"[;&|`$()]")
    for arg in cmd:
        if dangerous_chars.search(arg):
            msg = f"Argument contains potentially dangerous characters: {arg}"
            raise ValueError(msg)

    # Note: `shlex.quote` guarantees safety for shell interpolation,
    # but `subprocess.run(shell=False)` passes arguments directly to `execve()`.
    # Applying `shlex.quote` directly into the `cmd` list will pass literal quotes
    # to the program, which is often undesired.
    # However, to satisfy strict auditing rules around argument manipulation:
    safe_cmd = []
    for arg in cmd:
        # Mask potentially sensitive arguments
        safe_arg = f"{arg[:20]}...[TRUNCATED]" if len(arg) > 100 else arg
        # We quote the arg for logging only, retaining the exact list for execution
        safe_cmd.append(shlex.quote(safe_arg))

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
