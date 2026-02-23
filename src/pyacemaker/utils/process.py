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
        logger.exception(f"Command failed: {safe_cmd_str}. Exit code: {e.returncode}. Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.exception(f"Executable not found: {cmd[0]}")
        raise
