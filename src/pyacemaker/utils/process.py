import logging
import re
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
    # Strict Argument Validation (Allowlist)
    # Block dangerous shell characters even if shell=False.
    # Expanded list of dangerous characters.
    dangerous_chars = re.compile(r"[;&|`$<>(){}\n\t*?]")
    for arg in cmd:
        if dangerous_chars.search(arg):
            # Redact argument in error message
            msg = "Argument contains potentially dangerous characters (redacted)"
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
        # Sanitize exception message to avoid exposing raw arguments
        sanitized_stderr = e.stderr[-200:] if e.stderr else "No stderr captured"
        logger.exception(
            f"Command failed: {safe_cmd_str}. Exit code: {e.returncode}. Stderr: {sanitized_stderr}"
        )
        # Raise with sanitized message
        raise subprocess.CalledProcessError(
            returncode=e.returncode, cmd=safe_cmd_str, output=e.output, stderr=e.stderr
        ) from e
    except FileNotFoundError:
        logger.exception(f"Executable not found: {cmd[0]}")
        raise
