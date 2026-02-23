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
    cmd_str = " ".join(cmd)
    logger.debug(f"Running command: {cmd_str}")

    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=text,
            shell=False,  # Enforce security
        )
    except subprocess.CalledProcessError as e:
        logger.exception(f"Command failed: {cmd_str}. Exit code: {e.returncode}. Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.exception(f"Executable not found: {cmd[0]}")
        raise
