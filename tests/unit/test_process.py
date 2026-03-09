import logging
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.utils.process import run_command


def test_run_command_success():
    """Test successful command execution."""
    with patch("shutil.which", return_value="/bin/echo"), \
         patch("pyacemaker.utils.process.subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = run_command(["echo", "hello"])

        mock_run.assert_called_once_with(
            ["echo", "hello"],
            cwd=None,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        assert result == mock_result


def test_run_command_dangerous_characters():
    """Test that command with dangerous characters raises ValueError."""
    dangerous_cmds = [
        ["echo", "hello;"],
        ["ls", "&"],
        ["cat", "file|"],
        ["echo", "`ls`"],
        ["echo", "$USER"],
        ["rm", "-rf", "/&&echo"],
        ["cat", "file||die"],
        ["echo", "$(ls)"],
    ]

    with patch("shutil.which", return_value="/bin/echo"), \
         patch("pyacemaker.utils.process.subprocess.run") as mock_run:
        for cmd in dangerous_cmds:
            with pytest.raises(ValueError, match="Argument contains potentially dangerous characters"):
                run_command(cmd)

        # Verify that subprocess.run was never actually called
        mock_run.assert_not_called()

def test_run_command_shell_true():
    """Test that run_command properly prevents passing shell=True if somehow attempted."""
    with patch("shutil.which", return_value="/bin/echo"), \
         patch("pyacemaker.utils.process.subprocess.run") as mock_run:
        # Since run_command doesn't accept shell as a kwarg, it always passes shell=False
        # But if someone tries to hack the cmd array itself with shell constructs
        # the dangerous_characters check should catch it or the assert will prove shell=False.
        run_command(["echo", "hello"])
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs.get("shell") is False


def test_run_command_long_arguments(caplog):
    """Test that long arguments are truncated in the log."""
    long_arg = "a" * 150
    cmd = ["echo", long_arg]

    with caplog.at_level(logging.DEBUG), \
         patch("shutil.which", return_value="/bin/echo"), \
         patch("pyacemaker.utils.process.subprocess.run"):
        run_command(cmd)

    assert "[TRUNCATED]" in caplog.text
    assert long_arg[:20] in caplog.text
    assert len(caplog.text) < 150  # Make sure the full string wasn't logged


def test_run_command_called_process_error(caplog):
    """Test that CalledProcessError is caught, logged, and re-raised."""
    cmd = ["false"]
    error = subprocess.CalledProcessError(1, cmd, stderr="Command failed")

    with (
        patch("shutil.which", return_value="/bin/false"),
        patch("pyacemaker.utils.process.subprocess.run", side_effect=error),
        pytest.raises(subprocess.CalledProcessError),
    ):
        run_command(cmd)

    assert "Command failed: false" in caplog.text
    assert "Exit code: 1" in caplog.text
    assert "Stderr: Command failed" in caplog.text


def test_run_command_file_not_found_error(caplog):
    """Test that FileNotFoundError is raised when command is not in PATH."""
    cmd = ["nonexistent_command"]

    with (
        patch("shutil.which", return_value=None),
        pytest.raises(FileNotFoundError, match="Command not found or not executable")
    ):
        run_command(cmd)

def test_run_command_empty_cmd():
    """Test that empty command raises ValueError."""
    with pytest.raises(ValueError, match="Command list cannot be empty"):
        run_command([])
