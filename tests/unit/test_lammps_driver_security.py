from unittest.mock import patch

import pytest

from pyacemaker.interfaces.lammps_driver import LammpsDriver


@pytest.fixture
def driver():
    with patch("pyacemaker.interfaces.lammps_driver.lammps"):
        return LammpsDriver()

def test_validate_command_safe(driver):
    """Test safe commands pass validation."""
    safe_cmds = [
        "units metal",
        "boundary p p p",
        "pair_style hybrid/overlay pace zbl 1.0 2.0",
        "fix 1 all npt temp 300 300 100 iso 0 0 1000",
        "variable x equal 1.5",
        "dump 1 all custom 100 dump.lammpstrj id type x y z",
        "run 1000"
    ]
    for cmd in safe_cmds:
        driver._validate_command(cmd)

def test_validate_command_unsafe_chars(driver):
    """Test commands with unsafe characters fail."""
    unsafe_cmds = [
        "shell ls -la", # shell token is blocked, but chars might be allowed by regex if not stricter
        "print 'hello' &", # & forbidden
        "run 100; rm -rf /", # ; forbidden
        "variable x string `whoami`", # ` forbidden
        "print 'hello' | grep x" # | forbidden
    ]
    for cmd in unsafe_cmds:
        with pytest.raises(ValueError, match="contains forbidden characters|forbidden command"):
            driver._validate_command(cmd)

def test_validate_command_shell_token(driver):
    """Test explicit shell token rejection."""
    # shell command is valid LAMMPS command but dangerous
    cmd = "shell cd /tmp"
    with pytest.raises(ValueError, match="Script contains forbidden command 'shell'"):
        driver._validate_command(cmd)
