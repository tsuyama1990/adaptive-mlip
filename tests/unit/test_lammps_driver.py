import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if "lammps" not in sys.modules:
    sys.modules["lammps"] = MagicMock()

from pyacemaker.interfaces.lammps_driver import LammpsDriver


@pytest.fixture
def mock_lammps() -> Any:
    with patch("pyacemaker.interfaces.lammps_driver.lammps") as mock:
        yield mock


def test_lammps_driver_init(mock_lammps: Any) -> None:
    """Tests LammpsDriver initialization."""
    driver = LammpsDriver()
    mock_lammps.assert_called_once()
    assert driver.lmp == mock_lammps.return_value


def test_lammps_driver_init_failure(mock_lammps: Any) -> None:
    """Tests initialization failure."""
    mock_lammps.side_effect = OSError("Library not found")
    with pytest.raises(RuntimeError, match="Failed to initialize LAMMPS"):
        LammpsDriver()


def test_lammps_driver_run(mock_lammps: Any) -> None:
    """Tests running a script."""
    driver = LammpsDriver()
    script = "clear\nunits metal"
    driver.run(script)
    driver.lmp.command.assert_any_call("units metal")


def test_lammps_driver_run_unsafe(mock_lammps: Any) -> None:
    """Tests rejection of unsafe (non-ASCII) scripts."""
    driver = LammpsDriver()
    # Non-ascii:
    script_unsafe = "print 'Hello \uffff'"
    with pytest.raises(ValueError, match="Script contains non-ASCII"):
        driver.run(script_unsafe)


def test_lammps_driver_run_forbidden_chars(mock_lammps: Any) -> None:
    """Tests rejection of scripts with forbidden characters."""
    driver = LammpsDriver()
    # Pipe is forbidden
    script = "print 'Hello' | grep World"
    with pytest.raises(ValueError, match="forbidden characters"):
        driver.run(script)


def test_lammps_driver_run_forbidden_command(mock_lammps: Any) -> None:
    """Tests rejection of scripts with forbidden commands."""
    driver = LammpsDriver()
    # shell command is forbidden
    script = "shell rm -rf /"
    with pytest.raises(ValueError, match="forbidden command 'shell'"):
        driver.run(script)


def test_lammps_driver_run_file_calls_command(mock_lammps: Any, tmp_path: Path) -> None:
    """Test run_file executes line-by-line via lmp.command() instead of lmp.file()."""
    driver = LammpsDriver()
    script_content = "clear\nunits metal\n"
    script_path = tmp_path / "test.lmp"
    script_path.write_text(script_content)

    driver.run_file(script_path)

    # Verify command was called for each line
    driver.lmp.command.assert_any_call("clear")
    driver.lmp.command.assert_any_call("units metal")

    # Verify lmp.file was NOT called
    driver.lmp.file.assert_not_called()


def test_lammps_driver_run_shell(mock_lammps: Any) -> None:
    """Tests rejection of shell command."""
    driver = LammpsDriver()
    script = "shell ls"
    with pytest.raises(ValueError, match="forbidden command 'shell'"):
        driver.run(script)


def test_lammps_driver_extract_variable(mock_lammps: Any) -> None:
    """Tests extracting a variable."""
    driver = LammpsDriver()
    driver.lmp.extract_variable.return_value = 123.45
    val = driver.extract_variable("my_var")
    driver.lmp.extract_variable.assert_called_with("my_var", None, 0)
    assert val == 123.45


def test_lammps_driver_get_atoms(mock_lammps: Any) -> None:
    """Tests retrieving Atoms object."""
    driver = LammpsDriver()
    driver.lmp.get_natoms.return_value = 2
    driver.lmp.extract_box.return_value = (
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        0.0,
        0.0,
        0.0,
        [1, 1, 1],
        0,
    )

    with patch("pyacemaker.interfaces.lammps_driver.np.ctypeslib.as_array") as mock_as_array:

        def as_array_side_effect(ptr: Any, shape: tuple[int, ...]) -> Any:
            if shape == (2, 3):
                return np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            if shape == (2,):
                return np.array([1, 2], dtype=np.int32)
            return np.zeros(shape)

        mock_as_array.side_effect = as_array_side_effect
        atoms = driver.get_atoms(["Al", "Ni"])

    assert len(atoms) == 2
    assert atoms[0].symbol == "Al"
    assert atoms[1].symbol == "Ni"


def test_lammps_driver_get_atoms_invalid_type(mock_lammps: Any) -> None:
    """Tests error when LAMMPS type is out of range."""
    driver = LammpsDriver()
    driver.lmp.get_natoms.return_value = 1
    driver.lmp.extract_box.return_value = (
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        0.0,
        0.0,
        0.0,
        [1, 1, 1],
        0,
    )

    with patch("pyacemaker.interfaces.lammps_driver.np.ctypeslib.as_array") as mock_as_array:
        # Return type 2, but only 1 element provided
        mock_as_array.side_effect = lambda ptr, shape: (
            np.array([2], dtype=np.int32) if shape == (1,) else np.zeros((1, 3))
        )

        with pytest.raises(ValueError, match="index out of range"):
            driver.get_atoms(["Al"])
