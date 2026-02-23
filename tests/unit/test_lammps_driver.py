import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

# If lammps is not importable, we mock it for the test definition to avoid ImportError
# But the test itself will use patching.
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
    # mock_lammps is the class mock
    mock_lammps.assert_called_once()
    assert driver.lmp == mock_lammps.return_value


def test_lammps_driver_run(mock_lammps: Any) -> None:
    """Tests running a script."""
    driver = LammpsDriver()
    script = "clear\nunits metal"
    driver.run(script)

    # Verify calls
    driver.lmp.command.assert_any_call("clear")
    driver.lmp.command.assert_any_call("units metal")


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

    # Mock return values
    driver.lmp.get_natoms.return_value = 2

    # Mock extract_box
    driver.lmp.extract_box.return_value = (
        [0.0, 0.0, 0.0], [10.0, 10.0, 10.0], 0.0, 0.0, 0.0, [1, 1, 1], 0
    )

    # Patch numpy.ctypeslib.as_array used in the driver
    with patch("pyacemaker.interfaces.lammps_driver.np.ctypeslib.as_array") as mock_as_array:
        # Side effect handles different array shapes: Positions (N, 3) and Types (N)
        def as_array_side_effect(ptr: Any, shape: tuple[int, ...]) -> Any:
            if shape == (2, 3):
                return np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            if shape == (2,):
                return np.array([1, 2], dtype=np.int32)
            return np.zeros(shape)

        mock_as_array.side_effect = as_array_side_effect

        atoms = driver.get_atoms(["Al", "Ni"])

    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert atoms[0].symbol == "Al"
    assert atoms[1].symbol == "Ni"
    assert np.allclose(atoms.positions[1], [2.0, 0.0, 0.0])
    assert np.allclose(atoms.cell, [[10, 0, 0], [0, 10, 0], [0, 0, 10]])
