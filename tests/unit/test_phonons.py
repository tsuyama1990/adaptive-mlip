from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.utils.phonons import PhononCalculator


@pytest.fixture
def mock_atoms() -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]], cell=[5, 5, 5], pbc=True)

@pytest.fixture
def mock_phonopy() -> Generator[MagicMock, None, None]:
    with patch("pyacemaker.utils.phonons.Phonopy") as mock:
        yield mock

def test_phonon_calculator_init(mock_phonopy: MagicMock, mock_atoms: Atoms) -> None:
    calc = PhononCalculator(mock_atoms, supercell_matrix=[2, 2, 2])
    assert calc.supercell_matrix == [2, 2, 2]
    mock_phonopy.assert_called_once()

def test_calculate_forces(mock_phonopy: MagicMock, mock_atoms: Atoms) -> None:
    calc = PhononCalculator(mock_atoms, supercell_matrix=[2, 2, 2])

    mock_instance = mock_phonopy.return_value

    # Mock supercell to provide valid data for Atoms constructor
    mock_supercell = MagicMock()
    mock_supercell.get_chemical_symbols.return_value = ["H", "H"]
    mock_supercell.get_scaled_positions.return_value = [[0,0,0], [0,0,0.5]]
    mock_supercell.get_cell.return_value = [[5,0,0], [0,5,0], [0,0,5]]

    mock_instance.supercells_with_displacements = [mock_supercell]

    # Mock force calculator function
    # Return valid force array
    mock_force_fn = MagicMock(return_value=[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])

    calc.calculate_forces(mock_force_fn)

    mock_force_fn.assert_called()
    assert len(mock_instance.forces) == 1
    mock_instance.produce_force_constants.assert_called_once()

def test_check_stability_stable(mock_phonopy: MagicMock, mock_atoms: Atoms) -> None:
    calc = PhononCalculator(mock_atoms, supercell_matrix=[2, 2, 2])
    mock_instance = mock_phonopy.return_value

    # Mock get_mesh_dict
    mock_instance.get_mesh_dict.return_value = {"frequencies": np.array([1.0, 2.0, 3.0])}

    is_stable, imaginary_freqs = calc.check_stability(tolerance=-0.05)
    assert is_stable is True
    assert len(imaginary_freqs) == 0

def test_check_stability_unstable(mock_phonopy: MagicMock, mock_atoms: Atoms) -> None:
    calc = PhononCalculator(mock_atoms, supercell_matrix=[2, 2, 2])
    mock_instance = mock_phonopy.return_value

    # Mock get_mesh_dict with unstable frequencies
    mock_instance.get_mesh_dict.return_value = {"frequencies": np.array([1.0, -1.0, -2.0])}

    is_stable, imaginary_freqs = calc.check_stability(tolerance=-0.05)
    assert is_stable is False
    assert len(imaginary_freqs) == 2
    assert -1.0 in imaginary_freqs
    assert -2.0 in imaginary_freqs

def test_get_band_structure_plot(mock_phonopy: MagicMock, mock_atoms: Atoms) -> None:
    calc = PhononCalculator(mock_atoms, supercell_matrix=[2, 2, 2])
    mock_instance = mock_phonopy.return_value

    mock_plot = MagicMock()
    mock_instance.plot_band_structure.return_value = mock_plot

    with patch("pyacemaker.utils.phonons.plot_to_base64") as mock_b64:
        mock_b64.return_value = "base64image"
        plot_b64 = calc.get_band_structure_plot()
        assert plot_b64 == "base64image"
