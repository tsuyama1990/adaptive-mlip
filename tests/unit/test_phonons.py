from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.base import BaseEngine
from pyacemaker.utils.phonons import PhononCalculator


class TestPhononCalculator:
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock(spec=BaseEngine)
        result = MagicMock()
        result.forces = [[0.0, 0.0, 0.0]]
        engine.compute_static_properties.return_value = result
        return engine

    @pytest.fixture
    def calculator(self, mock_engine):
        return PhononCalculator(
            engine=mock_engine, supercell_matrix=[2, 2, 2], displacement=0.01, imaginary_tol=-0.05
        )

    @patch("pyacemaker.utils.phonons.Phonopy")
    def test_check_stability_stable(self, mock_phonopy_cls, calculator):
        mock_phonopy = mock_phonopy_cls.return_value

        # Mock band structure frequencies
        # Phonopy.get_band_structure_dict() returns a dict with 'frequencies' list of arrays
        mock_phonopy.get_band_structure_dict.return_value = {
            "frequencies": [np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5])],
            "qpoints": [],
            "distances": [],
        }

        structure = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.8, 2.8, 2.8], pbc=True)
        potential_path = Path("pot.yace")

        # We also need to mock produce_force_constants or forces
        # The calculator will call generate_displacements, then compute forces, then set_forces
        mock_phonopy.get_supercells_with_displacements.return_value = [structure.copy()]

        is_stable, plot = calculator.check_stability(structure, potential_path)

        assert is_stable is True
        assert isinstance(plot, str)

    @patch("pyacemaker.utils.phonons.Phonopy")
    def test_check_stability_unstable(self, mock_phonopy_cls, calculator):
        mock_phonopy = mock_phonopy_cls.return_value

        # Mock imaginary frequencies (negative values)
        mock_phonopy.get_band_structure_dict.return_value = {
            "frequencies": [
                np.array([-1.0, 2.0, 3.0])  # -1.0 is < -0.05 tolerance
            ],
            "qpoints": [],
            "distances": [],
        }

        structure = Atoms("Fe", positions=[[0, 0, 0]], cell=[2.8, 2.8, 2.8], pbc=True)
        potential_path = Path("pot.yace")

        mock_phonopy.get_supercells_with_displacements.return_value = [structure.copy()]

        is_stable, plot = calculator.check_stability(structure, potential_path)

        assert is_stable is False
