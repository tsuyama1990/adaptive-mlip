from unittest.mock import MagicMock

import numpy as np
import pytest

from pyacemaker.core.base import BaseEngine
from pyacemaker.utils.elastic import ElasticCalculator


class TestElasticCalculator:
    @pytest.fixture
    def mock_engine(self):
        return MagicMock(spec=BaseEngine)

    @pytest.fixture
    def calculator(self, mock_engine):
        return ElasticCalculator(engine=mock_engine, strain=0.01, steps=5)

    def test_check_born_stability_cubic_stable(self):
        c_ij = {"C11": 200.0, "C12": 100.0, "C44": 100.0}
        # C11 - C12 > 0 -> 100 > 0 (OK)
        # C11 + 2C12 > 0 -> 400 > 0 (OK)
        # C44 > 0 -> 100 > 0 (OK)
        assert ElasticCalculator.check_stability_criteria(c_ij, "cubic") is True

    def test_check_born_stability_cubic_unstable(self):
        c_ij = {"C11": 50.0, "C12": 100.0, "C44": 100.0}
        # C11 - C12 = -50 < 0 (Fail)
        assert ElasticCalculator.check_stability_criteria(c_ij, "cubic") is False

    def test_calculate_bulk_modulus(self):
        c_ij = {"C11": 200.0, "C12": 100.0, "C44": 100.0}
        # For cubic: B = (C11 + 2C12) / 3 = 400 / 3 = 133.33
        B = ElasticCalculator.calculate_bulk_modulus(c_ij, "cubic")
        assert np.isclose(B, 133.333, atol=0.01)

    def test_calculate_properties(self, calculator, mock_engine):
        from pathlib import Path

        import numpy as np
        from ase import Atoms

        # Create a simple cubic structure
        atoms = Atoms("Fe", cell=[2.87, 2.87, 2.87], pbc=True)
        potential_path = Path("dummy.yace")

        # We need to mock _get_stress to return different stress values based on the strain applied.
        # It's called for C11/C12 (steps) and then C44 (steps)
        # So we expect 2 * steps calls to _get_stress

        strains = np.linspace(-calculator.strain, calculator.strain, calculator.steps)

        # Assume C11 = 200 GPa, C12 = 100 GPa, C44 = 100 GPa
        # Stress in Bar = Stress in GPa * 1e4
        C11_bar = 200 * 1e4
        C12_bar = 100 * 1e4
        C44_bar = 100 * 1e4

        call_count = 0

        def mock_get_stress(atoms, path):
            nonlocal call_count
            stress = np.zeros(6)
            if call_count < calculator.steps:
                # Normal strain in xx direction
                idx = call_count
                eps = strains[idx]
                stress[0] = C11_bar * eps
                stress[1] = C12_bar * eps
            else:
                # Shear strain
                idx = call_count - calculator.steps
                eps = strains[idx]
                stress[5] = C44_bar * eps

            call_count += 1
            # Mock the MDSimulationResult to return this stress
            result = MagicMock()
            result.stress = stress.tolist()
            return result

        mock_engine.compute_static_properties.side_effect = mock_get_stress

        is_stable, c_ij, B, plot_b64 = calculator.calculate_properties(atoms, potential_path)

        # Verify calls
        assert mock_engine.compute_static_properties.call_count == 2 * calculator.steps

        # Verify outputs
        assert is_stable is True
        assert np.isclose(c_ij["C11"], 200.0, rtol=1e-3)
        assert np.isclose(c_ij["C12"], 100.0, rtol=1e-3)
        assert np.isclose(c_ij["C44"], 100.0, rtol=1e-3)
        assert np.isclose(B, (200.0 + 2 * 100.0) / 3.0, rtol=1e-3)
        assert len(plot_b64) > 100

    def test_calculate_bulk_modulus_non_cubic(self):
        c_ij = {"C11": 200.0, "C12": 100.0, "C44": 100.0}
        B = ElasticCalculator.calculate_bulk_modulus(c_ij, "tetragonal")
        assert B == 0.0

    def test_check_born_stability_non_cubic(self):
        c_ij = {"C11": 200.0, "C12": 100.0, "C44": 100.0}
        assert ElasticCalculator.check_stability_criteria(c_ij, "tetragonal") is False
