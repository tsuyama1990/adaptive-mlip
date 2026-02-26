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

    def test_calculate_voigt_reuss_hill(self):
        # Test average moduli
        pass
