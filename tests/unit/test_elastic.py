from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.utils.elastic import ElasticCalculator


@pytest.fixture
def mock_atoms() -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]], cell=[5, 5, 5], pbc=True)

def test_elastic_calculator_init(mock_atoms: Atoms) -> None:
    calc = ElasticCalculator(mock_atoms, strain=0.01)
    assert calc.strain == 0.01
    assert calc.atoms == mock_atoms

def test_calculate_stress(mock_atoms: Atoms) -> None:
    # This test verifies that we can calculate stress using a provided force/stress calculator
    calc = ElasticCalculator(mock_atoms, strain=0.01)

    mock_calc_fn = MagicMock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # We need to mock _apply_strain to avoid actual ASE operations that might be slow/complex
    # But wait, apply_strain is pure geometry, it's fine.
    # The issue is _compute_stress_tensor is just a helper for testing in my original thought,
    # but actual implementation calls stress_fn.

    # So we just run it and check results if we mock the stress function to return consistent values
    # If stress is constant (identity), then delta stress is 0, so Cij is 0.

    Cij, B, G = calc.calculate(mock_calc_fn)

    # If stress is always identity, delta_stress = 0, so Cij = 0
    assert B == 0.0
    assert G == 0.0
    assert Cij[0][0] == 0.0

def test_check_stability_stable() -> None:
    # Simple isotropic stability check: B > 0, G > 0
    is_stable = ElasticCalculator.check_stability(bulk_modulus=100.0, shear_modulus=50.0)
    assert is_stable is True

def test_check_stability_unstable_bulk() -> None:
    is_stable = ElasticCalculator.check_stability(bulk_modulus=-10.0, shear_modulus=50.0)
    assert is_stable is False

def test_check_stability_unstable_shear() -> None:
    is_stable = ElasticCalculator.check_stability(bulk_modulus=100.0, shear_modulus=-5.0)
    assert is_stable is False
