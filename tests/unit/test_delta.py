from pyacemaker.utils.delta import compute_zbl_energy, get_lj_params


def test_get_lj_params_known_element() -> None:
    # Test for Fe (Iron)
    params = get_lj_params("Fe")
    assert "epsilon" in params
    assert "sigma" in params
    # Values might vary depending on source, but should be positive
    assert params["epsilon"] > 0
    assert params["sigma"] > 0


def test_get_lj_params_unknown_element_fallback() -> None:
    # Should return generic parameters or raise error depending on implementation
    # Assuming it returns a default or calculates from atomic number
    params = get_lj_params("X")
    assert params["epsilon"] > 0


def test_zbl_energy_repulsive() -> None:
    # ZBL potential is purely repulsive
    # Energy should decrease as distance increases
    r_small = 0.5
    r_large = 2.0

    e_small = compute_zbl_energy("Fe", "Fe", r_small)
    e_large = compute_zbl_energy("Fe", "Fe", r_large)

    assert e_small > e_large
    assert e_small > 0


def test_zbl_energy_zero_at_cutoff() -> None:
    # ZBL is usually short range, but check behavior at large R
    r_very_large = 10.0
    e = compute_zbl_energy("H", "H", r_very_large)
    assert e < 1e-5  # Should be very small
