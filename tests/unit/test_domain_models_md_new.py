from pyacemaker.domain_models.md import MDConfig


def test_md_config_defaults() -> None:
    config = MDConfig(
        temperature=300.0,
        pressure=0.0,
        timestep=0.001,
        n_steps=1000
    )
    assert config.velocity_seed == 12345
    assert config.minimize_steps == 100
    assert config.minimize_max_iter == 1000
    assert config.minimize_tol == 1.0e-4
    assert config.minimize_ftol == 1.0e-6

def test_md_config_overrides() -> None:
    config = MDConfig(
        temperature=300.0,
        pressure=0.0,
        timestep=0.001,
        n_steps=1000,
        velocity_seed=54321,
        minimize_steps=500,
        minimize_tol=1e-8
    )
    assert config.velocity_seed == 54321
    assert config.minimize_steps == 500
    assert config.minimize_tol == 1e-8
