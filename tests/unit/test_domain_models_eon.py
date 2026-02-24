from pathlib import Path

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.eon import EONConfig


def test_eon_config_valid() -> None:
    config = EONConfig(
        potential_path=Path("pot.yace"),
        temperature=300.0,
        akmc_steps=100,
        supercell=[2, 2, 2],
    )
    assert config.potential_path == Path("pot.yace")
    assert config.temperature == 300.0
    assert config.akmc_steps == 100
    assert config.supercell == [2, 2, 2]


def test_eon_config_defaults() -> None:
    config = EONConfig(potential_path=Path("pot.yace"))
    assert config.temperature == 300.0
    assert config.akmc_steps == 100
    assert config.supercell == [1, 1, 1]
    assert config.eon_executable == "eonclient"
    assert not config.enabled


def test_eon_config_invalid_temp() -> None:
    with pytest.raises(ValidationError):
        EONConfig(potential_path=Path("pot.yace"), temperature=-10.0)


def test_eon_config_invalid_steps() -> None:
    with pytest.raises(ValidationError):
        EONConfig(potential_path=Path("pot.yace"), akmc_steps=0)


def test_eon_config_invalid_supercell() -> None:
    with pytest.raises(ValidationError):
        EONConfig(potential_path=Path("pot.yace"), supercell=[1, 1])  # Too short

    with pytest.raises(ValidationError):
        EONConfig(potential_path=Path("pot.yace"), supercell=[1, 1, 1, 1])  # Too long
