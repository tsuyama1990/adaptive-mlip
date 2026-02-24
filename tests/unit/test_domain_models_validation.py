import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.validation import (
    ElasticConfig,
    PhononConfig,
    ValidationConfig,
)


def test_phonon_config_defaults() -> None:
    config = PhononConfig()
    assert config.supercell_size == (2, 2, 2)
    assert config.displacement == 0.01
    assert config.symprec == 1e-5


def test_phonon_config_validation() -> None:
    with pytest.raises(ValidationError):
        PhononConfig(displacement=-0.1)
    with pytest.raises(ValidationError):
        PhononConfig(symprec=-1e-5)


def test_elastic_config_defaults() -> None:
    config = ElasticConfig()
    assert config.strain_magnitude == 0.01


def test_elastic_config_validation() -> None:
    with pytest.raises(ValidationError):
        ElasticConfig(strain_magnitude=0.0)
    with pytest.raises(ValidationError):
        ElasticConfig(strain_magnitude=-0.01)


def test_validation_config_defaults() -> None:
    config = ValidationConfig()
    assert config.enabled is True
    assert isinstance(config.phonon, PhononConfig)
    assert isinstance(config.elastic, ElasticConfig)


def test_validation_config_custom() -> None:
    config = ValidationConfig(
        enabled=False,
        phonon=PhononConfig(supercell_size=(3, 3, 3)),
        elastic=ElasticConfig(strain_magnitude=0.02)
    )
    assert config.enabled is False
    assert config.phonon.supercell_size == (3, 3, 3)
    assert config.elastic.strain_magnitude == 0.02
