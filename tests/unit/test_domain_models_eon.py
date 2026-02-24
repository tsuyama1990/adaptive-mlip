import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.eon import EONConfig


def test_eon_config_valid() -> None:
    """Test creating a valid EON configuration."""
    config = EONConfig(
        enabled=True,
        temperature=300.0,
        search_method="akmc",
        num_replicas=4,
    )
    assert config.enabled is True
    assert config.temperature == 300.0
    assert config.search_method == "akmc"
    assert config.num_replicas == 4


def test_eon_config_defaults() -> None:
    """Test defaults for EON configuration."""
    config = EONConfig(temperature=500.0)
    assert config.enabled is False
    assert config.search_method == "akmc"
    assert config.potential_path is None
    assert config.num_replicas == 1
    assert config.confidence == 0.99


def test_eon_config_invalid_temp() -> None:
    """Test invalid temperature raises ValidationError."""
    with pytest.raises(ValidationError):
        EONConfig(temperature=-10.0)


def test_eon_config_invalid_confidence() -> None:
    """Test invalid confidence raises ValidationError."""
    with pytest.raises(ValidationError):
        EONConfig(temperature=300.0, confidence=1.5)


def test_eon_config_extra_field() -> None:
    """Test extra fields are forbidden."""
    with pytest.raises(ValidationError):
        EONConfig(temperature=300.0, unknown_field="invalid")
