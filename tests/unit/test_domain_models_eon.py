import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.eon import EONConfig


def test_eon_config_valid():
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        config = EONConfig(
            potential_path=path,
            temperature=300.0,
            akmc_steps=100,
            random_seed=12345,
            otf_threshold=0.1,
        )
        assert config.potential_path == path
        assert config.temperature == 300.0
        assert config.otf_threshold == 0.1
        assert config.random_seed == 12345


def test_eon_config_defaults():
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        config = EONConfig(potential_path=path)
        assert config.otf_threshold == 0.05
        assert config.eon_executable == "eonclient"
        assert config.supercell == [1, 1, 1]


def test_eon_config_invalid_potential_path():
    with pytest.raises(ValidationError) as excinfo:
        EONConfig(potential_path=Path("non_existent_file.yace"))
    assert "Potential file does not exist" in str(excinfo.value)


def test_eon_config_invalid_temperature():
    with tempfile.NamedTemporaryFile(suffix=".yace") as tmp:
        path = Path(tmp.name)
        with pytest.raises(ValidationError):
            EONConfig(potential_path=path, temperature=-10.0)
