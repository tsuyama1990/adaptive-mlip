from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper


@pytest.fixture
def eon_config() -> EONConfig:
    return EONConfig(temperature=300.0, search_method="akmc", enabled=True)


def test_eon_wrapper_init(eon_config: EONConfig) -> None:
    wrapper = EONWrapper(eon_config)
    assert wrapper.config == eon_config


@patch("shutil.which")
@patch("subprocess.run")
def test_eon_run(
    mock_run: MagicMock, mock_which: MagicMock, eon_config: EONConfig, tmp_path: Path
) -> None:
    wrapper = EONWrapper(eon_config)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    # Mock successful execution
    mock_which.return_value = "/usr/bin/eonclient"
    mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

    # We expect run to create config.ini and maybe call eonclient
    wrapper.run(potential_path, work_dir=tmp_path)

    # Check if config.ini was created
    config_ini = tmp_path / "config.ini"
    assert config_ini.exists()
    content = config_ini.read_text()
    assert "temperature = 300.0" in content

    # Check if subprocess was called
    mock_run.assert_called()

    # Check command
    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "/usr/bin/eonclient"


def test_generate_pace_driver(eon_config: EONConfig, tmp_path: Path) -> None:
    wrapper = EONWrapper(eon_config)
    potential_path = tmp_path.resolve() / "pot.yace" # Resolve path to be safe
    potential_path.touch()
    driver_path = wrapper._generate_pace_driver(tmp_path, potential_path)

    assert driver_path.exists()
    content = driver_path.read_text()
    # It should reference the potential
    assert str(potential_path) in content
    # It should be executable
    assert "python" in content or "#!/usr/bin/env python" in content


def test_parse_results(eon_config: EONConfig, tmp_path: Path) -> None:
    wrapper = EONWrapper(eon_config)

    # Create dummy output files
    (tmp_path / "processtable.dat").write_text("0 1.5 2.5 1.0\n1 2.0 3.0 1.0")

    results = wrapper.parse_results(tmp_path)
    assert len(results) == 2
    assert results[0]["barrier"] == 1.5
