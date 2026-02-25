from pathlib import Path

import pytest

from pyacemaker.core.exceptions import EONError
from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper
from tests.unit.mock_process import MockProcessRunner


@pytest.fixture
def valid_eon_config(tmp_path: Path) -> EONConfig:
    pot = tmp_path / "pot.yace"
    pot.touch()
    return EONConfig(potential_path=pot, temperature=500.0)

def test_eon_wrapper_init(valid_eon_config: EONConfig) -> None:
    driver = EONWrapper(valid_eon_config)
    assert driver.config == valid_eon_config


def test_generate_config(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    driver = EONWrapper(valid_eon_config)

    config_path = tmp_path / "config.ini"
    driver.generate_config(config_path)

    assert config_path.exists()
    content = config_path.read_text()
    assert "temperature = 500.0" in content
    # Updated to expect script interface per spec
    assert "potential = script" in content
    assert "script_path = ./pace_driver.py" in content


def test_generate_driver_script(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    driver = EONWrapper(valid_eon_config)
    script_path = tmp_path / "pace_driver.py"

    driver.generate_driver_script(script_path)

    assert script_path.exists()
    content = script_path.read_text()
    assert "#!/usr/bin/env python3" in content
    assert "run_driver()" in content
    # Ensure no secrets (potential path) are hardcoded in the script content
    assert str(valid_eon_config.potential_path) not in content


def test_run_success(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    runner = MockProcessRunner()
    driver = EONWrapper(valid_eon_config, runner=runner)

    driver.run(working_dir=tmp_path)

    assert len(runner.commands) == 1
    cmd, cwd = runner.commands[0]
    assert cmd == ["eonclient"]
    assert cwd == tmp_path


def test_run_failure(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    runner = MockProcessRunner(returncode=1, stderr="Error occurred")
    driver = EONWrapper(valid_eon_config, runner=runner)

    with pytest.raises(EONError, match="EON execution failed"):
        driver.run(working_dir=tmp_path)


def test_parse_results(valid_eon_config: EONConfig, tmp_path: Path) -> None:
    driver = EONWrapper(valid_eon_config)

    (tmp_path / "dynamics.txt").write_text("Step 1: 0.5 eV barrier\n")
    (tmp_path / "processtable.dat").write_text("Process 1: Barrier 0.5 eV\n")

    results = driver.parse_results(tmp_path)
    assert "dynamics" in results
    assert "processtable" in results
    assert results["dynamics"] == "Step 1: 0.5 eV barrier\n"
