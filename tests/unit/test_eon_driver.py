import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.eon import EONConfig
from pyacemaker.interfaces.eon_driver import EONWrapper


@pytest.fixture
def mock_subprocess() -> MagicMock:
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run


@pytest.fixture
def mock_path_exists() -> MagicMock:
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists


def test_eon_wrapper_init() -> None:
    config = EONConfig(potential_path=Path("pot.yace"))
    driver = EONWrapper(config)
    assert driver.config == config


def test_generate_config(tmp_path: Path) -> None:
    config = EONConfig(potential_path=Path("pot.yace"), temperature=500.0)
    driver = EONWrapper(config)

    # Mock writing config
    config_path = tmp_path / "config.ini"
    driver.generate_config(config_path)

    # Check if file was written
    assert config_path.exists()
    content = config_path.read_text()
    assert "temperature = 500.0" in content
    assert "potentials_path = pot.yace" in content


def test_run_success(mock_subprocess: MagicMock, tmp_path: Path) -> None:
    config = EONConfig(potential_path=Path("pot.yace"))
    driver = EONWrapper(config)

    # Mock existence of required files
    with patch("pathlib.Path.exists", return_value=True):
        driver.run(working_dir=tmp_path)

    # Check if subprocess was called correctly
    mock_subprocess.assert_called_once()
    args, kwargs = mock_subprocess.call_args
    # args[0] is the command list
    assert args[0] == ["eonclient"]
    assert kwargs.get("cwd") == tmp_path  # cwd can be Path object if subprocess supports it, checking implementation


def test_run_failure(mock_subprocess: MagicMock, tmp_path: Path) -> None:
    # Simulate CalledProcessError
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, ["eonclient"], stderr="Error occurred")

    config = EONConfig(potential_path=Path("pot.yace"))
    driver = EONWrapper(config)

    with (
        pytest.raises(RuntimeError, match="EON execution failed"),
        patch("pathlib.Path.exists", return_value=True),
    ):
        driver.run(working_dir=tmp_path)


def test_parse_results(tmp_path: Path) -> None:
    config = EONConfig(potential_path=Path("pot.yace"))
    driver = EONWrapper(config)

    # Create fake results
    (tmp_path / "dynamics.txt").write_text("Step 1: 0.5 eV barrier\n")
    (tmp_path / "processtable.dat").write_text("Process 1: Barrier 0.5 eV\n")

    results = driver.parse_results(tmp_path)
    assert "dynamics" in results
    assert "processtable" in results
    assert results["dynamics"] == "Step 1: 0.5 eV barrier\n"
