from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.main import main


def test_main_dry_run(caplog: Any, tmp_path: Path) -> None:
    config_data = """
project_name: TestProject
structure:
    elements: [Fe]
    supercell_size: [1,1,1]
dft:
    code: qe
    functional: PBE
    kpoints_density: 0.04
    encut: 500
training:
    potential_type: ace
    cutoff_radius: 5.0
    max_basis_size: 500
md:
    temperature: 300
    pressure: 0.0
    timestep: 0.001
    n_steps: 1000
workflow:
    max_iterations: 10
"""
    p = tmp_path / "config.yaml"
    p.write_text(config_data)

    with patch(
        "argparse.ArgumentParser.parse_args", return_value=MagicMock(config=str(p), dry_run=True)
    ), patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

    assert "Configuration loaded successfully" in caplog.text
    assert "Project: TestProject initialized" in caplog.text


def test_main_run(caplog: Any, tmp_path: Path) -> None:
    config_data = """
project_name: TestRun
structure:
    elements: [Al]
    supercell_size: [1,1,1]
dft:
    code: qe
    functional: PBE
    kpoints_density: 0.04
    encut: 400
training:
    potential_type: ace
    cutoff_radius: 4.0
    max_basis_size: 500
md:
    temperature: 300
    pressure: 0.0
    timestep: 0.001
    n_steps: 1000
workflow:
    max_iterations: 10
"""
    p = tmp_path / "run_config.yaml"
    p.write_text(config_data)

    with (
        patch(
            "argparse.ArgumentParser.parse_args",
            return_value=MagicMock(config=str(p), dry_run=False),
        ),
        patch("pyacemaker.orchestrator.Orchestrator.run") as mock_run,
        patch("pathlib.Path.cwd", return_value=tmp_path),
    ):
        main()
        mock_run.assert_called_once()

    assert "Configuration loaded successfully" in caplog.text


def test_main_invalid_config(caplog: Any, tmp_path: Path) -> None:
    config_data = """
project_name: BadConfig
# Missing required fields
"""
    p = tmp_path / "bad.yaml"
    p.write_text(config_data)

    with patch(
        "argparse.ArgumentParser.parse_args", return_value=MagicMock(config=str(p), dry_run=False)
    ), patch("pathlib.Path.cwd", return_value=tmp_path):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1

    assert "Fatal error during execution" in caplog.text
