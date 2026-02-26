from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models.defaults import (
    LOG_CONFIG_LOADED,
    LOG_DRY_RUN_COMPLETE,
    LOG_PROJECT_INIT,
)
from pyacemaker.main import main
from tests.conftest import create_dummy_pseudopotentials


def test_main_dry_run(caplog: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Al"])

    config_data = """
project_name: TestProject
structure:
    elements: [Al]
    supercell_size: [1,1,1]
dft:
    code: qe
    functional: PBE
    kpoints_density: 0.04
    encut: 400
    pseudopotentials:
        Al: Al.UPF
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
    p = tmp_path / "config.yaml"
    p.write_text(config_data)

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(config=str(p), dry_run=True, scenario=None),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0

    assert LOG_CONFIG_LOADED in caplog.text
    assert LOG_PROJECT_INIT.format(project_name="TestProject") in caplog.text
    assert LOG_DRY_RUN_COMPLETE in caplog.text


def test_main_file_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(config="non_existent.yaml", dry_run=False, scenario=None),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1


def test_main_run(caplog: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    create_dummy_pseudopotentials(tmp_path, ["Al"])

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
    pseudopotentials:
        Al: Al.UPF
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
            return_value=MagicMock(config=str(p), dry_run=False, scenario=None),
        ),
        patch("pyacemaker.orchestrator.Orchestrator.run") as mock_run,
    ):
        main()

    mock_run.assert_called_once()
    assert LOG_CONFIG_LOADED in caplog.text
