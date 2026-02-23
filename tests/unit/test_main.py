from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.main import main


def test_main_dry_run(capsys: Any, tmp_path: Path) -> None:
    config_data = """
project_name: TestProject
structure:
    elements: [Fe]
    supercell_size: [1,1,1]
dft:
    encut: 500
training:
    cutoff_radius: 5.0
md:
    temperature: 300
workflow:
    max_iterations: 10
"""
    p = tmp_path / "config.yaml"
    p.write_text(config_data)

    with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(config=str(p), dry_run=True)):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0

    captured = capsys.readouterr()
    assert "Configuration loaded successfully" in captured.out
    assert "Project: TestProject initialized" in captured.out
    assert "Dry run complete" in captured.out

def test_main_run(capsys: Any, tmp_path: Path) -> None:
    config_data = """
project_name: TestRun
structure:
    elements: [Al]
    supercell_size: [1,1,1]
dft:
    encut: 400
training:
    cutoff_radius: 4.0
md:
    temperature: 300
workflow:
    max_iterations: 10
"""
    p = tmp_path / "run_config.yaml"
    p.write_text(config_data)

    with (
        patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(config=str(p), dry_run=False)),
        patch("pyacemaker.orchestrator.Orchestrator.run") as mock_run
    ):
        main()
        mock_run.assert_called_once()

    captured = capsys.readouterr()
    assert "Configuration loaded successfully" in captured.out

def test_main_invalid_config(capsys: Any, tmp_path: Path) -> None:
    config_data = """
project_name: BadConfig
# Missing required fields
"""
    p = tmp_path / "bad.yaml"
    p.write_text(config_data)

    with patch("argparse.ArgumentParser.parse_args", return_value=MagicMock(config=str(p), dry_run=False)):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 1

    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "validation error" in captured.err
