from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models import PyAceConfig
from tests.conftest import create_test_config_dict


def test_scenario_01_01_hello_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario 01-01: "Hello Config"
    Objective: Verify that the system can load a configuration file and initialize.
    """
    # 1. Preparation
    monkeypatch.chdir(tmp_path)
    # Create dummy pseudo files
    (tmp_path / "H.UPF").touch()
    (tmp_path / "O.UPF").touch()

    config_file = tmp_path / "config.yaml"
    # Create valid config manually as before
    path = config_file
    config_content = """
project_name: UAT_Project
structure:
    elements: [H, O]
    supercell_size: [1, 1, 1]
dft:
    code: qe
    functional: PBE
    kpoints_density: 0.04
    encut: 500.0
    pseudopotentials:
        H: H.UPF
        O: O.UPF
training:
    potential_type: ace
    cutoff_radius: 5.0
    max_basis_size: 500
md:
    temperature: 300.0
    pressure: 0.0
    timestep: 0.001
    n_steps: 1000
    uncertainty_threshold: 0.1
    check_interval: 50
workflow:
    max_iterations: 10
    state_file_path: uat_state.json
"""
    path.write_text(config_content)

    # 2. Action
    from pyacemaker.main import main

    # Use 'init' command which is the Cycle 01 equivalent of "Hello Config"
    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(command="init", config=config_file),
    ):
        # Should execute successfully without SystemExit
        main()


def test_scenario_01_02_guardrails_check_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario 01-02: "Guardrails Check" (Temperature)
    Objective: Verify that the system rejects invalid physical parameters (negative temperature).
    """
    # 1. Preparation
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    # We use Pydantic model directly validation
    config_dict = create_test_config_dict(md={"temperature": -50.0})

    # 2. Action & 3. Expectation
    # Pydantic raises ValidationError
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PyAceConfig(**config_dict)


def test_scenario_01_02_guardrails_check_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Scenario 01-02: "Guardrails Check" (Cutoff)
    Objective: Verify that the system rejects invalid physical parameters (negative cutoff).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Fe.UPF").touch()

    config_dict = create_test_config_dict(training={"cutoff_radius": -1.0})
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PyAceConfig(**config_dict)
