from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.utils.io import write_lammps_streaming
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

    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(config=str(config_file), dry_run=True, scenario=None),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


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

def test_streaming_write_large_structure(tmp_path: Path) -> None:
    """
    Verify streaming writer for large structures.
    Uses a generator or small structure to simulate the path.
    """
    from ase.build import bulk

    # Ensure orthogonal box for streaming
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms = atoms.repeat((2, 2, 2))

    output_file = tmp_path / "stream_test.lmp"

    with output_file.open("w") as f:
        write_lammps_streaming(f, atoms, ["Cu"])

    assert output_file.exists()
    content = output_file.read_text()
    assert "atoms" in content
    assert str(len(atoms)) in content

def test_streaming_write_failure(tmp_path: Path) -> None:
    """
    Verify failure handling for streaming writer (e.g. invalid object).
    """
    class BadObj:
        pass

    output_file = tmp_path / "fail_test.lmp"

    # Expect TypeError because BadObj has no len()
    with pytest.raises((AttributeError, TypeError)), output_file.open("w") as f:
        write_lammps_streaming(f, BadObj(), ["Cu"]) # type: ignore
