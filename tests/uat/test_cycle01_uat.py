from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.domain_models.data import AtomStructure
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

def test_scenario_01_03_mock_oracle_execution(tmp_path: Path) -> None:
    """
    Scenario 01-03: "Mock Oracle Execution"
    Objective: Verify that Mock Oracle can compute energy/forces for a structure
    via the AtomStructure interface.
    """
    # 1. Setup
    # Create a simple structure
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)
    input_structure = AtomStructure(atoms=atoms)

    # 2. Mock Oracle Instantiation
    # We use the MockOracle from pyacemaker.modules.mock_oracle
    # (Assuming it will be implemented in Logic Implementation phase)
    # For now, we import it inside a try block or expect it to be available
    # if we followed TDD order strictly (tests first, then code).
    # Since we are writing tests *before* code, this import might fail if run now.
    # But this is the test definition.

    try:
        from pyacemaker.modules.mock_oracle import MockOracle
    except ImportError:
        pytest.skip("MockOracle not implemented yet")

    oracle = MockOracle()

    # 3. Execution
    # compute returns Iterator[AtomStructure]
    results = oracle.compute(iter([input_structure]))
    result_structure = next(results)

    # 4. Assertions
    assert isinstance(result_structure, AtomStructure)
    assert result_structure.energy is not None
    assert result_structure.forces is not None
    assert result_structure.stress is not None

    # Check physical plausibility (finite values)
    import numpy as np
    assert np.isfinite(result_structure.energy)
    assert np.all(np.isfinite(result_structure.forces))
