
import pytest
import yaml

from pyacemaker.domain_models.config import PyAceConfig
from pyacemaker.orchestrator import Orchestrator


@pytest.fixture
def minimal_config(tmp_path):
    # Create dummy UPF
    upf_path = tmp_path / "Si.upf"
    upf_path.write_text("<UPF version=\"2.0.1\">\nPP_HEADER\n</UPF>")

    config_dict = {
        "project_name": "TestProject",
        "structure": {
            "elements": ["Si"],
            "supercell_size": [1, 1, 1],
            "num_structures": 5,
            "r_cut": 2.0
        },
        "workflow": {
            "max_iterations": 1,
            "data_dir": str(tmp_path / "data"),
            "potentials_dir": str(tmp_path / "potentials"),
             "active_learning_dir": str(tmp_path / "al"),
             "state_file_path": str(tmp_path / "state.json")
        },
        "logging": {"level": "DEBUG"},
        # Minimal dummy configs to satisfy Pydantic
        "dft": {
            "code": "vasp",
            "functional": "pbe",
            "kpoints_density": 0.1,
            "encut": 500.0,
            "pseudopotentials": {"Si": str(upf_path)},
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 5.0,
            "max_basis_size": 100
        },
        "md": {
            "temperature": 300.0,
            "pressure": 0.0,
            "timestep": 0.001,
            "n_steps": 100,
            "thermo_freq": 10,
            "dump_freq": 10
        },
        "validation": {},
        "eon": None,
        "scenario": None
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)
    return config_path

def test_cycle01_execution(minimal_config, tmp_path):
    # This test will fail until Orchestrator is updated
    with minimal_config.open() as f:
        config_data = yaml.safe_load(f)
    config = PyAceConfig(**config_data)

    orch = Orchestrator(config=config)
    orch.initialize_workspace()

    # Run Step 1
    orch.run_step1()

    # Verify output
    expected_output = tmp_path / "data/step1_initial.xyz"
    assert expected_output.exists()

    # Verify content (using ASE or simple line count)
    with expected_output.open() as f:
        lines = f.readlines()
        assert len(lines) > 0

    # Verify state
    state_file = tmp_path / "state.json"
    assert state_file.exists()
