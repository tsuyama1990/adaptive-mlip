import shutil
from pathlib import Path

import pytest
import yaml

from pyacemaker.domain_models import PyAceConfig
from pyacemaker.orchestrator import Orchestrator

@pytest.fixture
def temp_workspace(tmp_path):
    """Creates a temporary workspace for integration tests."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    # Create mock pseudopotential file
    mock_pp = work_dir / "mock.upf"
    mock_pp.touch()

    return work_dir

@pytest.fixture
def minimal_config(temp_workspace):
    """Creates a minimal valid configuration for Step 1."""
    config_dict = {
        "project_name": "cycle01_test",
        "structure": {
            "elements": ["Cu"],
            "supercell_size": [2, 2, 2],
            "num_structures": 5,
            "r_cut": 2.0
        },
        "dft": {
            "code": "mock", # Use mock oracle
            "functional": "PBE",
            "kpoints_density": 0.05,
            "encut": 500,
            "pseudopotentials": {"Cu": str(temp_workspace / "mock.upf")}
        },
        "training": {
            "potential_type": "ace",
            "cutoff_radius": 5.0,
            "max_basis_size": 500,
            "max_iterations": 10,
            "pacemaker": {
                 "embedding_type": "fs",
                 "ndensity": 2,
                 "max_deg": 6,
                 "r0": 1.0,
                 "loss_kappa": 0.0,
                 "loss_l1_coeffs": 1e-8,
                 "loss_l2_coeffs": 1e-8,
                 "repulsion_sigma": 1.0,
                 "optimizer": "L-BFGS-B"
            }
        },
        "md": {
            "n_steps": 100,
            "temperature": 300,
            "pressure": 0.0,
            "timestep": 0.001
        },
        "workflow": {
            "max_iterations": 2,
            "data_dir": str(temp_workspace / "data"),
            "active_learning_dir": str(temp_workspace / "al"),
            "potentials_dir": str(temp_workspace / "potentials"),
            "state_file_path": str(temp_workspace / "workflow_state.json")
        }
    }
    return PyAceConfig(**config_dict)

def test_orchestrator_initialization(minimal_config):
    """Test Orchestrator initializes workspace correctly."""
    orch = Orchestrator(minimal_config)
    orch.initialize_workspace()

    assert Path(minimal_config.workflow.data_dir).exists()
    assert Path(minimal_config.workflow.potentials_dir).exists()
    assert Path(minimal_config.workflow.state_file_path).exists()

def test_run_step1_direct_sampling(minimal_config):
    """Test Step 1: DIRECT Sampling execution."""
    orch = Orchestrator(minimal_config)
    orch.initialize_workspace()

    # Run Step 1
    orch.run_step1()

    # Verify Output
    expected_output = Path(minimal_config.workflow.data_dir) / "step1_initial.xyz"
    assert expected_output.exists()

    # Verify Content (basic check)
    with open(expected_output, "r") as f:
        content = f.read()
        assert "Lammps data file via pyacemaker streaming" not in content # Should be extxyz format from ASE write
        assert "Properties=species:S:1:pos:R:3" in content or "Properties" in content

    from ase.io import read
    frames = read(expected_output, index=":")
    assert len(frames) == 5
    assert all(len(f) > 0 for f in frames)
