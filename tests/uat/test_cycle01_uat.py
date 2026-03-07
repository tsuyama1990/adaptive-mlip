from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyacemaker.domain_models import PyAceConfig
from tests.conftest import create_test_config_dict


def test_scenario_2_1_nextgen_architecture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Simulates the core UAT scenario for v2.1.0: Zero-Shot Distillation,
    Tiered Oracle Fallback, Intelligent Extraction, and Seamless Resume.
    """
    from unittest.mock import MagicMock

    import numpy as np
    from ase import Atoms

    from pyacemaker.core.oracle import DFTManager, MACEManager, TieredOracle
    from pyacemaker.utils.extraction import extract_intelligent_cluster

    # 1. Zero-Shot Distillation Setup (Phase 1)
    # The configuration requires distillation to be enabled.
    (tmp_path / "Fe.UPF").touch()
    (tmp_path / "Mg.UPF").touch()
    (tmp_path / "O.UPF").touch()

    config_dict = create_test_config_dict(
        workflow={
            "max_iterations": 1,
            "distillation": {
                "enable": True,
                "uncertainty_threshold": 0.05,
                "sampling_structures_per_system": 100,
            },
            "loop_strategy": {
                "use_tiered_oracle": True,
                "incremental_update": True,
                "replay_buffer_size": 10,
            },
            "cutout": {
                "core_radius": 2.0,
                "buffer_radius": 1.0,
                "enable_passivation": True,
                "passivation_element": "H",
            },
        },
        dft={
            "pseudopotentials": {
                "Fe": str(tmp_path / "Fe.UPF"),
                "Mg": str(tmp_path / "Mg.UPF"),
                "O": str(tmp_path / "O.UPF"),
            }
        },
    )
    config = PyAceConfig.model_validate(config_dict)

    assert config.workflow.distillation.enable is True
    assert config.workflow.loop_strategy.use_tiered_oracle is True
    assert config.workflow.loop_strategy.incremental_update is True

    # 2. Tiered Oracle Mock (Phase 2 & 3)
    mace = MACEManager(use_mock=True)

    # We must mock DFTManager properly to avoid TypeError checks on its dependencies
    from pyacemaker.interfaces.qe_driver import QEDriver

    dft_config = config.dft
    dft_mock = DFTManager(config=dft_config, driver=MagicMock(spec=QEDriver))
    dft_mock.compute = MagicMock()  # Mock the inner compute

    # Mock MACE to return high uncertainty so it triggers DFT fallback
    def mock_infer(atoms: Atoms) -> Atoms:
        res = atoms.copy()  # type: ignore[no-untyped-call]
        res.info["energy"] = -10.0
        res.arrays["forces"] = np.zeros((len(res), 3))
        res.arrays["c_gamma"] = np.ones(len(res)) * 0.1  # High uncertainty > 0.05
        return res

    mace._infer = mock_infer  # type: ignore[method-assign]

    tiered = TieredOracle(mace_manager=mace, dft_manager=dft_mock, uncertainty_threshold=0.05)

    test_atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    list(tiered.compute(iter([test_atoms])))

    # Verify DFT fallback occurred
    dft_mock.compute.assert_called_once()

    # 3. Intelligent Cluster Extraction (Phase 3)
    # Simulate a halt with an epicentre
    halt_atoms = Atoms("MgO", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[10, 10, 10], pbc=True)
    cutout_config = config.workflow.cutout
    cluster = extract_intelligent_cluster(halt_atoms, target_atoms=[0], config=cutout_config)

    # Assert cluster has proper arrays and passivation
    assert "force_weight" in cluster.arrays

    # 4. Seamless Resume (Phase 4)
    engine_mock = MagicMock()
    engine_mock.run.return_value = MagicMock(halted=False, n_steps=1000)

    # Simulate resuming MD
    engine_mock.run(halt_atoms, "dummy.yace", resume_from_step=500)

    engine_mock.run.assert_called_with(halt_atoms, "dummy.yace", resume_from_step=500)


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
    potential_type: pace
    cutoff_radius: 5.0
    max_basis_size: 500
md:
    temperature: 300.0
    pressure: 0.0
    timestep: 0.001
    n_steps: 1000
workflow:
    max_iterations: 10
    state_file_path: uat_state.json
    otf:
        fix_halt: true
        check_interval: 50
        local_n_candidates: 20
        local_n_select: 5
        max_retries: 3
"""
    path.write_text(config_content)

    # 2. Action
    from pyacemaker.main import main

    with patch(
        "pyacemaker.main.CLIParser.parse",
        return_value=MagicMock(config=str(config_file), dry_run=True, scenario=None),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_scenario_01_02_guardrails_check_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_scenario_01_02_guardrails_check_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
