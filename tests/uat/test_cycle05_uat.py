from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.domain_models.compiler import SemanticCompiler
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.scenario import IntentRequest


def test_uat_05_01_soft_start_thermalization(tmp_path: Path) -> None:
    """
    Scenario ID: UAT-05-01
    Objective: Verify soft start thermalization logic is correctly inserted.
    """
    config = MDConfig(
        n_steps=2000,
        fix_halt=False,
        temperature=300.0,
        pressure=1.0,
        timestep=0.001,
        soft_start_steps=150,
        soft_start_langevin_damp=0.2,
    )
    engine = LammpsEngine(config)

    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Capture script generation output
    script_content = []

    with patch("pyacemaker.core.engine.LammpsDriver") as mock_driver:
        driver_instance = mock_driver.return_value
        driver_instance.extract_variable.return_value = 0.0
        driver_instance.get_forces.return_value = np.zeros((1, 3))
        driver_instance.get_stress.return_value = np.zeros(6)

        def capture_run(path: str) -> None:
            script_content.append(Path(path).read_text())

        driver_instance.run_file.side_effect = capture_run

        with (
            patch("pyacemaker.core.engine.Path.exists", return_value=True),
            patch("pyacemaker.core.engine.Path.stat") as mock_stat,
            patch("pyacemaker.core.validator.Path.is_file", return_value=True),
            patch("pyacemaker.core.validator.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.utils.path.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.core.io_manager.validate_path_safe", return_value=pot_path),
            patch("pathlib.Path.is_symlink", return_value=False),
            patch("pyacemaker.core.lammps_generator.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.core.engine.LammpsEngine._validate_script_content"),
            patch("pyacemaker.core.lammps_generator.Path.is_relative_to", return_value=True),
            patch("pyacemaker.core.engine.LammpsEngine._execute_simulation") as mock_exec,
        ):

            def side_effect_exec(driver: Any, script_path: Path) -> None:
                script_content.append(script_path.read_text())

            mock_exec.side_effect = side_effect_exec
            mock_stat.return_value.st_size = 100

            # Resume from step 1500 with override
            engine.run(atoms, pot_path, resume_from_step=1500, override_n_steps=400)

    assert len(script_content) == 1
    script = script_content[0]

    # Verify seamless resume parameters
    assert "read_restart" in script
    assert "unfix main_ensemble" in script
    assert "fix soft_nve all nve" in script
    assert "fix soft_langevin all langevin 300.0 300.0 0.2" in script
    assert "run 150" in script  # soft start steps
    assert "unfix soft_nve" in script
    assert "unfix soft_langevin" in script
    assert "fix main_ensemble all npt temp 300.0 300.0" in script
    assert "run 250" in script  # overridden steps (400) minus soft start steps (150)


def test_uat_05_a_heuristics_translation_speed_priority() -> None:
    """
    SCENARIO-05-A [Priority: High] - Successful Heuristics Translation (Speed Priority)
    """
    payload = {
        "accuracy_speed_slider": 2,
        "target_material": "Pt",
        "nodes": [
            {
                "id": "node1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Pt",
                    "lattice_constant": 3.92,
                },
            },
            {
                "id": "node2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
            {
                "id": "node3",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"},
            },
        ],
        "edges": [{"source": "node1", "target": "node3"}, {"source": "node3", "target": "node2"}],
    }

    intent = IntentRequest(**payload)
    cfg = SemanticCompiler.compile(intent)

    assert cfg.md.uncertainty_threshold > 0.1
    assert cfg.md.timestep > 0.0015
    assert cfg.md.check_interval > 10

    assert cfg.dft.smearing_type == "mv"
    assert cfg.dft.smearing_width == 0.02


def test_uat_05_b_accurate_mathematical_slider_evaluation() -> None:
    """
    SCENARIO-05-B [Priority: High] - Accurate Mathematical Slider Evaluation (Accuracy Priority)
    """
    payload = {
        "accuracy_speed_slider": 9,
        "target_material": "Fe",
        "nodes": [
            {
                "id": "node1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "Fe",
                    "lattice_constant": 2.87,
                },
            },
            {
                "id": "node2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
            {
                "id": "node3",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"},
            },
        ],
        "edges": [{"source": "node1", "target": "node3"}, {"source": "node3", "target": "node2"}],
    }

    intent = IntentRequest(**payload)
    cfg = SemanticCompiler.compile(intent)

    assert cfg.md.uncertainty_threshold < 0.05
    assert cfg.md.timestep <= 0.0006
    assert cfg.md.check_interval <= 2
    assert cfg.dft.encut > 60.0


def test_uat_05_c_preservation_of_manual_overrides() -> None:
    """
    SCENARIO-05-C [Priority: Medium] - Preservation of Manual Overrides (Expert Mode)
    """
    payload = {
        "accuracy_speed_slider": 5,
        "target_material": "W",
        "advanced_settings": {"ecutwfc": 60.0, "learning_rate": 0.0123},
        "nodes": [
            {
                "id": "node1",
                "type": "INITIAL_STRUCTURE",
                "data": {
                    "type": "INITIAL_STRUCTURE",
                    "chemical_symbol": "W",
                    "lattice_constant": 3.16,
                },
            },
            {
                "id": "node2",
                "type": "ACTIVE_LEARNING_LOOP",
                "data": {"type": "ACTIVE_LEARNING_LOOP"},
            },
            {
                "id": "node3",
                "type": "MACE_TRAINING",
                "data": {"type": "MACE_TRAINING"},
            },
        ],
        "edges": [{"source": "node1", "target": "node3"}, {"source": "node3", "target": "node2"}],
    }

    intent = IntentRequest(**payload)
    cfg = SemanticCompiler.compile(intent)

    # Overrides
    assert cfg.dft.encut == 60.0
    assert cfg.training.pacemaker.learning_rate == 0.0123

    # Heuristics maintained for untouched
    assert cfg.md.timestep > 0.0001
