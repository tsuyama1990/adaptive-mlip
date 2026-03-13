import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from pyacemaker.core.engine import LammpsEngine, TwoTierEvaluator
from pyacemaker.core.exceptions import MDHaltInterrupt
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.workflow import ActiveLearningThresholds, CutoutConfig
from pyacemaker.utils.extraction import extract_intelligent_cluster


def test_uat_03_01_two_tier_evaluator() -> None:
    """
    Scenario ID: UAT-03-01
    Objective: Verify TwoTierEvaluator thermal noise filtering.
    Behavior 01: The Two-Tier Evaluator must ignore transient thermal noise spikes.
    """
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, smooth_steps=3, threshold_add_train=0.02
    )
    evaluator = TwoTierEvaluator(thresholds)

    mock_lmp = types.SimpleNamespace()

    # Step 1: Spike (0.06 > 0.05)
    mock_lmp.extract_variable = lambda name, *args: 0.06 if name == "max_g" else 0.0
    evaluator(mock_lmp)
    assert evaluator.consecutive_exceedances == 1

    # Step 2: Drop (0.02 <= 0.05) - thermal noise ignored
    mock_lmp.extract_variable = lambda name, *args: 0.02 if name == "max_g" else 0.0
    evaluator(mock_lmp)
    assert evaluator.consecutive_exceedances == 0

    # Step 3, 4, 5: Sustained high uncertainty (Halt triggered)
    mock_lmp.extract_variable = lambda name, *args: 0.06 if name == "max_g" else 0.0
    evaluator(mock_lmp)  # 1
    evaluator(mock_lmp)  # 2

    with pytest.raises(MDHaltInterrupt):
        evaluator(mock_lmp)  # 3 -> Halts


def test_uat_03_02_intelligent_cutout() -> None:
    """
    Scenario ID: UAT-03-02
    Objective: Verify intelligent cutout force weights.
    Behavior 02: The Intelligent Cutout must perfectly assign force weights based on radii.
    """
    atoms = bulk("Cu", "sc", a=2.5).repeat((3, 3, 3))

    config = CutoutConfig(
        core_radius=2.6, buffer_radius=1.0, enable_pre_relaxation=False, enable_passivation=False
    )

    # Epicenter at index 13
    cluster = extract_intelligent_cluster(atoms, target_atoms=[13], config=config)

    weights = cluster.get_array("force_weight")

    # Verify core atoms (dist <= 2.6) have weight 1.0
    n_core = np.sum(weights == 1.0)
    assert n_core == 7  # 1 center + 6 nearest neighbors in SC lattice

    # Verify buffer atoms (2.6 < dist <= 3.6) have weight 0.0
    n_buffer = np.sum(weights == 0.0)
    assert n_buffer == 12  # 12 next-nearest neighbors


def test_uat_03_03_seamless_md_resume(tmp_path) -> None:
    """
    Scenario ID: UAT-03-03
    Objective: Verify seamless continuous MD resume logic preserves trajectory parameters.
    Behavior 03: The Seamless Resume mechanism must preserve the physical trajectory.
    """
    config = MDConfig(n_steps=2000, fix_halt=False, temperature=300.0, pressure=1.0, timestep=0.001)
    engine = LammpsEngine(config)

    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "pot.yace"
    pot_path.touch()

    # Capture script generation output
    script_content = []

    # Mock LammpsDriver to just capture what it's asked to run
    with patch("pyacemaker.core.engine.LammpsDriver") as mock_driver:
        driver_instance = mock_driver.return_value
        driver_instance.extract_variable.return_value = 0.0
        driver_instance.get_forces.return_value = np.zeros((1, 3))
        driver_instance.get_stress.return_value = np.zeros(6)

        def capture_run(path: str) -> None:
            script_content.append(Path(path).read_text())

        driver_instance.run_file.side_effect = capture_run

        # Initial run (Halt simulated externally at step 1500)
        # For testing resume, we need a valid restart file
        with (
            patch("pyacemaker.core.engine.Path.exists", return_value=True),
            patch("pyacemaker.core.engine.Path.stat") as mock_stat,
            patch("pyacemaker.core.validator.Path.is_file", return_value=True),
            patch("pyacemaker.core.validator.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.utils.path.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.core.lammps_generator.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.core.engine.LammpsEngine._validate_script_content"),
            patch("pyacemaker.core.lammps_generator.Path.is_relative_to", return_value=True),
            patch("pyacemaker.core.engine.LammpsEngine._execute_simulation") as mock_exec,
        ):

            def side_effect_exec(driver, script_path):
                script_content.append(script_path.read_text())

            mock_exec.side_effect = side_effect_exec

            mock_stat.return_value.st_size = 100

            # Resume from step 1500 with override
            engine.run(atoms, pot_path, resume_from_step=1500, override_n_steps=400)

    assert len(script_content) == 1
    script = script_content[0]

    # Verify seamless resume parameters
    assert "read_restart" in script
    assert "reset_timestep ${step}" in script
    assert "run 400" in script # overridden steps
