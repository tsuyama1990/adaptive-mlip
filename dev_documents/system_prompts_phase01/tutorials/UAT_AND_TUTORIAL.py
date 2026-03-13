import typing
from typing import Any

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def init() -> typing.Any:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import tempfile
    import types
    from pathlib import Path
    from unittest.mock import patch

    import marimo as mo
    import numpy as np
    from ase import Atoms
    from ase.build import bulk

    from pyacemaker.core.engine import LammpsEngine, TwoTierEvaluator
    from pyacemaker.core.exceptions import MDHaltInterrupt
    from pyacemaker.domain_models.md import MDConfig
    from pyacemaker.domain_models.workflow import ActiveLearningThresholds, CutoutConfig
    from pyacemaker.utils.extraction import extract_intelligent_cluster

    mo.md("# PYACEMAKER Cycle 03: Advanced Workflow Demonstration")
    return (
        ActiveLearningThresholds,
        Atoms,
        CutoutConfig,
        LammpsEngine,
        MDConfig,
        MDHaltInterrupt,
        Path,
        TwoTierEvaluator,
        bulk,
        extract_intelligent_cluster,
        mo,
        np,
        patch,
        tempfile,
        types,
    )


@app.cell
def text_1(mo: Any) -> Any:
    return mo.md(
        "## Scenario UAT-01 & 03: The Two-Tier Evaluator & Seamless Resume\nThis demonstrates the noise-filtering logic that prevents premature halts and how a seamless resume works."
    )


@app.cell
def cell_2(
    ActiveLearningThresholds: Any,
    TwoTierEvaluator: Any,
    MDHaltInterrupt: Any,
    mo: Any,
    types: Any,
) -> Any:
    thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, smooth_steps=3, threshold_add_train=0.02
    )
    evaluator = TwoTierEvaluator(thresholds)
    mock_lmp = types.SimpleNamespace()

    output = []

    # Step 1
    mock_lmp.extract_variable = lambda name, *args: 0.06 if name == "max_g" else 0.0
    evaluator(mock_lmp)
    output.append(
        f"Step 1: Spike to 0.06. Consecutive exceedances: {evaluator.consecutive_exceedances}"
    )

    # Step 2
    mock_lmp.extract_variable = lambda name, *args: 0.02 if name == "max_g" else 0.0
    evaluator(mock_lmp)
    output.append(
        f"Step 2: Drop to 0.02. Consecutive exceedances: {evaluator.consecutive_exceedances}"
    )

    # Step 3, 4, 5
    mock_lmp.extract_variable = lambda name, *args: 0.06 if name == "max_g" else 0.0
    evaluator(mock_lmp)
    output.append(
        f"Step 3: Spike to 0.06. Consecutive exceedances: {evaluator.consecutive_exceedances}"
    )
    evaluator(mock_lmp)
    output.append(
        f"Step 4: Spike to 0.06. Consecutive exceedances: {evaluator.consecutive_exceedances}"
    )

    try:
        evaluator(mock_lmp)
    except MDHaltInterrupt as e:
        output.append(f"Step 5: {e}")

    mo.md("### Results:\n" + "\n".join([f"- {line}" for line in output]))
    return evaluator, mock_lmp, output, thresholds


@app.cell
def cell_3(
    MDConfig: Any,
    LammpsEngine: Any,
    Atoms: Any,
    np: Any,
    Path: Any,
    tempfile: Any,
    mo: Any,
    patch: Any,
) -> Any:
    # Scenario: Seamless Resume
    config = MDConfig(n_steps=2000, fix_halt=False, temperature=300.0, pressure=1.0, timestep=0.001)
    engine = LammpsEngine(config)
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pot_path = Path(tmp_dir) / "pot.yace"
        pot_path.touch()

        script_content = []

        # Mocking external lammps interactions here as instructed for the tutorial
        with (
            patch("pyacemaker.core.engine.LammpsDriver") as mock_driver,
            patch("pyacemaker.core.engine.Path.exists", return_value=True),
            patch("pyacemaker.core.engine.Path.stat") as mock_stat,
            patch("pyacemaker.core.validator.Path.is_file", return_value=True),
            patch("pyacemaker.core.validator.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.utils.path.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.core.lammps_generator.validate_path_safe", return_value=pot_path),
            patch("pyacemaker.core.engine.LammpsEngine._validate_script_content"),
            patch("pyacemaker.core.lammps_generator.Path.is_relative_to", return_value=True),
            patch("pyacemaker.core.engine.LammpsEngine._execute_simulation") as mock_exec,
            patch("pyacemaker.core.engine.LammpsEngine._extract_results") as mock_extract,
        ):

            def side_effect_exec(driver: Any, script_path: Any) -> None:
                script_content.append(script_path.read_text())

            mock_exec.side_effect = side_effect_exec
            mock_stat.return_value.st_size = 100
            mock_extract.return_value = None

            # Simulate Resume from step 1500
            engine.run(atoms, pot_path, resume_from_step=1500)

    mo.md(f"### Generated Resume Script Fragment:\n```lammps\n{script_content[0]}\n```")
    return (
        atoms,
        config,
        engine,
        mock_driver,
        mock_exec,
        mock_stat,
        pot_path,
        script_content,
        side_effect_exec,
        tmp_dir,
    )


@app.cell
def text_2(mo: Any) -> Any:
    return mo.md(
        "## Scenario UAT-02: Intelligent Cutout and Safe Passivation\nDemonstrates precision isolation of atomic regions."
    )


@app.cell
def cell_4(
    CutoutConfig: Any,
    bulk: Any,
    extract_intelligent_cluster: Any,
    mo: Any,
    np: Any,
) -> Any:
    atoms_sc = bulk("Cu", "sc", a=2.5).repeat((3, 3, 3))
    cutout_config = CutoutConfig(
        core_radius=2.6, buffer_radius=1.0, enable_pre_relaxation=False, enable_passivation=False
    )

    cluster = extract_intelligent_cluster(atoms_sc, target_atoms=[13], config=cutout_config)
    weights = cluster.get_array("force_weight")

    n_core = np.sum(weights == 1.0)
    n_buffer = np.sum(weights == 0.0)

    mo.md(
        f"### Cutout Results:\n- Extracted Core Atoms (Weight 1.0): {n_core}\n- Extracted Buffer Atoms (Weight 0.0): {n_buffer}\n- Total: {len(cluster)}"
    )
    return atoms_sc, cluster, cutout_config, n_buffer, n_core, weights


if __name__ == "__main__":
    app.run()
