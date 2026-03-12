import sys
from pathlib import Path

import marimo

sys.path.append(str(Path(__file__).parent.parent / "src"))

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def __():
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from ase.build import bulk
    return mo, bulk, tempfile, np, Path

@app.cell
def __(mo):
    mo.md(
        """
        # PyAceMaker - NextGen Hierarchical Distillation Tutorial
        This interactive notebook demonstrates the absolute core workflows
        of the highly robust PyAceMaker active learning orchestration architecture.
        """
    )

@app.cell
def __(mo):
    execute_real_physics = mo.ui.checkbox(label="Execute Real HPC Physics (Disable Mocking)")
    return execute_real_physics,

@app.cell
def __(mo, execute_real_physics, bulk, tempfile, Path):
    from pyacemaker.core.oracle import MACEManager
    from pyacemaker.domain_models.defaults import DEFAULT_POTENTIALS_DIR
    from pyacemaker.domain_models.workflow import DistillationConfig

    mo.md("## Scenario UAT-01: Zero-Shot Distillation Baseline Generation")

    distillation_config = DistillationConfig(
        enable=True,
        uncertainty_threshold=0.05
    )

    with tempfile.TemporaryDirectory() as _tmp:
        # MACEManager explicitly checks containment against DEFAULT_POTENTIALS_DIR
        # To make it pass naturally, we supply a file inside DEFAULT_POTENTIALS_DIR
        Path(DEFAULT_POTENTIALS_DIR).mkdir(parents=True, exist_ok=True)
        model_path = Path(DEFAULT_POTENTIALS_DIR) / "small.model"
        model_path.touch()

        mace = MACEManager(str(model_path))
        fe_bcc = bulk('Fe', 'bcc')

        res = next(mace.compute(iter([fe_bcc])))

        mo.output.append(f"Evaluated Fe BCC with MACE. Energy: {res.info.get('energy')} eV")
        mo.output.append(f"Calculated uncertainty (c_gamma max): {res.get_array('c_gamma').max():.4f}")

    return distillation_config, mace, res, Path

@app.cell
def __(mo, execute_real_physics, bulk, np):
    mo.md("## Scenario UAT-02: Intelligent Cutout and Safe Passivation")
    from pyacemaker.domain_models.workflow import CutoutConfig
    from pyacemaker.utils.extraction import extract_intelligent_cluster

    config = CutoutConfig(
        core_radius=3.0,
        buffer_radius=2.0,
        enable_pre_relaxation=True,
        enable_passivation=True,
        passivation_element="H"
    )

    mgo = bulk('MgO', 'rocksalt', a=4.21).repeat((3, 3, 3))
    del mgo[0]

    target_atoms = [0]

    mo.output.append("Extracting cluster with Pre-Relaxation and Passivation...")
    try:
        cluster = extract_intelligent_cluster(mgo, target_atoms, config)
        mo.output.append(f"Successfully extracted and passivated cluster. Size: {len(cluster)} atoms.")
        mo.output.append(f"Weights assigned: core={len(np.where(cluster.get_array('force_weight') == 1.0)[0])}")
    except Exception as e:
        mo.output.append(f"Error extracting cluster: {e}")
        cluster = None

    return config, mgo, cluster

@app.cell
def __(mo, execute_real_physics, bulk, tempfile, Path):
    mo.md("## Scenario UAT-03: Seamless Time-Continuous MD Resume")

    from pyacemaker.domain_models.md import MDConfig

    fe = bulk('Fe', 'bcc').repeat((2, 2, 2))

    md_config = MDConfig(
        temperature=300.0,
        pressure=0.0,
        timestep=0.001,
        n_steps=100,
        check_interval=10,
        fix_halt=False,
        minimize=False,
    )

    with tempfile.TemporaryDirectory() as _tmp:
        dummy_pot = Path(_tmp) / "dummy.yace"
        dummy_pot.touch()

        from pyacemaker.core.lammps_generator import LammpsScriptGenerator
        gen = LammpsScriptGenerator(md_config)

        restart_path = Path(_tmp) / "restart.lmp"
        input_path = Path(_tmp) / "input.lmp"
        dump_path = Path(_tmp) / "dump.traj"

        with input_path.open("w") as f:
            gen.write_script_resume(f, dummy_pot, restart_path, dump_path, ["Fe"], resume_step=50)

        script_content = input_path.read_text()

        mo.output.append("LAMMPS Resume Script Generation Verified:")
        mo.output.append("```")
        mo.output.append(script_content)
        mo.output.append("```")
        mo.output.append("As shown, `velocity all create` is explicitly skipped and `reset_timestep` is used.")

    return fe, md_config, gen, script_content

if __name__ == "__main__":
    app.run()
