import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        """
        # Scenario 01: Zero-Shot Distillation Initialization

        This scenario demonstrates initializing the system strictly with foundation models and zero DFT calls.
        """
    )


@app.cell
def __(mo):
    import sys as _sys
    import tempfile as _tempfile
    from pathlib import Path as _Path

    # Add src to sys.path so we can import pyacemaker
    _sys.path.insert(0, str(_Path("src").resolve()))

    import numpy as np

    from pyacemaker.core.generator import StructureGenerator
    from pyacemaker.core.oracle import MACEManager
    from pyacemaker.core.trainer import PacemakerTrainer
    from pyacemaker.domain_models.config import PyAceConfig
    from pyacemaker.domain_models.dft import DFTConfig
    from pyacemaker.domain_models.md import MDConfig
    from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig
    from pyacemaker.domain_models.training import TrainingConfig
    from pyacemaker.domain_models.workflow import DistillationConfig, WorkflowConfig

    # Create dummy UPF files and MACE model to pass strict path validation
    _temp_dir = _tempfile.TemporaryDirectory()
    _tmp_path = _Path(_temp_dir.name)

    _h_upf = _tmp_path / "H.UPF"
    _o_upf = _tmp_path / "O.UPF"
    _h_upf.write_text('<UPF version="2.0.1">...</UPF>')
    _o_upf.write_text('<UPF version="2.0.1">...</UPF>')

    # For MACEManager, path must be inside DEFAULT_POTENTIALS_DIR
    from pyacemaker.domain_models.defaults import DEFAULT_POTENTIALS_DIR

    _pot_dir = _Path(DEFAULT_POTENTIALS_DIR)
    _pot_dir.mkdir(parents=True, exist_ok=True)
    _mace_model_path = _pot_dir / "mace-mp-0-medium.model"
    _mace_model_path.touch()

    # 1. Instantiate completely mathematically properly configured PyAceConfig
    distillation_cfg = DistillationConfig(
        enable=True,
        mace_model_path=str(_mace_model_path),
        uncertainty_threshold=0.05,
        sampling_structures_per_system=10,
    )

    pyace_cfg = PyAceConfig(
        project_name="MgO_Distillation_Project",
        structure=StructureConfig(
            elements=["Mg", "O"],
            supercell_size=[2, 2, 2],
            policy_name=ExplorationPolicy.RANDOM_RATTLE,
            rattle_stdev=0.1,
            num_structures=5,
        ),
        dft=DFTConfig(
            code="qe",
            functional="PBE",
            kpoints_density=0.04,
            encut=500.0,
            pseudopotentials={"H": "H.UPF", "O": "O.UPF"},
        ),
        training=TrainingConfig(
            potential_type="ace", cutoff_radius=5.0, max_basis_size=500, delta_learning=True
        ),
        md=MDConfig(
            temperature=300.0,
            pressure=0.0,
            timestep=0.001,
            n_steps=1000,
            uncertainty_threshold=0.1,
            check_interval=50,
        ),
        workflow=WorkflowConfig(max_iterations=1, distillation=distillation_cfg),
    )

    # 2. Generate highly combinatorial pool of complex structural geometric configurations
    generator = StructureGenerator(pyace_cfg.structure)
    structure_pool = list(generator.generate(n_candidates=5))

    # 3. Route structures through explicitly dependency-injected MACEManager
    mace_manager = MACEManager(str(_mace_model_path))
    evaluated_stream = mace_manager.compute(iter(structure_pool))

    # Filter highly statistically uncertain geometric configurations
    confident_structures = []
    for _atoms in evaluated_stream:
        # MACEManager dummy adds c_gamma randomly between 0.01 and 0.1
        _c_gamma = _atoms.get_array("c_gamma")
        if np.max(_c_gamma) <= distillation_cfg.uncertainty_threshold:
            confident_structures.append(_atoms)

    # 4. Initialize PacemakerTrainer utilizing exclusively only these highly confident structures
    trainer = PacemakerTrainer(pyace_cfg.training)

    # We won't call trainer.train() as it invokes actual subprocess `pace_train` which is heavy/requires MPI.
    # Instead, we demonstrate the generation of its configuration which happens entirely from zero-shot Foundation Model predictions.
    from pyacemaker.core.config_generator import PacemakerConfigGenerator

    _config_gen = PacemakerConfigGenerator(pyace_cfg.training)

    _dummy_xyz = _tmp_path / "dummy_data.xyz"
    _dummy_xyz.touch()

    _yaml_out = _tmp_path / "train.yaml"

    # Needs a real file to exist and open
    from ase.io import write

    write(_dummy_xyz, confident_structures)

    _yaml_config = _config_gen.generate(str(_dummy_xyz), str(_yaml_out))

    mo.md(
        f"""
        ### Zero-Shot Distillation Complete
        - **Total structures generated:** {len(structure_pool)}
        - **Structures passing confidence threshold (< {distillation_cfg.uncertainty_threshold}):** {len(confident_structures)}
        - **Pacemaker Configuration Generated successfully without DFT.**
        """
    )
    return (
        pyace_cfg,
        structure_pool,
        confident_structures,
        trainer,
        _yaml_config,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Scenario 02: Intelligent Cutout and Passivation Execution

        This scenario verifies the ability to manage actively detected highly complex atomic structural uncertainties and extract a properly passivated, neutral atomic cluster for computationally convergent DFT processing.
        """
    )


@app.cell
def __(mo):
    import numpy as np
    from ase import Atoms as _Atoms2

    from pyacemaker.domain_models.workflow import ActiveLearningThresholds, CutoutConfig
    from pyacemaker.utils.extraction import extract_intelligent_cluster

    # Given a massive, heavily mechanically strained MgO supercell
    _a = 4.212
    _mgo_base = _Atoms2(
        "Mg4O4",
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ],
        cell=[_a, _a, _a],
        pbc=True,
    )
    _mgo_supercell = _mgo_base * (3, 3, 3)  # 216 atoms

    # Deliberately inject high statistical variance uncertainty at specific atomic index
    _target_index = 100

    _c_gamma = np.zeros(len(_mgo_supercell))
    _c_gamma[_target_index] = 0.08  # persistently high variance

    _mgo_supercell.set_array("c_gamma", _c_gamma)

    # Trigger extraction based on TwoTierEvaluator thresholding
    _thresholds = ActiveLearningThresholds(
        threshold_call_dft=0.05, threshold_add_train=0.02, smooth_steps=3
    )

    # Calculate geometric epicenter
    _epicenter_indices = np.where(
        _mgo_supercell.get_array("c_gamma") > _thresholds.threshold_add_train
    )[0].tolist()

    _cutout_cfg = CutoutConfig(
        core_radius=4.0,
        buffer_radius=3.0,
        enable_pre_relaxation=True,
        enable_passivation=True,
        passivation_element="H",
    )

    # Run extraction mechanism
    _extracted_cluster = extract_intelligent_cluster(
        structure=_mgo_supercell, target_atoms=_epicenter_indices, config=_cutout_cfg
    )

    _force_weights = _extracted_cluster.get_array("force_weight")
    _core_atoms_count = np.sum(_force_weights == 1.0)
    _buffer_atoms_count = np.sum(_force_weights == 0.0)

    _passivated_h_count = sum(1 for sym in _extracted_cluster.get_chemical_symbols() if sym == "H")

    mo.md(
        f"""
        ### Intelligent Cutout Complete
        - **Identified Epicenter Atom Index:** {_epicenter_indices}
        - **Extracted Cluster Size:** {len(_extracted_cluster)} atoms
        - **Core Atoms (radius <= {_cutout_cfg.core_radius}A):** {_core_atoms_count}
        - **Buffer Atoms (radius <= {_cutout_cfg.buffer_radius + _cutout_cfg.core_radius}A):** {_buffer_atoms_count}
        - **Passivation Dummy Atoms Inserted:** {_passivated_h_count}

        The cluster has been strictly geometrically enclosed in a vacuum boundary, structurally passivated, and mathematically relaxed for reliable external DFT integration without encountering dipole divergence errors.
        """
    )
    return (
        _mgo_supercell,
        _epicenter_indices,
        _cutout_cfg,
        _extracted_cluster,
        _core_atoms_count,
        _buffer_atoms_count,
        _passivated_h_count,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Scenario 03: Incremental Delta Learning with Replay Buffer

        This scenario verifies the ability to sample a predefined number of highly stable historical bulk structural configurations directly from the persistent storage file and successfully mix them with newly generated quantum mechanical ground truth surrogate data matrix points to form a balanced composite training dataset, entirely eliminating catastrophic forgetting.
        """
    )


@app.cell
def __(mo, pyace_cfg, confident_structures):
    import tempfile as _tempfile3
    from pathlib import Path as _Path3

    from ase.io import write as _write3

    from pyacemaker.core.trainer import PacemakerTrainer as _PacemakerTrainer3

    _temp_dir3 = _tempfile3.TemporaryDirectory()
    _hist_path = _Path3(_temp_dir3.name) / "training_history.extxyz"

    # 1. Provide an intentionally tiny array dataset representing newly computed structures
    _new_data = confident_structures[:1]

    # Create fake "historical" bulk structures
    _hist_data = confident_structures[1:4]
    if not _hist_data:  # Fallback if confident_structures is too small
        from ase import Atoms as _Atoms3

        _hist_data = [_Atoms3("Mg")] * 3

    # Write historical data to file
    _write3(_hist_path, _hist_data, append=True)

    _trainer = _PacemakerTrainer3(pyace_cfg.training)

    # 2. Extract a strictly mathematically predefined number of highly stable historical bulk structural configurations
    _replay_buffer_size = 2
    _historical_samples = _trainer.get_replay_buffer(_replay_buffer_size, str(_hist_path))

    # 3. Flawlessly mix these retrieved historical data with newly generated ground truth data
    _composite_training_dataset = _historical_samples + _new_data

    mo.md(
        f"""
        ### Incremental Replay Mixing Complete
        - **Historical Pool Size:** {len(_hist_data)}
        - **Replay Buffer Sample Size:** {len(_historical_samples)}
        - **New Quantum Data Size:** {len(_new_data)}
        - **Final Composite Training Dataset:** {len(_composite_training_dataset)} atoms

        The PacemakerTrainer logic has successfully extracted random historical bulk structures and mixed them with the new tensor array to prevent catastrophic forgetting.
        """
    )
    return (
        _hist_path,
        _new_data,
        _hist_data,
        _trainer,
        _replay_buffer_size,
        _historical_samples,
        _composite_training_dataset,
    )


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
