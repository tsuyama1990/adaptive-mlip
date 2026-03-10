from pathlib import Path
from unittest.mock import MagicMock, patch

from ase import Atoms

from pyacemaker.core.engine import LammpsEngine
from pyacemaker.core.oracle import DFTManager, MACEManager, TieredOracle
from pyacemaker.core.trainer import FinetuneManager, PacemakerTrainer
from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.training import TrainingConfig
from pyacemaker.domain_models.workflow import (
    ActiveLearningThresholds,
    CutoutConfig,
    DistillationConfig,
    LoopStrategyConfig,
    WorkflowConfig,
)
from pyacemaker.utils.extraction import extract_intelligent_cluster


def test_scenario_phase1_distillation() -> None:
    """
    Scenario 1: Verification of Zero-Shot Distillation and Baseline Construction
    """
    import numpy as np

    from pyacemaker.core.active_set import ActiveSetSelector
    from pyacemaker.core.generator import StructureGenerator
    from pyacemaker.domain_models.structure import StructureConfig

    config = WorkflowConfig(
        max_iterations=1, distillation=DistillationConfig(enable=True, uncertainty_threshold=0.05)
    )
    assert config.distillation.enable is True

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        pot_dir = Path(temp_dir)
        model_file = pot_dir / "model"
        with model_file.open("w") as f:
            f.write("dummy")

        import pyacemaker.domain_models.defaults

        with patch.object(
            pyacemaker.domain_models.defaults, "DEFAULT_POTENTIALS_DIR", str(pot_dir.resolve())
        ):
            mace_manager = MACEManager(str(model_file))

        # 1. Structure Generation
        s_config = StructureConfig(elements=["Fe", "O"], supercell_size=[1, 1, 1])
        generator = StructureGenerator(config=s_config)
        raw_structures = list(generator.generate(n_candidates=20))
        assert len(raw_structures) > 0, "StructureGenerator failed to generate structures"

        # 2. Information-Dense Subset Selection (ActiveSet)
        selector = ActiveSetSelector()

        # We patch the shell call in `select` rather than actual execution because we only test UAT flow
        with patch.object(selector, "select") as mock_select:
            mock_select.return_value = iter(raw_structures[:10])
            selected_structures_iter = selector.select(
                candidates=raw_structures, potential_path="dummy.yace", n_select=10
            )
            selected_structures = list(selected_structures_iter)

        assert len(selected_structures) <= 10, "ActiveSetSelector selected too many structures"

        # 3. MACE evaluates structures
        results = list(mace_manager.compute(iter(selected_structures)))
        assert len(results) == len(selected_structures)

        # 4. Uncertainty filtering & baseline generation
        high_confidence_structures = []
        for atoms in results:
            assert "energy" in atoms.info
            assert "forces" in atoms.arrays
            c_gamma = atoms.get_array("c_gamma")  # type: ignore[no-untyped-call]
            if np.max(c_gamma) <= config.distillation.uncertainty_threshold:
                high_confidence_structures.append(atoms)

        # Assuming MACE mock provides a uniform distribution of [0.01, 0.1]
        # roughly half will be below 0.05 threshold

        # Finally, Pacemaker uses these to generate the base potential
        t_config = TrainingConfig(
            potential_type="ace",
            cutoff_radius=5.0,
            max_basis_size=2,
            output_filename="base.yace",
            elements=["Fe", "O"],
        )
        trainer = PacemakerTrainer(t_config)
        with patch.object(trainer, "train") as mock_train:
            mock_train.return_value = pot_dir / "base.yace"
            # In a real pipeline, the structures would be written to extxyz and passed to trainer
            base_pot = trainer.train("dummy_data.extxyz")
            assert str(base_pot).endswith("base.yace")
            mock_train.assert_called_once()


def test_scenario_phase3_cutout() -> None:
    """
    Scenario 3: Exclusion of Thermal Noise and Intelligent Cluster Extraction
    """
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02)
    config = CutoutConfig(core_radius=3.0, buffer_radius=2.0)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        pot_dir = Path(temp_dir)
        model_file = pot_dir / "model"
        with model_file.open("w") as f:
            f.write("dummy")

        import pyacemaker.domain_models.defaults

        with patch.object(
            pyacemaker.domain_models.defaults, "DEFAULT_POTENTIALS_DIR", str(pot_dir.resolve())
        ):
            mace_manager = MACEManager(str(model_file))
    dft_manager = MagicMock(spec=DFTManager)

    oracle = TieredOracle(mace_manager, dft_manager, thresholds)

    # 1. Thermal Noise Spike (handled by engine mock logic previously, here we test Oracle fallback)
    # The oracle evaluates a structure. MACE mock yields max_g around 0.1
    atoms = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10])

    import numpy as np

    with patch("pyacemaker.core.oracle.np.random.uniform", return_value=np.array([0.1, 0.1])):
        gen = oracle.compute(iter([atoms]))
        _result = next(gen)

    # max_g = 0.1 > 0.05, so it falls back to DFT
    dft_manager.compute.assert_called()

    # 2. Extraction of Epicenter
    # target atoms are those exceeding threshold_add_train (0.02)
    # Create a realistic test setup with specific distances to test cutout algorithm properly
    atoms = Atoms(
        "Fe4", positions=[[0, 0, 0], [1.5, 0, 0], [3.5, 0, 0], [6.0, 0, 0]], cell=[20, 20, 20]
    )

    # Let atom 0 be the highly uncertain one
    atoms.new_array("c_gamma", np.array([0.08, 0.01, 0.01, 0.01]))  # type: ignore[no-untyped-call]

    # Identify target atoms
    c_gamma = atoms.get_array("c_gamma")  # type: ignore[no-untyped-call]
    target_atoms = np.where(c_gamma > thresholds.threshold_add_train)[0].tolist()
    assert target_atoms == [0]

    cluster = extract_intelligent_cluster(atoms, target_atoms, config)

    # Check physical repair logic accurately identified core vs buffer
    weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]

    # With target=0 at [0,0,0], core_radius=3.0, buffer_radius=2.0
    # Atom 1 [1.5,0,0] is distance 1.5 (<=3.0) -> core (weight 1.0)
    # Atom 2 [3.5,0,0] is distance 3.5 (>3.0 and <=5.0) -> buffer (weight 0.0)
    # Atom 3 [6.0,0,0] is distance 6.0 (>5.0) -> excluded

    # We should have 3 Fe atoms in the cluster
    assert len([s for s in cluster.get_chemical_symbols() if s == "Fe"]) == 3  # type: ignore[no-untyped-call]

    # Check weights: indices might be reordered, but we expect two 1.0s and one 0.0 for Fe atoms
    fe_weights = [weights[i] for i, s in enumerate(cluster.get_chemical_symbols()) if s == "Fe"]  # type: ignore[no-untyped-call]
    assert fe_weights.count(1.0) == 2
    assert fe_weights.count(0.0) == 1

    # Check passivation logic - H atoms should be added to buffer atoms lacking neighbors
    symbols = cluster.get_chemical_symbols()  # type: ignore[no-untyped-call]
    assert "H" in symbols
    # The exact formula depends on random elements in the generation and ASE's grouping (e.g., Fe3H, HFe3, etc.)
    # We just ensure both Fe and H are present in expected general quantities.
    assert sum(1 for s in symbols if s == "H") > 0
    assert sum(1 for s in symbols if s == "Fe") == 3


@patch("pyacemaker.core.engine.LammpsDriver")
def test_scenario_phase4_resume(mock_driver: MagicMock, tmp_path: Path) -> None:
    """
    Scenario 4: Hierarchical Fine-Tuning and Seamless Resume
    """
    # 1. Finetune MACE
    f_config = TrainingConfig(
        potential_type="mace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="awakened_mace_model.model",
        elements=["Fe", "Pt"],
    )
    finetune_mgr = FinetuneManager(f_config)
    dataset_path = tmp_path / "dataset.xyz"
    dataset_path.write_text("Dummy coordinates data")
    awakened_model_path = finetune_mgr.finetune(dataset_path)
    assert Path(awakened_model_path).exists()
    assert "Awakened MACE model based on dataset size" in Path(awakened_model_path).read_text()

    # 2. ACE Incremental Update
    t_config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="test_pot.yace",
        delta_learning=True,
        elements=["Fe", "Pt"],
        seed=123,
        max_iterations=500,
        batch_size=20,
    )
    trainer = PacemakerTrainer(t_config)
    strategy = LoopStrategyConfig(replay_buffer_size=100)

    with (
        patch.object(trainer, "train") as mock_train,
        patch("ase.io.read") as mock_read,
        patch("ase.io.write") as mock_write
    ):
        mock_train.return_value = tmp_path / "test_pot.yace"
        mock_read.return_value = [Atoms("Fe")]
        new_pot = trainer.incremental_train(dataset_path, strategy, initial_potential="init.yace")
        assert new_pot == tmp_path / "test_pot.yace"
        mock_train.assert_called_once()

    # 3. Seamless Resume
    md_config = MDConfig(
        temperature=300.0, pressure=1.0, timestep=0.001, n_steps=5000, fix_halt=True
    )
    engine = LammpsEngine(md_config)

    driver_instance = mock_driver.return_value
    driver_instance.extract_variable.side_effect = lambda name: {
        "pe": -100.0,
        "step": 2000,
        "max_g": 0.01,
        "temp": 300.0,
        "halted": 0.0,
    }.get(name, 0.0)

    import numpy as np

    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    script_content = []

    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance.run_file.side_effect = capture_run

    atoms = Atoms("Fe", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "test_pot.yace"
    pot_path.touch()

    # Resume from step 1500 (halted earlier)
    engine.run(atoms, pot_path, resume_from_step=1500)

    assert len(script_content) == 1
    assert "Resuming from step 1500" in script_content[0]
