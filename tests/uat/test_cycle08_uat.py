from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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

        import numpy as np

        import pyacemaker.domain_models.defaults

        class FakeMACEModel:
            def get_property(self, name, atoms=None, allow_calculation=True):
                if name == "node_energy_variance":
                    if hasattr(self, "variance_fixed"):
                        return self.variance_fixed
                    return np.random.uniform(0.01, 0.1, size=len(atoms)) if atoms else np.zeros(0)
                return None

            def get_potential_energy(self, atoms=None, force_consistent=False):
                return -10.0 * len(atoms) if atoms else 0.0

            def get_forces(self, atoms=None):
                return np.zeros((len(atoms), 3)) if atoms else np.zeros((0, 3))

        with patch.object(
            pyacemaker.domain_models.defaults, "DEFAULT_POTENTIALS_DIR", str(pot_dir.resolve())
        ), patch("mace.calculators.mace_mp", return_value=FakeMACEModel()):
            mace_manager = MACEManager(str(model_file))

        atoms1 = Atoms("Fe", cell=[2, 2, 2], pbc=True)
        atoms2 = Atoms("Pt", cell=[2, 2, 2], pbc=True)

        results = list(mace_manager.compute(iter([atoms1, atoms2])))

        assert len(results) == 2
        # We assert the calculator holds energy and forces, as we moved away from info/arrays mock
        assert results[0].calc is not None
        assert results[0].get_potential_energy() is not None  # type: ignore[no-untyped-call]
        assert results[0].get_forces() is not None  # type: ignore[no-untyped-call]

        for atoms in results:
            c_gamma = atoms.get_array("c_gamma")  # type: ignore[no-untyped-call]
            assert (c_gamma <= 0.1).all()


def test_scenario_phase3_cutout() -> None:
    thresholds = ActiveLearningThresholds(threshold_call_dft=0.05, threshold_add_train=0.02)
    config = CutoutConfig(core_radius=3.0, buffer_radius=2.0)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        pot_dir = Path(temp_dir)
        model_file = pot_dir / "model"
        with model_file.open("w") as f:
            f.write("dummy")

        import numpy as np

        import pyacemaker.domain_models.defaults

        class FakeMACEModel:
            def __init__(self) -> None:
                self.variance_fixed = np.array([0.1, 0.1])

            def get_property(self, name, atoms=None, allow_calculation=True):
                if name == "node_energy_variance":
                    return self.variance_fixed
                return None

            def get_potential_energy(self, atoms=None, force_consistent=False):
                return -10.0 * len(atoms) if atoms else 0.0

            def get_forces(self, atoms=None):
                return np.zeros((len(atoms), 3)) if atoms else np.zeros((0, 3))

        with patch.object(
            pyacemaker.domain_models.defaults, "DEFAULT_POTENTIALS_DIR", str(pot_dir.resolve())
        ), patch("mace.calculators.mace_mp", return_value=FakeMACEModel()):
            mace_manager = MACEManager(str(model_file))

        atoms = Atoms("FePt", positions=[[0, 0, 0], [1, 1, 1]], cell=[10, 10, 10])
        dft_manager = MagicMock(spec=DFTManager)
        dft_manager.compute.return_value = iter([atoms])

        oracle = TieredOracle(mace_manager, dft_manager, thresholds)

        gen = oracle.compute(iter([atoms]))
        _result = next(gen)

        dft_manager.compute.assert_called()

        target_atoms = [0]

        with patch("pyacemaker.utils.extraction._pre_relax_buffer", return_value=atoms):
            cluster = extract_intelligent_cluster(
                atoms, target_atoms, config, calculator=getattr(mace_manager, "calc", None)
            )

        weights = cluster.get_array("force_weight")  # type: ignore[no-untyped-call]
        assert 1.0 in weights

        symbols = cluster.get_chemical_symbols()  # type: ignore[no-untyped-call]
        assert len(symbols) > 0


@patch("pyacemaker.core.trainer.run_command")
@patch("pyacemaker.core.trainer.shutil.which")
@patch("pyacemaker.core.engine.LammpsDriver")
def test_scenario_phase4_resume(mock_driver: MagicMock, mock_which: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    from ase import Atoms
    from ase.io import write

    mock_which.return_value = "/bin/mace_run_train"

    def side_effect_run(cmd: list[str]) -> None:
        (tmp_path / "awakened_mace_model.model").touch()

    mock_run.side_effect = side_effect_run

    finetune_mgr = FinetuneManager()
    dataset_path = tmp_path / "dataset.xyz"
    write(str(dataset_path), Atoms("Fe"), format="extxyz")

    awakened_model = finetune_mgr.finetune(dataset_path)
    assert str(tmp_path / "awakened_mace_model.model") in awakened_model

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

    with patch.object(trainer, "train") as mock_train:
        mock_train.return_value = tmp_path / "test_pot.yace"
        new_pot = trainer.incremental_train(dataset_path, strategy, initial_potential="init.yace")
        assert new_pot == tmp_path / "test_pot.yace"
        mock_train.assert_called_once()

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

    driver_instance.get_forces.return_value = np.zeros((1, 3))
    driver_instance.get_stress.return_value = np.zeros(6)

    script_content = []

    def capture_run(path: str) -> None:
        script_content.append(Path(path).read_text())

    driver_instance.run_file.side_effect = capture_run

    atoms = Atoms("Fe", cell=[10, 10, 10], pbc=True)
    pot_path = tmp_path / "test_pot.yace"
    pot_path.touch()

    engine.run(atoms, pot_path, resume_from_step=1500)

    assert len(script_content) == 1
    assert "Resuming from step 1500" in script_content[0]
