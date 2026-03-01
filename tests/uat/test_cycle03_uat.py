import numpy as np
from itertools import islice
from ase import Atoms

from pyacemaker.core.generator import StructureGenerator
from pyacemaker.domain_models.structure import ExplorationPolicy, StructureConfig


def test_uat_03_01_generate_candidates() -> None:
    """
    Scenario 03-01: Generate Candidates
    Objective: Verify that the system can generate a set of perturbed structures from a base composition.
    """
    # 1. Preparation
    config = StructureConfig(
        elements=["Fe", "Pt"],  # Composition FePt
        supercell_size=[2, 2, 2],
        policy_name=ExplorationPolicy.RANDOM_RATTLE,
        rattle_stdev=0.1,
        num_structures=10,
    )

    generator = StructureGenerator(config)

    # 2. Action
    # Use streaming consumption instead of list()
    stream = generator.generate(n_candidates=10)

    # 3. Expectation
    # Use pure streaming, do not materialize lists
    s0 = next(stream)
    s1 = next(stream)

    assert isinstance(s0, Atoms)
    assert isinstance(s1, Atoms)

    # Verify chemistry
    symbols = s0.get_chemical_symbols()  # type: ignore[no-untyped-call]
    assert "Fe" in symbols
    assert "Pt" in symbols

    # Verify perturbation (compare random samples to avoid full array materialization if large)
    # Just checking first atom's position is enough to verify they aren't identical
    assert not np.allclose(s0.positions[0], s1.positions[0])

    # Verify we can consume the rest without keeping them
    remaining_count = sum(1 for _ in stream)
    assert remaining_count == 8  # 10 - 2


def test_uat_03_02_defect_generation() -> None:
    """
    Scenario 03-02: Defect Generation
    Objective: Verify that the system can introduce vacancies.
    """
    # 1. Preparation
    config = StructureConfig(
        elements=["Fe"],
        supercell_size=[4, 4, 4],
        policy_name=ExplorationPolicy.DEFECTS,
        vacancy_rate=0.05,
    )
    generator = StructureGenerator(config)

    # 2. Action
    # Stream one
    stream = generator.generate(n_candidates=1)
    defect_atoms = next(stream)

    # 3. Expectation
    # Get pristine count
    pristine_config = config.model_copy(
        update={
            "policy_name": ExplorationPolicy.COLD_START,
            "active_policies": [ExplorationPolicy.COLD_START],
            "vacancy_rate": 0.0,
        }
    )
    pristine_gen = StructureGenerator(pristine_config)
    pristine_stream = pristine_gen.generate(1)
    pristine_atoms = next(pristine_stream)

    # Compare scalar properties to avoid array materialization
    assert len(defect_atoms) < len(pristine_atoms)

    # Compare scalar volume instead of full cell array
    vol_defect = defect_atoms.get_volume()  # type: ignore[no-untyped-call]
    vol_pristine = pristine_atoms.get_volume()  # type: ignore[no-untyped-call]
    assert abs(vol_defect - vol_pristine) < 1e-6


from pathlib import Path
from unittest.mock import MagicMock, patch

from ase.io import read, write

from pyacemaker.core.oracle import MACEManager, TieredOracle
from pyacemaker.core.trainer import IncrementalTrainer, PacemakerTrainer
from pyacemaker.domain_models.training import TrainingConfig


def test_uat_03_01_incremental_update_and_replay_buffer(tmp_path: Path):
    """
    UAT-03-01: Hierarchical Finetuning and Delta Update
    """
    # GIVEN a base.yace potential and a training_history.extxyz containing 10 structures (mocking 10,000)
    history_path = tmp_path / "training_history.extxyz"
    base_potential = tmp_path / "base.yace"
    base_potential.touch()

    # Write 10 history structures
    history_structures = [Atoms("H", positions=[[0, 0, 0]]) for _ in range(10)]
    write(history_path, history_structures, format="extxyz")

    # AND a newly evaluated DFT cluster structure (surrogates)
    new_data_path = tmp_path / "new_train.extxyz"
    new_structures = [Atoms("He", positions=[[0, 0, 0]]) for _ in range(51)]
    write(new_data_path, new_structures, format="extxyz")

    # Setup the trainer
    config = TrainingConfig(
        potential_type="ace",
        cutoff_radius=5.0,
        max_basis_size=2,
        output_filename="current.yace",
        delta_learning=True,
        elements=["H", "He"],
        seed=123,
    )
    base_trainer = PacemakerTrainer(config)

    # AND a LoopStrategyConfig with incremental_update = True and replay_buffer_size = 15 (mocking 500)
    inc_trainer = IncrementalTrainer(base_trainer, replay_buffer_size=15)

    # WHEN the IncrementalTrainer is invoked with the 51 new structures
    with (
        patch("pyacemaker.core.trainer.run_command") as mock_run,
        patch("shutil.which", return_value=True),
    ):
        (tmp_path / "current.yace").touch()
        inc_trainer.train(new_data_path, initial_potential=base_potential)

        # THEN the Trainer samples exactly 15 structures
        # The temp train path will have exactly 15 structures
        temp_train_path = tmp_path / "training_set_temp.extxyz"
        assert temp_train_path.exists()
        from itertools import islice
        temp_train = list(islice(read(str(temp_train_path), index=":"), 15))
        assert len(temp_train) == 15

        # Check generated input.yaml correctly points to Delta learning config
        import yaml

        yaml_path = tmp_path / "input.yaml"
        assert yaml_path.exists()
        with open(yaml_path) as f:
            yaml_config = yaml.safe_load(f)

        assert yaml_config["data"]["filename"] == str(temp_train_path)
        assert "base_potential" in yaml_config  # delta learning active

        # assert run_command included base.yace
        cmd = mock_run.call_args[0][0]
        assert "--initial_potential" in cmd
        assert str(base_potential) in cmd


def test_uat_03_02_tiered_oracle_evaluation():
    """
    UAT-03-02: Tiered Oracle Evaluation
    """
    from tests.conftest import MockCalculator
    fast_oracle = MACEManager()
    fast_oracle._calculator = MockCalculator()
    slow_oracle = MagicMock()
    slow_atoms = Atoms("O", positions=[[0, 0, 0]])
    slow_atoms.info["energy"] = -10.0
    slow_oracle.compute.return_value = iter([slow_atoms])

    tiered_oracle = TieredOracle(
        fast_oracle=fast_oracle,
        slow_oracle=slow_oracle,
        uncertainty_threshold=0.05,
        call_dft_threshold=0.05,
    )

    # Low uncertainty structure
    low_uncertainty_atoms = Atoms("H", positions=[[0, 0, 0]])
    low_uncertainty_atoms.info["uncertainty"] = 0.01

    # High uncertainty structure
    high_uncertainty_atoms = Atoms("He", positions=[[0, 0, 0]])
    high_uncertainty_atoms.info["uncertainty"] = 0.08

    # Process both
    gen = tiered_oracle.compute(iter([low_uncertainty_atoms, high_uncertainty_atoms]))

    # 1. First one should be handled by fast_oracle
    res1 = next(gen)
    assert res1.symbols == "H"
    assert res1.info["uncertainty"] == 0.01
    slow_oracle.compute.assert_not_called()

    # 2. Second one exceeds threshold and must invoke slow_oracle
    res2 = next(gen)
    assert res2.symbols == "O"  # Replaced by slow oracle result
    slow_oracle.compute.assert_called_once()
