import tempfile
from pathlib import Path

from ase.build import bulk

from pyacemaker.core.trainer import FinetuneManager, IncrementalTrainer


def test_finetune_manager() -> None:
    manager = FinetuneManager("MACE-MP-0")
    with tempfile.TemporaryDirectory() as td:
        dummy_data = Path(td) / "data.xyz"
        dummy_data.touch()

        new_model = manager.finetune(dummy_data)
        assert "finetuned_MACE-MP-0.pt" in new_model.name
        assert new_model.exists()


def test_incremental_trainer() -> None:
    # Mock base trainer
    class MockTrainer:
        def train(self, path, init=None):
            return path

    base = MockTrainer()
    trainer = IncrementalTrainer(base, replay_buffer_size=2)  # type: ignore

    a1 = bulk("Cu")
    a2 = bulk("Ag")
    a3 = bulk("Au")

    trainer.add_to_buffer(a1)
    trainer.add_to_buffer(a2)
    assert len(trainer._buffer) == 2

    # Adding third should pop the first
    trainer.add_to_buffer(a3)
    assert len(trainer._buffer) == 2
    assert trainer._buffer[0] == a2

    # Check train delegation
    assert trainer.train("test.xyz", "init.yace") == "test.xyz"
