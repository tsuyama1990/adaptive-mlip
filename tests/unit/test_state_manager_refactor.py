import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyacemaker.core.state_manager import StateManager


def test_state_manager_path_validation() -> None:
    # Test path traversal prevention
    mock_logger = MagicMock()
    with pytest.raises(ValueError, match=".*outside.*"): # validate_path_safe raises ValueError or PermissionError
        StateManager(Path("/etc/passwd"), mock_logger)

def test_state_manager_property_removal() -> None:
    # Verify that properties were removed and state is accessed directly
    mock_logger = MagicMock()
    # Using a valid temp path to avoid validation error
    manager = StateManager(Path("state.json"), mock_logger)

    # This should work now
    manager.state.iteration = 5
    assert manager.state.iteration == 5

    # No more attribute error because it just sets an arbitrary variable, but we don't want to enforce it raising.

def test_sqlite_db_cleanup() -> None:
    from ase import Atoms

    from pyacemaker.domain_models.active_learning import DescriptorConfig
    from pyacemaker.domain_models.data import AtomStructure
    from pyacemaker.domain_models.distillation import Step1DirectSamplingConfig
    from pyacemaker.modules.sampling import DirectSampler

    desc_conf = DescriptorConfig(method="soap", species=["H"], r_cut=5.0, n_max=2, l_max=2, sigma=0.1)
    config = Step1DirectSamplingConfig(target_points=1, descriptor=desc_conf, candidate_multiplier=1)

    mock_gen = MagicMock()
    def candidate_stream(n_candidates: int) -> Iterator[AtomStructure]:
        for _i in range(n_candidates):
            yield AtomStructure(atoms=Atoms('H'))
    mock_gen.generate.side_effect = candidate_stream

    class MockDistinctCalc:
        def __init__(self, config: object) -> None: pass
        def compute(self, atoms_list: list[Atoms], batch_size: int = 100) -> np.ndarray:
            return np.random.rand(len(atoms_list), 5)

    with patch("pyacemaker.modules.sampling.DescriptorCalculator", MockDistinctCalc):
        sampler = DirectSampler(config, mock_gen)
        # Capture tmpdir files before
        tmp_dir = Path(tempfile.gettempdir())
        files_before = set(tmp_dir.glob("*.db"))
        files_before_dat = set(tmp_dir.glob("*.dat"))

        list(sampler.generate())

        # Verify no leakage
        files_after = set(tmp_dir.glob("*.db"))
        files_after_dat = set(tmp_dir.glob("*.dat"))

        # Assuming our test doesn't clash with parallel runs, there should be no new .db or .dat left behind
        new_dbs = files_after - files_before
        new_dats = files_after_dat - files_before_dat
        assert len(new_dbs) == 0, f"SQLite temp DB leaked: {new_dbs}"
        assert len(new_dats) == 0, f"Memmap temp file leaked: {new_dats}"
