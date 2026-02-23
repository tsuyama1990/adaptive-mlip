from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from ase import Atoms

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig
from tests.conftest import MockCalculator


@pytest.fixture
def mock_dft_config(tmp_path) -> DFTConfig:
    (tmp_path / "H.UPF").touch()
    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"H": str(tmp_path / "H.UPF")},
    )


def test_dft_manager_streaming_behavior(mock_dft_config: DFTConfig) -> None:
    """
    Verify that DFTManager processes structures lazily (streaming).
    It should not consume the entire iterator upfront.
    """
    # Create a generator that yields atoms and tracks yield count
    yield_count = 0

    def structure_generator() -> Iterator[Atoms]:
        nonlocal yield_count
        for _ in range(5):
            yield_count += 1
            yield Atoms("H", cell=[10, 10, 10], pbc=True)

    # Mock Driver
    mock_driver = MagicMock()
    mock_driver.get_calculator.return_value = MockCalculator(fail_count=0)

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # Start computation
    result_stream = manager.compute(structure_generator())

    # Initially, nothing should be yielded from source
    # Note: DFTManager.compute might pre-fetch if it batches.
    # The default batch size is 10. So if we iterate once, it might try to fill a batch.
    # However, DFTManager implementation processes one by one in the current cycle.

    assert yield_count == 0

    # Consume 1 item
    next(result_stream)
    assert yield_count == 1

    # Consume 2nd item
    next(result_stream)
    assert yield_count == 2
