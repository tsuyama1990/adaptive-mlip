from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig


@pytest.fixture
def mock_dft_config(tmp_path: Path) -> DFTConfig:
    pot_file = tmp_path / "H.UPF"
    pot_file.touch()
    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"H": str(pot_file)},
        n_workers=1,
    )


def test_dft_manager_streaming_behavior(mock_dft_config: DFTConfig) -> None:
    """
    Verify that DFTManager computes properties streaming (in batches)
    and does NOT consume the whole generator upfront.
    """
    # 1. Create an infinite or large generator
    def infinite_structures() -> Iterator[Atoms]:
        while True:
            yield Atoms("H", positions=[[0, 0, 0]])

    # 2. Mock driver
    mock_driver = MagicMock()
    calc = MagicMock()
    calc.get_stress.return_value = np.zeros(6)
    mock_driver.get_calculator.return_value = calc

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # 3. Call compute with small batch size
    stream = manager.compute(infinite_structures(), batch_size=1)

    # 4. Consume items
    first = next(stream)

    # We expect it to have consumed roughly one chunk (2 items).
    assert mock_driver.get_calculator.call_count <= 5

    second = next(stream)
    assert len(first) == 1

    # Still within first chunk
    assert mock_driver.get_calculator.call_count <= 5
