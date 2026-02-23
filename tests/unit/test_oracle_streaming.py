from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.core.oracle import DFTManager
from pyacemaker.domain_models import DFTConfig


@pytest.fixture
def mock_dft_config(tmp_path, monkeypatch) -> DFTConfig:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "H.UPF").touch()

    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        pseudopotentials={"H": "H.UPF"},
    )

def test_dft_manager_streaming_behavior(mock_dft_config: DFTConfig) -> None:
    """
    Verify that DFTManager computes properties one by one (streaming)
    and does NOT consume the whole generator upfront.
    """
    # 1. Create an infinite or large generator
    def infinite_structures() -> Any:
        i = 0
        while True:
            # Yield single atom each time
            yield Atoms("H", positions=[[0, 0, 0]])
            i += 1

    # 2. Mock driver
    mock_driver = MagicMock()
    # Mock calculator methods to return valid data (get_stress expects array)
    calc = MagicMock()
    calc.get_stress.return_value = np.zeros(6)
    mock_driver.get_calculator.return_value = calc

    manager = DFTManager(mock_dft_config, driver=mock_driver)

    # 3. Call compute
    # This should return a generator immediately without hanging
    stream = manager.compute(infinite_structures())

    # 4. Consume just a few items manually
    # Do NOT use list(stream) as it would be infinite
    first = next(stream)
    second = next(stream)

    assert len(first) == 1
    assert len(second) == 1

    # If we reached here, it means compute didn't consume the whole iterator.
    # Verify driver calls matches consumed count
    assert mock_driver.get_calculator.call_count == 2

    # Optional: consume one more to be sure
    next(stream)
    assert mock_driver.get_calculator.call_count == 3
