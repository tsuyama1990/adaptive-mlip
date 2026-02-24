from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.validation import ElasticConfig, ValidationStatus
from pyacemaker.utils.elastic import ElasticCalculator


@pytest.fixture
def mock_md_config() -> MDConfig:
    return MDConfig(
        thermo_freq=10,
        dump_freq=10,
        n_steps=100,
        timestep=0.001,
        temperature=300.0,
        pressure=1.0,
    )

def test_elastic_calculator_calculate(mocker: Any, mock_md_config: MDConfig) -> None:
    mock_run_lammps = mocker.patch("pyacemaker.utils.elastic.run_static_lammps")

    from ase.units import GPa

    C11 = 100 * GPa
    C12 = 50 * GPa
    C44 = 25 * GPa

    C_true = np.zeros((6, 6))
    C_true[0:3, 0:3] = C12
    np.fill_diagonal(C_true[0:3, 0:3], C11)
    C_true[3:6, 3:6] = np.eye(3) * C44

    delta = 0.01

    def side_effect(atoms: Atoms, potential: Path, config: MDConfig, file_manager: Any = None) -> tuple[float, np.ndarray, np.ndarray]:
        call_count = mock_run_lammps.call_count - 1
        j = call_count // 2
        sign = 1 if call_count % 2 == 0 else -1

        strain = np.zeros(6)
        strain[j] = sign * delta

        stress = np.dot(C_true, strain)

        return 0.0, np.zeros((len(atoms), 3)), stress

    mock_run_lammps.side_effect = side_effect

    config = ElasticConfig(strain_magnitude=delta)
    calc = ElasticCalculator(config, mock_md_config)

    atoms = Atoms("Al", cell=[4,4,4], pbc=True)
    potential_path = Path("dummy.yace")

    result = calc.calculate(atoms, potential_path)

    assert result.status == ValidationStatus.PASS
    assert result.is_mechanically_stable

    c11_val = result.c_ij["C11"]
    assert np.isclose(c11_val, 100.0)

    # B = (C11 + 2C12)/3 = (100 + 100)/3 = 66.66
    assert np.isclose(result.bulk_modulus, 66.666, atol=0.1)

def test_elastic_calculator_unstable(mocker: Any, mock_md_config: MDConfig) -> None:
    mock_run_lammps = mocker.patch("pyacemaker.utils.elastic.run_static_lammps")

    from ase.units import GPa
    C11 = -100 * GPa
    C_true = np.eye(6) * C11

    delta = 0.01

    def side_effect(atoms: Atoms, potential: Path, config: MDConfig, file_manager: Any = None) -> tuple[float, np.ndarray, np.ndarray]:
        call_count = mock_run_lammps.call_count - 1
        j = call_count // 2
        sign = 1 if call_count % 2 == 0 else -1
        strain = np.zeros(6)
        strain[j] = sign * delta
        stress = np.dot(C_true, strain)
        return 0.0, np.zeros((len(atoms), 3)), stress

    mock_run_lammps.side_effect = side_effect

    config = ElasticConfig(strain_magnitude=delta)
    calc = ElasticCalculator(config, mock_md_config)
    atoms = Atoms("Al", cell=[4,4,4], pbc=True)
    potential_path = Path("dummy.yace")

    result = calc.calculate(atoms, potential_path)

    assert result.status == ValidationStatus.FAIL
    assert not result.is_mechanically_stable
