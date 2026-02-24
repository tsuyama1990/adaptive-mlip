from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.utils.lammps import run_static_lammps


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

def test_run_static_lammps_mocked(mocker: Any, mock_md_config: MDConfig) -> None:
    mock_driver_cls = mocker.patch("pyacemaker.utils.lammps.LammpsDriver")
    mock_driver = mock_driver_cls.return_value

    mock_driver.extract_variable.side_effect = [
        -10.0, # pe
        100.0, # pxx
        100.0, # pyy
        100.0, # pzz
        0.0,   # pyz
        0.0,   # pxz
        0.0,   # pxy
    ]

    mock_lmp = mock_driver.lmp
    mock_lmp.get_natoms.return_value = 2

    mock_as_array = mocker.patch("numpy.ctypeslib.as_array")

    def side_effect_as_array(ptr: Any, shape: tuple[int, ...]) -> np.ndarray:
        if shape == (2, 3):
            return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        if shape == (2,):
            return np.array([1, 2])
        return np.zeros(shape)

    mock_as_array.side_effect = side_effect_as_array

    atoms = Atoms("H2", positions=[[0,0,0], [1,0,0]])
    potential_path = Path("dummy.yace")
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.resolve", return_value=potential_path)

    mock_fm = mocker.Mock()
    mock_fm.prepare_workspace.return_value = (mocker.MagicMock(), Path("data.lmp"), Path("dump.lmp"), Path("log.lmp"), ["H"])

    energy, forces, stress = run_static_lammps(atoms, potential_path, mock_md_config, mock_fm)

    assert energy == -10.0
    assert np.allclose(forces, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    from ase import units
    assert np.allclose(stress[:3], -100.0 * units.bar)

    mock_driver.run.assert_called()
    assert mock_driver.lmp.close.called
