from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.domain_models.md import MDConfig
from pyacemaker.domain_models.validation import PhononConfig, ValidationStatus
from pyacemaker.utils.phonons import PhononCalculator


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

def test_phonon_calculator_calculate(mocker: Any, mock_md_config: MDConfig) -> None:
    mock_phonopy_cls = mocker.patch("pyacemaker.utils.phonons.Phonopy")
    mock_phonopy = mock_phonopy_cls.return_value

    mock_sc_atoms = mocker.MagicMock()
    mock_sc_atoms.symbols = ["H"]*2
    mock_sc_atoms.scaled_positions = np.array([[0,0,0], [0.5,0.5,0.5]])
    mock_sc_atoms.cell = np.eye(3)*10.0
    mock_sc_atoms.__len__.return_value = 2

    mock_phonopy.supercells_with_displacements = [mock_sc_atoms]

    mock_run_lammps = mocker.patch("pyacemaker.utils.phonons.run_static_lammps")
    mock_run_lammps.return_value = (0.0, np.zeros((2, 3)), np.zeros(6))

    mock_phonopy.run_mesh = mocker.Mock()
    mock_phonopy.get_mesh_dict.return_value = {"frequencies": np.array([[1.0, 2.0], [1.0, 2.0]])}

    mock_plot = mocker.Mock()
    mock_phonopy.auto_band_structure.return_value = mock_plot

    config = PhononConfig()
    calc = PhononCalculator(config, mock_md_config)

    atoms = Atoms("H", cell=[10,10,10], pbc=True)
    potential_path = Path("dummy.yace")
    output_dir = Path("output")

    result = calc.calculate(atoms, potential_path, output_dir)

    assert result.status == ValidationStatus.PASS
    assert not result.has_imaginary_modes

    mock_run_lammps.assert_called_once()

    mock_phonopy.produce_force_constants.assert_called()
    forces_arg = mock_phonopy.produce_force_constants.call_args[1]['forces']
    assert len(forces_arg) == 1
    assert np.allclose(forces_arg[0], np.zeros((2, 3)))

def test_phonon_calculator_imaginary(mocker: Any, mock_md_config: MDConfig) -> None:
    mock_phonopy_cls = mocker.patch("pyacemaker.utils.phonons.Phonopy")
    mock_phonopy = mock_phonopy_cls.return_value

    mock_sc_atoms = mocker.MagicMock()
    mock_sc_atoms.symbols = ["H"]*2
    mock_sc_atoms.scaled_positions = np.array([[0,0,0], [0.5,0.5,0.5]])
    mock_sc_atoms.cell = np.eye(3)*10.0
    mock_sc_atoms.__len__.return_value = 2
    mock_phonopy.supercells_with_displacements = [mock_sc_atoms]

    mock_run_lammps = mocker.patch("pyacemaker.utils.phonons.run_static_lammps")
    mock_run_lammps.return_value = (0.0, np.zeros((2, 3)), np.zeros(6))

    mock_phonopy.run_mesh = mocker.Mock()
    mock_phonopy.get_mesh_dict.return_value = {"frequencies": np.array([[-1.0, 2.0]])}

    mock_plot = mocker.Mock()
    mock_phonopy.auto_band_structure.return_value = mock_plot

    config = PhononConfig()
    calc = PhononCalculator(config, mock_md_config)
    atoms = Atoms("H", cell=[10,10,10], pbc=True)
    potential_path = Path("dummy.yace")
    output_dir = Path("output")

    result = calc.calculate(atoms, potential_path, output_dir)

    assert result.status == ValidationStatus.FAIL
    assert result.has_imaginary_modes
