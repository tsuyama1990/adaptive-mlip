from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from pyacemaker.constants import RECIPROCAL_FACTOR
from pyacemaker.domain_models import DFTConfig
from pyacemaker.interfaces.qe_driver import QEDriver


@pytest.fixture
def mock_dft_config() -> DFTConfig:
    return DFTConfig(
        code="pw.x",
        functional="PBE",
        kpoints_density=0.04,
        encut=500.0,
        mixing_beta=0.6,
        smearing_type="mv",
        smearing_width=0.02,
        diagonalization="david",
        pseudopotentials={"H": "H.UPF"},
    )


def test_qe_driver_get_calculator_kpoints(mock_dft_config: DFTConfig) -> None:
    """Test k-point generation based on density."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)

        # Verify k-points passed to Espresso
        call_args = MockEspresso.call_args[1]
        kpts = call_args.get("kpts")
        assert kpts is not None

        assert isinstance(kpts, tuple | list)
        assert len(kpts) == 3

        # Verify calculation:
        # spacing = 0.04, factor = 2*pi / 0.04 ~ 157.08, L = 10
        # N = ceil(157.08 / 10) = ceil(15.7) = 16
        expected_k = int(np.ceil((RECIPROCAL_FACTOR / 0.04) / 10.0))
        assert kpts == (expected_k, expected_k, expected_k)


def test_qe_driver_kpoints_non_pbc(mock_dft_config: DFTConfig) -> None:
    """Test k-point generation for non-periodic systems."""
    # Isolated atom, pbc=False
    atoms = Atoms("H", cell=[10, 10, 10], pbc=[False, False, False])
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kpts = MockEspresso.call_args[1].get("kpts")
        assert kpts == (1, 1, 1)

    # Surface (slab), pbc=[True, True, False]
    atoms = Atoms("H", cell=[10, 10, 10], pbc=[True, True, False])
    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)
        kpts = MockEspresso.call_args[1].get("kpts")

        # kx, ky should be calculated
        expected_k = int(np.ceil((RECIPROCAL_FACTOR / 0.04) / 10.0))
        assert kpts[0] == expected_k
        assert kpts[1] == expected_k
        assert kpts[2] == 1


def test_qe_driver_parameters(mock_dft_config: DFTConfig) -> None:
    """Test that parameters from config are passed to Espresso."""
    atoms = Atoms("H", cell=[10, 10, 10], pbc=True)
    driver = QEDriver()

    with patch("pyacemaker.interfaces.qe_driver.Espresso") as MockEspresso:
        driver.get_calculator(atoms, mock_dft_config)

        kwargs = MockEspresso.call_args[1]
        input_data = kwargs.get("input_data", {})

        # Comprehensive check of all parameters
        control = input_data.get("control", {})
        system = input_data.get("system", {})
        electrons = input_data.get("electrons", {})

        assert control["calculation"] == "scf"
        assert control["restart_mode"] == "from_scratch"
        assert control["disk_io"] == "low"

        assert system["ecutwfc"] == 500.0
        assert system["occupations"] == "smearing"
        assert system["smearing"] == "mv"
        assert system["degauss"] == 0.02

        assert electrons["mixing_beta"] == 0.6
        assert electrons["diagonalization"] == "david"
        assert electrons["conv_thr"] == 1.0e-8

        # Check pseudopotentials
        pseudos = kwargs.get("pseudopotentials")
        assert pseudos == {"H": "H.UPF"}
